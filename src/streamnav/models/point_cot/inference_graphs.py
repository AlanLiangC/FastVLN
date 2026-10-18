"""Bounded CUDA graph cache for independent, non-differentiable frame forwards."""

from collections import OrderedDict
from dataclasses import fields, is_dataclass, replace

import torch


def map_tensors(value, transform):
    if isinstance(value, torch.Tensor):
        return transform(value)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value, **{f.name: map_tensors(getattr(value, f.name), transform) for f in fields(value)}
        )
    if isinstance(value, tuple):
        return tuple(map_tensors(x, transform) for x in value)
    if isinstance(value, list):
        return [map_tensors(x, transform) for x in value]
    if isinstance(value, dict):
        return {k: map_tensors(v, transform) for k, v in value.items()}
    return value


def graph_signature(value):
    if isinstance(value, torch.Tensor):
        return (
            value.device,
            value.dtype,
            tuple(value.shape),
            tuple(value.stride()),
            tuple(value.flatten().tolist()) if value.device.type == "cpu" else None,
        )
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value),
            tuple((f.name, graph_signature(getattr(value, f.name))) for f in fields(value)),
        )
    if isinstance(value, (tuple, list)):
        return tuple(graph_signature(x) for x in value)
    if isinstance(value, dict):
        return tuple((k, graph_signature(v)) for k, v in value.items())
    return value


class InferenceGraphs:
    def __init__(self, max_entries=16):
        self.max_entries = max_entries
        self.entries: OrderedDict[tuple, tuple] = OrderedDict()
        self.hits = self.misses = 0

    def __call__(self, function, args, kwargs, tag=None):
        if torch.is_grad_enabled():
            raise RuntimeError("CUDA inference graphs cannot replace differentiable replay")
        key = (tag, graph_signature((args, kwargs)))
        if key not in self.entries:
            # Release the oldest graph before allocating a replacement's inputs,
            # workspace and private pool; never hold max_entries + 1 pools.
            if len(self.entries) >= self.max_entries:
                self.entries.popitem(last=False)
            self.misses += 1
            static_args, static_kwargs = map_tensors((args, kwargs), lambda t: t.clone())
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    function(*static_args, **static_kwargs)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            from fla import utils as fla_utils

            # FLA caches just the most recent sequence-index tensor. An index
            # borrowed from warmup can be freed when another shape runs, while
            # graph nodes still hold its raw address. Build these indices inside
            # capture so their allocations belong to this graph's private pool.
            previous_cache_flag = fla_utils.FLA_DISABLE_TENSOR_CACHE
            fla_utils.FLA_DISABLE_TENSOR_CACHE = True
            try:
                with torch.cuda.graph(graph, stream=stream):
                    outputs = function(*static_args, **static_kwargs)
            finally:
                fla_utils.FLA_DISABLE_TENSOR_CACHE = previous_cache_flag
            self.entries[key] = (graph, static_args, static_kwargs, outputs)
        else:
            self.hits += 1
            self.entries.move_to_end(key)
        graph, static_args, static_kwargs, outputs = self.entries[key]
        incoming, destinations = [], []
        map_tensors((args, kwargs), lambda t: incoming.append(t))
        map_tensors((static_args, static_kwargs), lambda t: destinations.append(t))
        for source, destination in zip(incoming, destinations, strict=True):
            if source.device.type == "cuda":
                destination.copy_(source)
        graph.replay()
        # Cache outputs are overwritten by the next replay. Every environment's
        # returned state must own its storage, including states retained for BPTT.
        return map_tensors(outputs, lambda t: t.clone())
