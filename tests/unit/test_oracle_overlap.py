from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from streamnav.envs.habitat_client import VectorHabitatEnvs
from streamnav.errors import OracleUnavailableError


class OracleClient:
    def __init__(self, index, barrier):
        self.index = index
        self.barrier = barrier

    def get_oracle_supervision(self):
        # Requiring all clients to enter catches accidental serial dispatch.
        self.barrier.wait(timeout=5)
        if self.index == 1:
            raise OracleUnavailableError("test: unreachable goal")
        return {"action": self.index, "perception": {"episode_id": str(self.index)}}


def test_async_oracle_fanout_preserves_order_and_unavailable_slots():
    envs = VectorHabitatEnvs.__new__(VectorHabitatEnvs)
    envs.clients = [OracleClient(i, Barrier(1)) for i in range(3)]
    with ThreadPoolExecutor(max_workers=3) as executor:
        envs.executor = executor
        barrier = Barrier(4)
        for client in envs.clients:
            client.barrier = barrier
        pending = envs.submit_oracle_supervisions()
        # Submission returns while RPCs are still in flight, allowing vision
        # encoding to run before waiting for all labels.
        assert not any(future.done() for future in pending)
        barrier.wait(timeout=5)
        results = [future.result() for future in pending]
        assert results[0]["action"] == 0
        assert isinstance(results[1], OracleUnavailableError)
        assert results[2]["action"] == 2
        barrier = Barrier(3)
        for client in envs.clients:
            client.barrier = barrier
        blocking = envs.get_oracle_supervisions()
        assert blocking[0] == results[0]
        assert isinstance(blocking[1], OracleUnavailableError)
        assert blocking[2] == results[2]
