from torch.distributions import Categorical


class ObjectNavActionDistribution:
    def build(self, logits):
        if logits.shape[-1] not in (4, 6):
            raise ValueError("ObjectNav requires six action logits (four for legacy checkpoints)")
        return Categorical(logits=logits.float())
