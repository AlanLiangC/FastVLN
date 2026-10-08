from torch.distributions import Categorical


class ObjectNavActionDistribution:
    def build(self, logits):
        if logits.shape[-1] != 4:
            raise ValueError("ObjectNav requires four action logits")
        return Categorical(logits=logits.float())
