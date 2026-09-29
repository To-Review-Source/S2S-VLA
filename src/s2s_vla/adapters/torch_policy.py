import numpy as np
import torch
from s2s_vla.interfaces import Proposal


class FrozenTorchPolicy:
    def __init__(self, model, method, encode, sample, device="cpu"):
        self.model = model.to(device).eval()
        self.model.requires_grad_(False)
        self.method = method
        self.encode = encode
        self.sample = sample
        self.device = device

    def propose(self, context, count, rng):
        seed = int(rng.integers(0, 2**63 - 1))
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.inference_mode():
            features, cache = self.encode(self.model, context)
            chunks = self.sample(self.model, context, cache, count, self.method.horizon, generator)
        if isinstance(features, torch.Tensor):
            features = features.detach().float().cpu().numpy()
        if isinstance(chunks, torch.Tensor):
            chunks = chunks.detach().float().cpu().numpy()
        proposal = Proposal(np.asarray(features), np.asarray(chunks))
        proposal.validate(self.method, count)
        return proposal
