import torch.nn as nn


class SymmetricLoss(nn.Module):
	def __init__(self, delta=0.2):
		super().__init__()
		assert 0.0 <= delta <= 1.0, "Delta must be between 0 and 1"
		self.nll = nn.NLLLoss()
		self.kl = nn.KLDivLoss(reduction="batchmean", log_target=True)
		self.delta = delta

	def forward(self, output, mirror_output, target):
		return (1 - self.delta) * self.nll(output, target) + self.delta * (
			self.kl(output, mirror_output) + self.kl(mirror_output, output)
		) / 2


class OwnNLLLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.nll = nn.NLLLoss()

	def forward(self, output, mirror_output, target):
		return self.nll(output, target)
