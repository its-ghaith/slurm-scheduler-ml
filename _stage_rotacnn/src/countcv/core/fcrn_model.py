import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50


class ModelFactory:
	"""Builds a class object which takes only two arguments num_classes and input_channels.
	all other arguments are fixed."""

	def __init__(self, cls, **preset_args):
		self.cls = cls
		self.preset_args = preset_args
		self.__name__ = cls.__name__

	def __call__(self, input_channels: int, output_channels: int) -> torch.nn.Module:
		return self.cls(input_channels=input_channels, output_channels=output_channels, **self.preset_args)

	def __repr__(self):
		return self.__name__


class BaseModel(torch.nn.Module):
	def __init__(self, input_channels: int, tile_size: int, output_channels: int = 1, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.input_channels: int = input_channels
		self.output_channels = output_channels
		self.tile_size: int = tile_size

	def _create_gaussian_weight_mask(self, device: torch.device, sigma: float = 0.3) -> torch.Tensor:
		"""Create a Gaussian weight mask centered on the patch."""
		center = self.tile_size / 2
		y, x = torch.meshgrid(
			torch.arange(self.tile_size, device=device),
			torch.arange(self.tile_size, device=device),
			indexing="ij",
		)

		# Distance from center
		dist = torch.sqrt((x - center + 0.5) ** 2 + (y - center + 0.5) ** 2)
		max_dist = center * np.sqrt(2)

		# Gaussian falloff
		weight_mask = torch.exp(-(dist**2) / (2 * (sigma * max_dist) ** 2))

		return weight_mask.view(1, 1, self.tile_size, self.tile_size)

	def predict(self, imgs: torch.Tensor, device: torch.device) -> torch.Tensor:
		if self.tile_size is None:
			return self.forward(imgs)

		B, C, H, W = imgs.shape
		stride = max(1, int(self.tile_size * 0.75))  # 25% overlap
		weight_mask = self._create_gaussian_weight_mask(device=device)

		output = torch.zeros((B, self.output_channels, H, W), device=device)
		weight_sum = torch.zeros_like(output, device=device)

		# Compute tile starting positions
		ys = list(range(0, max(H - self.tile_size + 1, 1), stride))
		xs = list(range(0, max(W - self.tile_size + 1, 1), stride))

		# Ensure we cover the entire image
		if ys and ys[-1] != H - self.tile_size:
			ys.append(H - self.tile_size)
		if xs and xs[-1] != W - self.tile_size:
			xs.append(W - self.tile_size)

		with torch.no_grad():
			for y in ys:
				for x in xs:
					patch = imgs[:, :, y : y + self.tile_size, x : x + self.tile_size]
					ph, pw = patch.shape[2], patch.shape[3]

					# Pad edge tiles if needed (should be rare with proper preprocessing)
					if ph < self.tile_size or pw < self.tile_size:
						patch = F.pad(patch, (0, self.tile_size - pw, 0, self.tile_size - ph))

					pred = self.forward(patch)
					pred = pred[:, :, :ph, :pw]

					# Get corresponding weight mask portion
					weights = weight_mask[:, :, :ph, :pw]

					# Accumulate weighted predictions
					output[:, :, y : y + ph, x : x + pw] += pred * weights
					weight_sum[:, :, y : y + ph, x : x + pw] += weights

		# Normalize by total weights
		output = output / torch.clamp(weight_sum, min=1e-8)
		return output


class DensityMapLoss(nn.Module):
	"""
	Combines pixel-wise MSE with a count-level MAE term for density map regression.

	L = MSE(pred, target) + lambda_count * MAE(sum(pred), sum(target))
	"""

	def __init__(self, alpha: float, lambda_count: float = 0.1):
		super().__init__()
		self.lambda_count: float = lambda_count
		self.alpha: float = alpha
		self.mse = nn.MSELoss()
		self.mae = nn.L1Loss()

	def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			pred: (B, 1, H, W) predicted density maps
			target: (B, 1, H, W) ground truth density maps
		Returns:
			Combined loss scalar
		"""
		# Pixel-wise MSE
		loss_pixel = self.mse(pred, target)

		# Count-level MAE: sum over H and W per image
		pred_count = pred.view(pred.size(0), -1).sum(dim=1)
		target_count = target.view(target.size(0), -1).sum(dim=1)
		loss_count = self.mae(pred_count, target_count) / self.alpha

		# Combined loss
		loss = loss_pixel + self.lambda_count * loss_count
		return loss


class QuantileDensityMapLoss(nn.Module):
	"""
	Computes pinball loss for multiple quantile levels simultaneously.
	Combines pixel-wise pinball loss with a count-level pinball loss term
	for quantile regression on density maps.

	L_q = PinballLoss(pred_q, target, tau_q) + lambda_count * PinballLoss(sum(pred_q), sum(target), tau_q)

	Returns a loss tensor of shape (B, Q) where B is batch size and Q is number of quantiles.
	"""

	def __init__(
		self,
		alpha: float,
		tau: float | list[float] = 0.5,
		lambda_count: float = 0.1,
	):
		"""
		Args:
			tau: Quantile level(s) in (0, 1). Can be a single float or list of floats.
				tau=0.5 gives median regression.
			lambda_count: Weight for the count-level loss term.
		"""
		super().__init__()
		self.tau = torch.tensor([tau] if isinstance(tau, (float | int)) else tau)
		self.lambda_count = lambda_count
		self.alpha = alpha

	def pinball_loss(self, pred: torch.Tensor, target: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
		"""
		Compute pinball loss for quantile regression.

		Args:
			pred: (B, Q, ...) Predicted values for Q quantiles
			target: (B, 1, ...) Ground truth values (broadcasted across quantiles)
			tau: (Q,) Quantile levels
		Returns:
			(B, Q) loss per batch item per quantile
		"""
		# Ensure tau is on the same device as pred
		tau = tau.to(pred.device)

		# Reshape tau for broadcasting: (1, Q, 1, ...)
		tau_shape = [1, len(tau)] + [1] * (pred.ndim - 2)
		tau = tau.view(*tau_shape)

		# Compute errors: (B, Q, ...)
		errors = target - pred

		# Pinball loss: max(tau * errors, (tau - 1) * errors)
		loss = torch.max(tau * errors, (tau - 1) * errors)

		reduce_dims = list(range(2, loss.ndim))
		loss = loss.mean(dim=reduce_dims)

		return loss

	def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		"""
		Args:
		pred: (B, Q, H, W) predicted density maps for Q quantiles
		target: (B, 1, H, W) ground truth density maps

		Returns:
		(B, Q) loss tensor - loss per batch item per quantile
		"""
		B, Q = pred.size(0), pred.size(1)

		# Ensure we have the right number of quantiles
		assert len(self.tau) == Q, f"Expected {len(self.tau)} quantiles but got {Q}"

		# Pixel-wise pinball loss: (B, Q)
		loss_pixel = self.pinball_loss(pred, target, self.tau)

		# Count-level pinball loss
		# Sum over H and W: (B, Q)
		pred_count = pred.view(B, Q, -1).sum(dim=2)  # (B, Q)
		target_count = target.view(B, 1, -1).sum(dim=2)  # (B, 1)

		# Compute pinball loss on counts: (B, Q)
		loss_count = (
			self.pinball_loss(
				pred_count.unsqueeze(-1),  # (B, Q, 1) for broadcasting
				target_count.unsqueeze(-1),  # (B, 1, 1)
				self.tau,
			).squeeze(-1)
			/ self.alpha
		)  # Back to (B, Q)

		# Combined loss: (B, Q)
		loss = loss_pixel + self.lambda_count * loss_count

		if Q == 3:
			loss = loss.mean(dim=0)
			weights = torch.tensor([0.2, 0.6, 0.2], device=pred.device)
			loss = (loss * weights).sum()
		else:
			loss = loss.mean()
		return loss


# FCRN as in the paper


def conv_block(in_channels: int, out_channels: int, kernel_size=3, pool_size=2):
	return torch.nn.Sequential(
		torch.nn.Conv2d(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=kernel_size,
			padding=kernel_size // 2,
		),
		torch.nn.BatchNorm2d(out_channels),
		torch.nn.ReLU(inplace=True),
		torch.nn.MaxPool2d(kernel_size=pool_size),
	)


def deconv_block(in_channels: int, out_channels: int):
	return torch.nn.Sequential(
		torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
		torch.nn.BatchNorm2d(in_channels),
		torch.nn.ReLU(inplace=True),
		torch.nn.Conv2d(
			in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=3,
			padding=1,
		),
	)


class FCRNBase(BaseModel):
	"""Simple FCRN-A style model according to https://github.com/WeidiXie/cell_counting_v2"""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.encoder = torch.nn.Sequential(
			conv_block(self.input_channels, 32),
			conv_block(32, 64),
			conv_block(64, 128),
		)
		self.fc = conv_block(128, 256, kernel_size=1, pool_size=1)
		self.decoder = torch.nn.Sequential(
			deconv_block(256, 128),
			deconv_block(128, 64),
			deconv_block(64, 32),
		)
		self.head = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=32, out_channels=self.output_channels, kernel_size=1, bias=False),
			# torch.nn.ReLU(),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.encoder(x)
		x = self.fc(x)
		x = self.decoder(x)
		return self.head(x)


# FCRN with skip connections
class FCRNSkip(BaseModel):
	"""Own experiment: Added skip connections to the FCRN-A model"""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.enc1 = conv_block(self.input_channels, 32)  # In: bx3x256x256
		self.enc2 = conv_block(32, 64)  # bx32x128x128
		self.enc3 = conv_block(64, 128)  # bx64x64x64
		self.fc = torch.nn.Sequential(
			torch.nn.Conv2d(
				in_channels=128,
				out_channels=256,
				kernel_size=1,
				padding=0,
			),
			torch.nn.BatchNorm2d(256),
			torch.nn.ReLU(inplace=True),
		)  # bx128x32x32

		self.dec1 = deconv_block(256 + 128, 128)  # bx256x32x32
		self.dec2 = deconv_block(128 + 64, 64)  # bx128x64x64
		self.dec3 = deconv_block(64 + 32, 32)  # bx64x128x128
		self.head = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=32, out_channels=self.output_channels, kernel_size=1, bias=False),
			torch.nn.ReLU(),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		enc1 = self.enc1(x)
		enc2 = self.enc2(enc1)
		enc3 = self.enc3(enc2)
		enc4 = self.fc(enc3)
		dec1 = torch.cat([enc3, enc4], dim=1)
		dec1 = self.dec1(dec1)
		dec2 = torch.cat([dec1, enc2], dim=1)
		dec2 = self.dec2(dec2)
		dec3 = torch.cat([dec2, enc1], dim=1)
		dec3 = self.dec3(dec3)
		return self.head(dec3)


# SAUnet
class SelfAttention2d(nn.Module):
	"""Self attention block for the Self Attention Unet https://pmc.ncbi.nlm.nih.gov/articles/PMC8924707/#R8"""

	def __init__(self, in_channels, out_channels, dropout=0.1):
		super().__init__()
		self.out_channels = out_channels
		self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
		self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
		self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

		self.dropout = nn.Dropout(dropout)
		self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

		self.gamma = nn.Parameter(torch.zeros(1))
		self.res_conv = (
			nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
			if in_channels != out_channels
			else torch.nn.Identity()
		)

	def forward(self, x):
		B, C, H, W = x.size()

		proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # B x HW x C'
		proj_key = self.key_conv(x).view(B, -1, H * W)  # B x C' x HW
		attention = torch.bmm(proj_query, proj_key)  # B x HW x HW
		attention = F.softmax(attention, dim=-1)
		attention = self.dropout(attention)

		proj_value = self.value_conv(x).view(B, -1, H * W)  # B x C x HW
		out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x HW
		out = out.view(B, self.out_channels, H, W)

		out = self.out_conv(out)  # 1×1 conv before residual connection

		out = self.gamma * out + self.res_conv(x)  # Residual connection
		return out


class SAUnet(BaseModel):
	"""Unet with optional self attention block at the bottom. Directly adapted from
	https://github.com/mzlr/sau-net"""

	def __init__(self, self_attention: bool = True, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.pool = torch.nn.MaxPool2d(kernel_size=2)
		self.enc1 = self.enc_block(self.input_channels, 32)  # In: bx3x256x256
		self.enc2 = self.enc_block(32, 64)  # bx32x128x128
		self.enc3 = self.enc_block(64, 128)  # bx64x64x64
		if self_attention:
			self.self_attention = SelfAttention2d(128, 256)
		else:
			self.self_attention = self.enc_block(128, 256, kernel_size=1)
		self.upconv1 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False)
		self.dec1 = self.enc_block(128 + 128, 128)
		self.upconv2 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False)
		self.dec2 = self.enc_block(64 + 64, 64)  # bx128x64x64
		self.upconv3 = torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, bias=False)
		self.dec3 = self.enc_block(32 + 32, 32)  # bx64x128x128
		self.head = torch.nn.Conv2d(in_channels=32, out_channels=self.output_channels, kernel_size=1, bias=False)

	@staticmethod
	def enc_block(in_channels: int, out_channels: int, kernel_size=3, dilation=1):
		padding = dilation * (kernel_size // 2)
		return torch.nn.Sequential(
			torch.nn.Conv2d(
				in_channels,
				out_channels,
				kernel_size=kernel_size,
				padding=padding,
				bias=False,
				dilation=dilation,
			),
			torch.nn.BatchNorm2d(out_channels),
			torch.nn.ReLU(inplace=True),
			torch.nn.Conv2d(
				out_channels,
				out_channels,
				kernel_size=kernel_size,
				padding=padding,
				bias=False,
				dilation=dilation,
			),
			torch.nn.BatchNorm2d(out_channels),
			torch.nn.ReLU(inplace=True),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		e1 = self.enc1(x)
		e2 = self.enc2(self.pool(e1))
		e3 = self.enc3(self.pool(e2))
		b = self.self_attention(self.pool(e3))
		b = self.upconv1(b)
		d1 = torch.cat([b, e3], dim=1)
		d1 = self.dec1(d1)

		d1 = self.upconv2(d1)
		d2 = torch.cat([d1, e2], dim=1)
		d2 = self.dec2(d2)

		d2 = self.upconv3(d2)
		d3 = torch.cat([d2, e1], dim=1)
		d3 = self.dec3(d3)
		return self.head(d3)


# Own attempt at SOTA model


class ResidualEncBlock(torch.nn.Module):
	"""Memory-efficient 2D self-attention block with residual connection."""

	def __init__(self, in_channels: int, out_channels: int, kernel_size=3, dilation=1, groups=8):
		super().__init__()
		padding = dilation * (kernel_size // 2)

		self.conv = torch.nn.Sequential(
			torch.nn.Conv2d(
				in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False, dilation=dilation
			),
			torch.nn.GroupNorm(groups, out_channels),
			torch.nn.ReLU(inplace=True),
			torch.nn.Conv2d(
				out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False, dilation=dilation
			),
			torch.nn.GroupNorm(groups, out_channels),
		)

		# shortcut path (identity if channels match, 1x1 conv if not)
		self.skip = (
			nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
			if in_channels != out_channels
			else nn.Identity()
		)

		self.relu = torch.nn.ReLU(inplace=True)

	def forward(self, x):
		out = self.conv(x)
		skip = self.skip(x)
		return self.relu(out + skip)


class ASPP(nn.Module):
	"""Atrous Spatial Pyramid Pooling."""

	def __init__(self, in_channels, out_channels, atrous_rates=(1, 2, 3), groups=8):
		super().__init__()
		self.blocks = nn.ModuleList()
		self.blocks.append(
			nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
				nn.GroupNorm(groups, out_channels),
				nn.ReLU(inplace=True),
			)
		)
		for rate in atrous_rates:
			self.blocks.append(
				nn.Sequential(
					nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
					nn.GroupNorm(groups, out_channels),
					nn.ReLU(inplace=True),
				)
			)
		self.global_pool = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
			nn.GroupNorm(groups, out_channels),
			nn.ReLU(inplace=True),
		)
		self.project = nn.Sequential(
			nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, kernel_size=1, bias=False),
			nn.GroupNorm(groups, out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		size = x.shape[2:]
		res = [block(x) for block in self.blocks]
		global_feat = self.global_pool(x)
		global_feat = F.interpolate(global_feat, size=size, mode="bilinear", align_corners=True)
		res.append(global_feat)
		x = torch.cat(res, dim=1)
		return self.project(x)


# -----------------------------------
# Attention Gate
# -----------------------------------
class AttentionGate(nn.Module):
	"""Attention gate for U-Net skip connections."""

	def __init__(self, in_channels, gating_channels, inter_channels):
		super().__init__()
		self.theta = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
		self.phi = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, bias=False)
		self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x, g):
		theta_x = self.theta(x)
		phi_g = self.phi(g)
		f = self.relu(theta_x + phi_g)
		psi = self.sigmoid(self.psi(f))
		return x * psi


# -----------------------------------
# 4-block SOTA-inspired U-Net
# -----------------------------------
class SOTAnet4(BaseModel):
	"""4-block attention U-Net with ASPP, self-attention bottleneck, and dense decoder connections with proper channel
	preservation. Own creation, combines a number of sensible blocks that have been built since 2016."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.pool = nn.MaxPool2d(2)
		self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

		# Encoder
		self.enc1 = ResidualEncBlock(self.input_channels, 32)
		self.enc2 = ResidualEncBlock(32, 64, groups=16)
		self.enc3 = ResidualEncBlock(64, 128, groups=32)
		self.enc4 = ResidualEncBlock(128, 256, groups=32)

		# Bottleneck
		self.bottleneck_att = SelfAttention2d(256, 512, dropout=0.1)
		self.bottleneck_aspp = ASPP(512, 512, groups=32)

		# Attention gates
		self.att4 = AttentionGate(256, 512, 128)
		self.att3 = AttentionGate(128, 256, 64)
		self.att2 = AttentionGate(64, 128, 32)

		# 1x1 convs to reduce channels before dense concatenation (preserve main signal)
		# High-level decoder features keep more channels
		self.conv_g4 = nn.Conv2d(512, 256, 1)
		self.conv_g3 = nn.Conv2d(256, 128, 1)
		self.conv_g2 = nn.Conv2d(128, 64, 1)
		self.conv_g1 = nn.Conv2d(64, 32, 1)
		# Encoder skips reduced more aggressively
		self.conv_e4 = nn.Conv2d(256, 128, 1)
		self.conv_e3 = nn.Conv2d(128, 64, 1)
		self.conv_e2 = nn.Conv2d(64, 32, 1)
		self.conv_e1 = nn.Conv2d(32, 16, 1)

		# Decoder
		self.dec4 = ResidualEncBlock(256 + 128, 256, groups=32)  # g4 + e4_att (384)
		self.dec3 = ResidualEncBlock(128 + 64 + 256, 128, groups=32)  # g3 + e3_att + g4_up (448)
		self.dec2 = ResidualEncBlock(64 + 32 + 128, 64, groups=16)  # g2 + e2_att + g3_up (224)
		self.dec1 = ResidualEncBlock(32 + 16 + 64, 32)  # g1 + e1 + g2_up (112)

		# Output
		self.head = nn.Conv2d(32, self.output_channels, kernel_size=1)

	def forward(self, x):
		# Encoder
		e1 = self.enc1(x)  # 32 (h,w)
		e2 = self.enc2(self.pool(e1))  # 64 (h/2,w/2)
		e3 = self.enc3(self.pool(e2))  # 128 (h/4,w/4)
		e4 = self.enc4(self.pool(e3))  # 256 (h/8, w/8)

		# Bottleneck
		b = self.bottleneck_att(self.pool(e4))  # 512 (h/16, w/16)
		b = self.bottleneck_aspp(b)  # 512

		# Decoder 4
		g4 = self.upsample(b)  # 512
		e4_att = self.att4(e4, g4)
		d4 = self.dec4(torch.cat([self.conv_g4(g4), self.conv_e4(e4_att)], dim=1))

		# Decoder 3
		g3 = self.upsample(d4)
		e3_att = self.att3(e3, g3)
		g4_up = F.interpolate(g4, size=g3.shape[2:], mode="bilinear", align_corners=True)
		d3 = self.dec3(torch.cat([self.conv_g3(g3), self.conv_e3(e3_att), self.conv_g4(g4_up)], dim=1))

		# Decoder 2
		g2 = self.upsample(d3)
		e2_att = self.att2(e2, g2)
		g3_up = F.interpolate(g3, size=g2.shape[2:], mode="bilinear", align_corners=True)
		d2 = self.dec2(torch.cat([self.conv_g2(g2), self.conv_e2(e2_att), self.conv_g3(g3_up)], dim=1))

		# Decoder 1
		g1 = self.upsample(d2)
		g2_up = F.interpolate(g2, size=g1.shape[2:], mode="bilinear", align_corners=True)
		d1 = self.dec1(
			torch.cat(
				[self.conv_g1(g1), self.conv_e1(e1), self.conv_g2(g2_up)],
				dim=1,
			)
		)

		return self.head(d1)


class DeeplabV3(BaseModel):
	"""Implementation of the classical segmentation algorithm https://arxiv.org/abs/1706.05587
	with adapted final classifier layer to result in a density map. Idea not directly from a paper
	but losely connected to https://github.com/cvlab-stonybrook/LearningToCountEverything
	"""

	def __init__(self, freeze_backbone=True, unfreeze_layer4=False, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model = deeplabv3_resnet50(weights="COCO_WITH_VOC_LABELS_V1")
		self.model.classifier[4] = nn.Conv2d(256, self.output_channels, kernel_size=1)

		# Optionally adapt first convolution for non-RGB input
		if self.input_channels != 3:
			conv1 = self.model.backbone.conv1
			self.model.backbone.conv1 = nn.Conv2d(
				self.input_channels,
				conv1.out_channels,
				kernel_size=conv1.kernel_size,
				stride=conv1.stride,
				padding=conv1.padding,
				bias=False,
			)
			with torch.no_grad():
				self.model.backbone.conv1.weight[:] = conv1.weight.mean(dim=1, keepdim=True)

		# Freeze layers if requested
		if freeze_backbone:
			for param in self.model.backbone.parameters():
				param.requires_grad = False

			if unfreeze_layer4:
				for param in self.model.backbone.layer4.parameters():
					param.requires_grad = True

	def forward(self, x):
		return self.model(x)["out"]


class LightConvNeXtDecoder(nn.Module):
	def __init__(self, encoder_channels=(128, 256, 512, 1024), out_channels=1):
		super().__init__()
		decoder_channels = (int(encoder_channels[2] / 2), int(encoder_channels[1] / 2), int(encoder_channels[0] / 2))
		# progressively reduce depth
		self.conv3 = nn.Conv2d(encoder_channels[3], decoder_channels[0], kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(decoder_channels[0] + encoder_channels[2], decoder_channels[1], kernel_size=3, padding=1)
		self.conv1 = nn.Conv2d(decoder_channels[1] + encoder_channels[1], decoder_channels[2], kernel_size=3, padding=1)
		self.conv0 = nn.Conv2d(decoder_channels[2] + encoder_channels[0], out_channels, kernel_size=1)

	def forward(self, features):
		f0, f1, f2, f3 = features

		# Deepest feature (8×16)
		x = F.relu(self.conv3(f3))
		x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # 16×32
		x = torch.cat([x, f2], dim=1)

		# Second (16×32 → 32×64)
		x = F.relu(self.conv2(x))
		x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
		x = torch.cat([x, f1], dim=1)

		# Third (32×64 → 64×128)
		x = F.relu(self.conv1(x))
		x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
		x = torch.cat([x, f0], dim=1)

		# Final upsample to input resolution (×4: 64×128 → 256×512)
		x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
		out = self.conv0(x)
		return out
