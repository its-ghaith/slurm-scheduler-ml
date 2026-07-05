import escnn
import numpy as np
import torch
from escnn import gspaces
from escnn.nn import GeometricTensor


class SimpleCNN(torch.nn.Module):
	"""
	A simple Convolutional Neural Network for classification.
	"""

	def __init__(self, num_classes: int = 10, input_channels=1):
		super().__init__()
		self.features = torch.nn.Sequential(
			torch.nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
			torch.nn.ReLU(inplace=True),
			torch.nn.MaxPool2d(2, 2),  # 14x14
			torch.nn.Conv2d(32, 128, kernel_size=3, padding=1),
			torch.nn.ReLU(inplace=True),
			torch.nn.MaxPool2d(2, 2),  # 7x7
			torch.nn.Conv2d(128, 512, kernel_size=3, padding=1),
			torch.nn.ReLU(inplace=True),
			torch.nn.AdaptiveAvgPool2d((1, 1)),  # 512x1x1
		)
		self.classifier = torch.nn.Sequential(
			torch.nn.Linear(512, 128),
			torch.nn.ReLU(inplace=True),
			torch.nn.Dropout(0.25),
			torch.nn.Linear(128, num_classes),
			torch.nn.LogSoftmax(dim=1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x


class FlipEquiCNN(torch.nn.Module):
	"""
	A simple Convolutional Neural Network for classification.
	"""

	def __init__(self, num_classes: int = 10, input_channels=1):
		super().__init__()
		self.r2_act = gspaces.flip2dOnR2(np.pi / 2)  # reflection along vertical axis
		self.input_type = escnn.nn.FieldType(self.r2_act, input_channels * [self.r2_act.trivial_repr])
		self.out_type1 = escnn.nn.FieldType(
			self.r2_act, 32 * [self.r2_act.regular_repr]
		)  # ->  32 * 2-dimensional equivariant features
		self.out_type2 = escnn.nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr])
		self.out_type3 = escnn.nn.FieldType(self.r2_act, 512 * [self.r2_act.regular_repr])
		self.final_type = escnn.nn.FieldType(self.r2_act, 512 * [self.r2_act.trivial_repr])
		self.block1 = escnn.nn.SequentialModule(
			escnn.nn.R2Conv(self.input_type, self.out_type1, kernel_size=3, padding=1),
			escnn.nn.ReLU(self.out_type1, inplace=True),  # pointwise relu to preserve equivariance
			escnn.nn.PointwiseMaxPool(self.out_type1, kernel_size=2, stride=2),  # pointwise to preserve equivariance
		)

		self.block2 = escnn.nn.SequentialModule(
			escnn.nn.R2Conv(self.out_type1, self.out_type2, kernel_size=3, padding=1),
			escnn.nn.ReLU(self.out_type2, inplace=True),
			escnn.nn.PointwiseMaxPool(self.out_type2, kernel_size=2, stride=2),
		)

		self.block3 = escnn.nn.SequentialModule(
			escnn.nn.R2Conv(self.out_type2, self.out_type3, kernel_size=3, padding=1),
			escnn.nn.ReLU(self.out_type3, inplace=True),
			escnn.nn.PointwiseAdaptiveAvgPool(self.out_type3, output_size=1),
		)

		self.gpool = escnn.nn.GroupPooling(self.out_type3)  # outputs a trivial representation
		self.classifier = torch.nn.Sequential(
			torch.nn.Linear(512, 128),
			torch.nn.ReLU(inplace=True),
			torch.nn.Dropout(0.25),
			torch.nn.Linear(128, num_classes),
			torch.nn.LogSoftmax(dim=1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x_geo = GeometricTensor(x, self.input_type)
		x_geo = self.block1(x_geo)
		x_geo = self.block2(x_geo)
		x_geo = self.block3(x_geo)
		x_geo = self.gpool(x_geo)  # collapse group dimension (makes it invariant)
		x = x_geo.tensor.view(x_geo.tensor.size(0), -1)
		return self.classifier(x)


# copied from https://github.com/QUVA-Lab/escnn/blob/master/examples/model.ipynb
# changes made: add input_channels to use it for other datasets than nmist; renamed n_classes to num_classes
# added parameter img_width
class C8SteerableCNN(torch.nn.Module):
	def __init__(self, num_classes=10, input_channels=1, img_width=32):
		super().__init__()

		# the model is equivariant under rotations by 45 degrees, modelled by C8
		self.r2_act = gspaces.rot2dOnR2(N=8)

		# the input image is a scalar field, corresponding to the trivial representation
		in_type = escnn.nn.FieldType(self.r2_act, input_channels * [self.r2_act.trivial_repr])

		# we store the input type for wrapping the images into a geometric tensor during the forward pass
		self.input_type = in_type

		# convolution 1
		# first specify the output type of the convolutional layer
		# we choose 24 feature fields, each transforming under the regular representation of C8
		out_type = escnn.nn.FieldType(self.r2_act, 24 * [self.r2_act.regular_repr])
		self.block1 = escnn.nn.SequentialModule(
			escnn.nn.MaskModule(in_type, S=img_width, margin=1),  # changed S to work for other image sizes
			escnn.nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
			escnn.nn.InnerBatchNorm(out_type),
			escnn.nn.ReLU(out_type, inplace=True),
		)

		# convolution 2
		# the old output type is the input type to the next layer
		in_type = self.block1.out_type
		# the output type of the second convolution layer are 48 regular feature fields of C8
		out_type = escnn.nn.FieldType(self.r2_act, 48 * [self.r2_act.regular_repr])
		self.block2 = escnn.nn.SequentialModule(
			escnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),  # type: ignore
			escnn.nn.InnerBatchNorm(out_type),
			escnn.nn.ReLU(out_type, inplace=True),
		)
		self.pool1 = escnn.nn.SequentialModule(escnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2))

		# convolution 3
		# the old output type is the input type to the next layer
		in_type = self.block2.out_type
		# the output type of the third convolution layer are 48 regular feature fields of C8
		out_type = escnn.nn.FieldType(self.r2_act, 48 * [self.r2_act.regular_repr])
		self.block3 = escnn.nn.SequentialModule(
			escnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),  # type: ignore
			escnn.nn.InnerBatchNorm(out_type),
			escnn.nn.ReLU(out_type, inplace=True),
		)

		# convolution 4
		# the old output type is the input type to the next layer
		in_type = self.block3.out_type
		# the output type of the fourth convolution layer are 96 regular feature fields of C8
		out_type = escnn.nn.FieldType(self.r2_act, 96 * [self.r2_act.regular_repr])
		self.block4 = escnn.nn.SequentialModule(
			escnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),  # type: ignore
			escnn.nn.InnerBatchNorm(out_type),
			escnn.nn.ReLU(out_type, inplace=True),
		)
		self.pool2 = escnn.nn.SequentialModule(escnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2))

		# convolution 5
		# the old output type is the input type to the next layer
		in_type = self.block4.out_type
		# the output type of the fifth convolution layer are 96 regular feature fields of C8
		out_type = escnn.nn.FieldType(self.r2_act, 96 * [self.r2_act.regular_repr])
		self.block5 = escnn.nn.SequentialModule(
			escnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),  # type: ignore
			escnn.nn.InnerBatchNorm(out_type),
			escnn.nn.ReLU(out_type, inplace=True),
		)

		# convolution 6
		# the old output type is the input type to the next layer
		in_type = self.block5.out_type
		# the output type of the sixth convolution layer are 64 regular feature fields of C8
		out_type = escnn.nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr])
		self.block6 = escnn.nn.SequentialModule(
			escnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),  # type: ignore
			escnn.nn.InnerBatchNorm(out_type),
			escnn.nn.ReLU(out_type, inplace=True),
		)
		self.pool3 = escnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)

		self.gpool = escnn.nn.GroupPooling(out_type)

		# number of output channels
		c = self.gpool.out_type.size

		# Fully Connected
		self.fully_net = torch.nn.Sequential(
			torch.nn.Linear(c, 64),
			torch.nn.BatchNorm1d(64),
			torch.nn.ELU(inplace=True),
			torch.nn.Linear(64, num_classes),
		)

	def forward(self, input: torch.Tensor):
		# wrap the input tensor in a GeometricTensor
		# (associate it with the input type)
		x = escnn.nn.GeometricTensor(input, self.input_type)

		# apply each equivariant block

		# Each layer has an input and an output type
		# A layer takes a GeometricTensor in input.
		# This tensor needs to be associated with the same representation of the layer's input type
		#
		# The Layer outputs a new GeometricTensor, associated with the layer's output type.
		# As a result, consecutive layers need to have matching input/output types
		x = self.block1(x)
		x = self.block2(x)
		x = self.pool1(x)

		x = self.block3(x)
		x = self.block4(x)
		x = self.pool2(x)

		x = self.block5(x)
		x = self.block6(x)

		# pool over the spatial dimensions
		x = self.pool3(x)

		# pool over the group
		x = self.gpool(x)

		# unwrap the output GeometricTensor
		# (take the Pytorch tensor and discard the associated representation)
		x = x.tensor

		# classify with the final fully connected layers)
		x = self.fully_net(x.reshape(x.shape[0], -1))

		return x


class SmallSteerableCNN(torch.nn.Module):
	def __init__(self, num_classes=10, input_channels=1, img_width=32, N=8):
		super().__init__()

		# the model is equivariant under rotations by 45 degrees, modelled by C8
		self.r2_act = gspaces.rot2dOnR2(N=N)

		# the input image is a scalar field, corresponding to the trivial representation
		in_type = escnn.nn.FieldType(self.r2_act, input_channels * [self.r2_act.trivial_repr])

		# we store the input type for wrapping the images into a geometric tensor during the forward pass
		self.input_type = in_type

		# convolution 1
		# first specify the output type of the convolutional layer
		# we choose 12 feature fields, each transforming under the regular representation of C8
		out_type = escnn.nn.FieldType(self.r2_act, 12 * [self.r2_act.regular_repr])
		self.block1 = escnn.nn.SequentialModule(
			escnn.nn.MaskModule(in_type, S=img_width, margin=1),  # changed S to work for other image sizes
			escnn.nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
			escnn.nn.InnerBatchNorm(out_type),
			escnn.nn.ReLU(out_type, inplace=True),
			escnn.nn.PointwiseMaxPool2D(out_type, kernel_size=2, stride=2),  # 64 x 12 x 14 x 14
		)

		# convolution 2
		# the old output type is the input type to the next layer
		in_type = self.block1.out_type
		# the output type of the second convolution layer are 24 regular feature fields of C8
		out_type = escnn.nn.FieldType(self.r2_act, 24 * [self.r2_act.regular_repr])
		self.block2 = escnn.nn.SequentialModule(
			escnn.nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),  # type: ignore
			escnn.nn.InnerBatchNorm(out_type),
			escnn.nn.ReLU(out_type, inplace=True),
			escnn.nn.PointwiseMaxPool2D(out_type, kernel_size=2, stride=2),  # 64 x 24 x 7 x 7
		)

		# convolution 3
		# the old output type is the input type to the next layer
		in_type = self.block2.out_type
		# the output type of the third convolution layer are 48 regular feature fields of C8
		out_type = escnn.nn.FieldType(self.r2_act, 48 * [self.r2_act.regular_repr])
		self.block3 = escnn.nn.SequentialModule(
			escnn.nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),  # type: ignore
			escnn.nn.InnerBatchNorm(out_type),
			escnn.nn.ReLU(out_type, inplace=True),
			escnn.nn.PointwiseAdaptiveMaxPool2D(out_type, (1, 1)),  # 64 x 48 x 1 x 1
		)

		self.gpool = escnn.nn.GroupPooling(out_type)

		# number of output channels
		c = self.gpool.out_type.size
		# Fully Connected
		self.fully_net = torch.nn.Sequential(
			torch.nn.Linear(c, 64),
			torch.nn.BatchNorm1d(64),
			torch.nn.ELU(inplace=True),
			torch.nn.Linear(64, num_classes),
			torch.nn.Softmax(dim=1),
		)

	def forward(self, input: torch.Tensor):
		# wrap the input tensor in a GeometricTensor
		# (associate it with the input type)
		x = escnn.nn.GeometricTensor(input, self.input_type)

		# apply each equivariant block

		# Each layer has an input and an output type
		# A layer takes a GeometricTensor in input.
		# This tensor needs to be associated with the same representation of the layer's input type
		#
		# The Layer outputs a new GeometricTensor, associated with the layer's output type.
		# As a result, consecutive layers need to have matching input/output types
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)

		# pool over the group
		x = self.gpool(x)

		# unwrap the output GeometricTensor
		# (take the Pytorch tensor and discard the associated representation)
		x = x.tensor
		# classify with the final fully connected layers)
		x = self.fully_net(x.reshape(x.shape[0], -1))

		return x
