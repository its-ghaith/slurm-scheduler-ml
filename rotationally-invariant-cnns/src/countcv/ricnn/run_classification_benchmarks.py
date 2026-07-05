import random
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import yaml
from codecarbon import OfflineEmissionsTracker
from ricnn.models import FlipEquiCNN, SimpleCNN, SmallSteerableCNN
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from config.auth_mgr import AuthMgr

REORIENT_CONFIG = {"p_horizontal": 0.5, "rotation_degrees": 180}
REMOTE_MLFLOW = False  # Boolean, local tracking if False
MODEL_REGISTRY = {  # key: (model_class, model_parameters)
	"SimpleCNN": (SimpleCNN, {}),
	"FlipEquiCNN": (FlipEquiCNN, {}),
	"SmallSteerableCNN": (SmallSteerableCNN, {"N": 8, "img_width": 28}),
}


def get_mnist_loaders(batch_size: int, data_dir: str = "data", num_workers: int = 2, pin_memory: bool = True):
	"""
	Returns train/test DataLoaders for MNIST with appropriate transforms.
	"""
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

	train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
	test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

	train_loader = DataLoader(
		train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
	)
	test_loader = DataLoader(
		test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
	)

	return train_loader, test_loader


def get_cifar10_loaders(batch_size: int, data_dir: str = "data", num_workers: int = 2, pin_memory: bool = True):
	"""
	Returns train/test DataLoaders for MNIST fashion with appropriate transforms.
	"""
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
	test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

	train_loader = DataLoader(
		train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
	)
	test_loader = DataLoader(
		test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
	)

	return train_loader, test_loader


def set_seed(seed: int = 42):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


class ModelFactory:
	"""Builds a class object which takes only two arguments num_classes and input_channels.
	all other arguments are fixed."""

	def __init__(self, cls, **preset_args):
		self.cls = cls
		self.preset_args = preset_args
		self.__name__ = cls.__name__

	def __call__(self, num_classes, input_channels):
		return self.cls(num_classes=num_classes, input_channels=input_channels, **self.preset_args)

	def __repr__(self):
		return self.__name__


def train_one_epoch(
	model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device
):
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0

	for inputs, targets in loader:
		inputs, targets = inputs.to(device), targets.to(device)
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()

		running_loss += loss.item() * inputs.size(0)
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()

	epoch_loss = running_loss / total
	accuracy = 100.0 * correct / total
	return epoch_loss, accuracy


def evaluate(
	model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, reorient_config: None | dict
):
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0
	with torch.no_grad():
		for inputs, targets in loader:
			inputs, targets = inputs.to(device), targets.to(device)
			if reorient_config:
				transforms = v2.Compose(
					[
						v2.RandomHorizontalFlip(p=reorient_config["p_horizontal"]),
						v2.RandomRotation(degrees=reorient_config["rotation_degrees"]),
					]
				)
				inputs = transforms(inputs)
			outputs = model(inputs)
			loss = criterion(outputs, targets)
			running_loss += loss.item() * inputs.size(0)
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

	epoch_loss = running_loss / total
	accuracy = 100.0 * correct / total
	return epoch_loss, accuracy


def run_experiment(dataset_name, model_config: dict, reorient: bool, model_class: type[nn.Module], experiment_tag: str):
	config = SimpleNamespace(**model_config)
	set_seed(42)
	if REMOTE_MLFLOW:
		mlflow.environment_variables.MLFLOW_TRACKING_TOKEN.set(auth.get_token())
	mlflow.set_experiment(f"{dataset_name}-{experiment_tag}")

	with mlflow.start_run(run_name=f"{model_class.__name__}_{model_config}"):
		mlflow.log_param("dataset_name", dataset_name)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		Path("energy_logs").mkdir(exist_ok=True)
		# Initialize and start emissions tracker
		tracker = OfflineEmissionsTracker(
			log_level="error",
			output_dir="energy_logs",  # Custom folder
			output_file=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_emissions.csv",  # Custom filename
			country_iso_code="DEU",
		)
		tracker.start()

		if dataset_name == "mnist":
			train_loader, test_loader = get_mnist_loaders(batch_size=config.batch_size, num_workers=4)
			num_classes = 10
			input_channels = 1
		elif dataset_name == "cifar10":
			train_loader, test_loader = get_cifar10_loaders(batch_size=config.batch_size, num_workers=4)
			num_classes = 10
			input_channels = 3
		else:
			raise ValueError("No dataset of that name found.")

		model = model_class(num_classes=num_classes, input_channels=input_channels).to(device)
		criterion = nn.NLLLoss()
		optimizer = optim.Adam(model.parameters(), lr=model_config["lr"])
		mlflow.log_params(model_config)
		mlflow.log_param("model_type", model_class.__name__)

		for epoch in tqdm(range(config.epochs), desc=f"Training experiment: loss = {criterion.__class__.__name__}"):
			train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
			val_loss, val_acc = evaluate(model, test_loader, criterion, device, reorient_config=None)
			if reorient:
				reorient_loss, reorient_acc = evaluate(
					model, test_loader, criterion, device, reorient_config=REORIENT_CONFIG
				)
			else:
				reorient_loss, reorient_acc = None, None
			mlflow.log_metrics(
				{
					"epoch": epoch,
					"train_loss": train_loss,
					"train_acc": train_acc,
					"val_loss": val_loss,
					"val_acc": val_acc,
					"reorient_loss": reorient_loss,
					"reorient_acc": reorient_acc,
				},
				step=epoch,
			)
			if REMOTE_MLFLOW:  # Renew access token
				auth.renew_token()
		tracker.stop()
		mlflow.log_metrics(
			{
				"energy_kwh": tracker._total_energy.kWh,
				"duration_sec": tracker.final_emissions_data.duration,
			}
		)


if __name__ == "__main__":
	if REMOTE_MLFLOW:
		credentials = yaml.safe_load(Path("config/config.yaml").read_text())
		auth = AuthMgr(credentials["oidc_url"], credentials["client_id"])
		auth.login()

		mlflow.set_tracking_uri("https://mlflow.ai-env.de")
	else:
		Path("mlruns").mkdir(exist_ok=True)  # Create folder mlruns to locally track runs
		mlflow.set_tracking_uri("mlruns")

	dataset_name = "mnist"
	training_config = {"lr": 0.001, "epochs": 10, "batch_size": 64}

	experiment_tag = "simple_vs_escnn"
	reorient = True
	models = [
		"SimpleCNN",
		# "FlipEquiCNN",
		# "SmallSteerableCNN",
	]

	# Iterate over all models
	for model_name in models:
		model_class, preset_arguments = MODEL_REGISTRY[model_name]
		model_class = ModelFactory(model_class, **preset_arguments)
		run_experiment(dataset_name, training_config, reorient, model_class=model_class, experiment_tag=experiment_tag)
