from __future__ import annotations

import importlib
import inspect
from typing import Type

from .api import Controller, ControllerContext


def load_controller(path: str, context: ControllerContext) -> Controller:
    """Load `package.module:ControllerClass` and validate the public contract."""
    if ":" not in path:
        raise ValueError("Controller path must use 'package.module:ClassName'.")
    module_name, class_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    controller_type: Type[Controller] = getattr(module, class_name)
    if not inspect.isclass(controller_type) or not issubclass(controller_type, Controller):
        raise TypeError(f"{path} must inherit controller_benchmark.api.Controller")
    return controller_type(context)
