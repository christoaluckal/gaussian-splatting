from __future__ import annotations

import importlib


DEFAULT_SPLITTER_FUNCTION = "partition_viewpoints"
DEFAULT_PACKAGE_PREFIX = "scene.viewpoint_splitters"


def load_viewpoint_splitter(module_name: str):
    resolved_name = module_name
    if "." not in module_name:
        resolved_name = f"{DEFAULT_PACKAGE_PREFIX}.{module_name}"

    module = importlib.import_module(resolved_name)
    splitter = getattr(module, DEFAULT_SPLITTER_FUNCTION, None)
    if splitter is None:
        raise AttributeError(
            f"Viewpoint splitter module '{resolved_name}' must define "
            f"'{DEFAULT_SPLITTER_FUNCTION}(train_cameras, num_partitions, config)'."
        )
    return splitter

