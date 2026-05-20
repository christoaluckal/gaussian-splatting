from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ViewpointPartition:
    name: str
    camera_indices: list[int]
    metadata: dict[str, Any] | None = None

