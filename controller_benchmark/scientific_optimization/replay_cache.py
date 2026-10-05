from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Mapping

from controller_benchmark.optimization.replay import BaselineTrace, ReplayResult


CACHE_SCHEMA_VERSION = 2


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Replay-cache keys cannot contain non-finite floats")
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return _json_value(value.item())
    raise TypeError(f"Unsupported replay-cache value: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class PersistentReplayCache:
    """Content-addressed cache for deterministic Full100 controller replays.

    Cache identity includes the complete immutable trace, controller source,
    controller id, and exact decoded parameters. Floats are never rounded.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._trace_digests: dict[int, str] = {}
        self._plugin_digests: dict[str, str] = {}
        self._runtime_digest: str | None = None
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def _trace_digest(self, trace: BaselineTrace) -> str:
        identity = id(trace)
        cached = self._trace_digests.get(identity)
        if cached is not None:
            return cached
        digest = _sha256_text(
            _canonical_json(
                {
                    "case_id": trace.case_id,
                    "stage": trace.stage,
                    "task_type": trace.task_type,
                    "scenario": trace.scenario,
                    "quality_metric": trace.quality_metric,
                    "max_epochs": trace.max_epochs,
                    "metadata": trace.metadata,
                    "epochs": trace.epochs,
                    "source_job_id": trace.source_job_id,
                }
            )
        )
        self._trace_digests[identity] = digest
        return digest

    def _controller_runtime_digest(self) -> str:
        if self._runtime_digest is not None:
            return self._runtime_digest
        package_root = Path(__file__).resolve().parents[1]
        sources = [package_root / "api.py", package_root / "optimization" / "replay.py"]
        sources.extend(sorted((package_root / "controllers").rglob("*.py")))
        digest = hashlib.sha256()
        for source in sources:
            if not source.is_file():
                continue
            digest.update(source.relative_to(package_root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(source.read_bytes())
            digest.update(b"\0")
        self._runtime_digest = digest.hexdigest()
        return self._runtime_digest

    def _plugin_digest(self, plugin: str) -> str:
        cached = self._plugin_digests.get(plugin)
        if cached is not None:
            return cached
        module_name = plugin.split(":", 1)[0]
        specification = importlib.util.find_spec(module_name)
        source_digest = "source-unavailable"
        if specification and specification.origin:
            source = Path(specification.origin)
            if source.is_file():
                source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        digest = _sha256_text(
            _canonical_json({"plugin": plugin, "source_sha256": source_digest})
        )
        self._plugin_digests[plugin] = digest
        return digest

    def key(
        self,
        trace: BaselineTrace,
        *,
        controller_plugin: str,
        controller_id: str,
        parameters: Mapping[str, Any],
    ) -> str:
        return _sha256_text(
            _canonical_json(
                {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "trace_sha256": self._trace_digest(trace),
                    "controller_plugin_sha256": self._plugin_digest(controller_plugin),
                    "controller_runtime_sha256": self._controller_runtime_digest(),
                    # RAPEC-v9 currently uses the id in its deterministic draw seed.
                    "controller_id": controller_id,
                    "parameters": parameters,
                }
            )
        )

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> ReplayResult | None:
        path = self._path(key)
        if not path.is_file():
            with self._lock:
                self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
                return None
            result = ReplayResult(**payload["result"])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None
        with self._lock:
            self.hits += 1
        return result

    def put(self, key: str, result: ReplayResult) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "key": key,
            "result": result.to_dict(),
        }
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
        )
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        with self._lock:
            self.writes += 1

    def statistics(self) -> dict[str, int | str]:
        return {
            "root": str(self.root),
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
        }
