"""Pipeline stage ABC and DAG-based pipeline runner.

Experiments are composed as directed acyclic graphs of PipelineStage nodes.
The PipelineRunner resolves dependencies via topological sort, executes
stages in order, and supports caching and resume.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from midi_vae.config import ExperimentConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageIO:
    """Declares the typed inputs and outputs of a pipeline stage.

    Used by the runner to resolve dependencies between stages.
    """

    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()


class PipelineStage(ABC):
    """Abstract base for pipeline stages.

    Each stage declares its inputs/outputs via io(), implements run(),
    and can optionally provide a cache key for result caching.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize the stage with experiment config.

        Args:
            config: Full experiment configuration.
        """
        self.config = config

    @abstractmethod
    def io(self) -> StageIO:
        """Declare input and output keys for dependency resolution.

        Returns:
            StageIO with input keys this stage reads and output keys it produces.
        """
        ...

    @abstractmethod
    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute the stage.

        Args:
            context: Dict of all available data from previous stages,
                keyed by output names.

        Returns:
            Dict of output data produced by this stage.
        """
        ...

    @property
    def name(self) -> str:
        """Stage name, defaults to class name."""
        return self.__class__.__name__

    def cache_key(self, context: dict[str, Any]) -> str | None:
        """Compute a content-addressed cache key for this stage's output.

        Override to enable caching. Return None to disable caching for this stage.

        Args:
            context: Current pipeline context.

        Returns:
            A hex digest string, or None if caching is not supported.
        """
        return None


class PipelineRunner:
    """DAG-based pipeline executor with caching and resume support.

    Takes a list of stages, resolves dependencies via topological sort,
    and executes them in order. Supports caching results to disk and
    resuming from a previous checkpoint.
    """

    def __init__(
        self,
        stages: list[PipelineStage],
        config: ExperimentConfig,
        cache_dir: str | None = None,
    ) -> None:
        """Initialize the pipeline runner.

        Args:
            stages: List of pipeline stages to execute.
            config: Experiment configuration.
            cache_dir: Directory for caching stage outputs. None disables caching.
        """
        self.stages = stages
        self.config = config
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._execution_order: list[PipelineStage] | None = None

    def _topological_sort(self) -> list[PipelineStage]:
        """Sort stages in dependency order.

        Returns:
            Stages sorted so that all inputs are satisfied before execution.

        Raises:
            ValueError: If there are unresolved dependencies or cycles.
        """
        # Build output -> stage mapping
        output_providers: dict[str, PipelineStage] = {}
        for stage in self.stages:
            for output_key in stage.io().outputs:
                if output_key in output_providers:
                    raise ValueError(
                        f"Duplicate output '{output_key}': provided by both "
                        f"'{output_providers[output_key].name}' and '{stage.name}'"
                    )
                output_providers[output_key] = stage

        # Build adjacency: stage -> set of stages it depends on
        dependencies: dict[str, set[str]] = {}
        stage_by_name: dict[str, PipelineStage] = {}
        for stage in self.stages:
            stage_by_name[stage.name] = stage
            dependencies[stage.name] = set()
            for input_key in stage.io().inputs:
                if input_key in output_providers:
                    dep_stage = output_providers[input_key]
                    if dep_stage.name != stage.name:
                        dependencies[stage.name].add(dep_stage.name)

        # Kahn's algorithm
        sorted_names: list[str] = []
        ready = [name for name, deps in dependencies.items() if not deps]

        while ready:
            name = ready.pop(0)
            sorted_names.append(name)
            for other, deps in dependencies.items():
                if name in deps:
                    deps.remove(name)
                    if not deps:
                        ready.append(other)

        if len(sorted_names) != len(self.stages):
            unsorted = set(s.name for s in self.stages) - set(sorted_names)
            raise ValueError(f"Cycle or unresolved dependencies in stages: {unsorted}")

        return [stage_by_name[name] for name in sorted_names]

    def run(self, resume_from: str | None = None) -> dict[str, Any]:
        """Execute the pipeline.

        Args:
            resume_from: Stage name to resume from (skip earlier stages using cache).

        Returns:
            Final pipeline context with all stage outputs.
        """
        if self._execution_order is None:
            self._execution_order = self._topological_sort()

        context: dict[str, Any] = {}
        skip = resume_from is not None

        for stage in self._execution_order:
            stage_name = stage.name

            # Handle resume
            if skip:
                if stage_name == resume_from:
                    skip = False
                else:
                    # Try to load from cache
                    cached = self._load_cache(stage, context)
                    if cached is not None:
                        context.update(cached)
                        logger.info(f"[CACHED] {stage_name}")
                        continue
                    else:
                        logger.warning(
                            f"Cannot resume: no cache for '{stage_name}'. Running from here."
                        )
                        skip = False

            # Check cache
            cache_key = stage.cache_key(context)
            if cache_key is not None:
                cached = self._load_cache(stage, context)
                if cached is not None:
                    context.update(cached)
                    logger.info(f"[CACHED] {stage_name} (key={cache_key[:8]})")
                    continue

            # Execute stage
            logger.info(f"[RUN] {stage_name}")
            start = time.time()
            outputs = stage.run(context)
            elapsed = time.time() - start
            logger.info(f"[DONE] {stage_name} ({elapsed:.1f}s)")

            # Cache outputs
            if cache_key is not None and self.cache_dir:
                self._save_cache(stage, cache_key, outputs)

            context.update(outputs)

        return context

    def _cache_path(self, stage: PipelineStage, cache_key: str) -> Path:
        """Get the cache file path for a stage."""
        assert self.cache_dir is not None
        return self.cache_dir / f"{stage.name}_{cache_key}.json"

    def _load_cache(self, stage: PipelineStage, context: dict[str, Any]) -> dict[str, Any] | None:
        """Try to load cached stage output."""
        cache_key = stage.cache_key(context)
        if cache_key is None or self.cache_dir is None:
            return None

        cache_file = self._cache_path(stage, cache_key)
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning(f"Corrupted cache for {stage.name}, re-running")
                return None
        return None

    def _save_cache(self, stage: PipelineStage, cache_key: str, outputs: dict[str, Any]) -> None:
        """Save stage output to cache."""
        if self.cache_dir is None:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self._cache_path(stage, cache_key)
        try:
            with open(cache_file, "w") as f:
                json.dump(outputs, f)
        except (TypeError, OSError) as e:
            logger.warning(f"Cannot cache {stage.name}: {e}")


def compute_hash(*args: Any) -> str:
    """Compute a content-addressed hash from arbitrary arguments.

    Args:
        *args: Values to include in the hash.

    Returns:
        Hex digest string (first 16 chars of SHA-256).
    """
    hasher = hashlib.sha256()
    for arg in args:
        hasher.update(str(arg).encode())
    return hasher.hexdigest()[:16]
