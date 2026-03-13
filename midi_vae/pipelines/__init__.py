"""Pipeline stages and DAG-based runner for experiment execution."""

from midi_vae.pipelines.base import PipelineStage, PipelineRunner, StageIO
from midi_vae.pipelines.ingest import IngestStage
from midi_vae.pipelines.render import RenderStage
from midi_vae.pipelines.encode import EncodeStage
from midi_vae.pipelines.decode import DecodeStage
from midi_vae.pipelines.detect import DetectStage
from midi_vae.pipelines.evaluate import EvaluateStage
from midi_vae.pipelines.sweep import SweepExecutor

__all__ = [
    "PipelineStage",
    "PipelineRunner",
    "StageIO",
    "IngestStage",
    "RenderStage",
    "EncodeStage",
    "DecodeStage",
    "DetectStage",
    "EvaluateStage",
    "SweepExecutor",
]
