from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    description: str
    project_root: Path


@dataclass(frozen=True)
class PathsConfig:
    processed_dir: Path
    splits_dir: Path
    models_dir: Path
    normalization_stats_filename: str
    metrics_filename: str
    checkpoint_filename: str


@dataclass(frozen=True)
class AudioConfig:
    target_sample_rate: int
    chunk_length_s: float
    hop_length_s: float
    cache_waveforms: bool


@dataclass(frozen=True)
class LabelsConfig:
    max_gap_days: int


@dataclass(frozen=True)
class SplitConfig:
    random_seed: int
    subsample_fraction: float


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    dropout: float
    do_normalize: bool
    force_recompute_normalizations: bool


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    max_epochs: int
    patience: int
    head_learning_rate: float
    backbone_learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    use_amp: bool


@dataclass(frozen=True)
class FinetuningConfig:
    freeze_strategy: str
    unfreeze_last_n: int


@dataclass(frozen=True)
class ImbalanceConfig:
    use_class_weighting: bool
    use_balanced_sampler: bool


@dataclass(frozen=True)
class EvaluationConfig:
    default_threshold: float
    threshold_scan_start: float
    threshold_scan_end: float
    threshold_scan_step: float


@dataclass(frozen=True)
class RuntimeConfig:
    num_workers: int
    pin_memory: bool
    log_every_n_batches: int


@dataclass(frozen=True)
class QueenPipelineConfig:
    project: ProjectConfig
    paths: PathsConfig
    audio: AudioConfig
    labels: LabelsConfig
    split: SplitConfig
    model: ModelConfig
    training: TrainingConfig
    finetuning: FinetuningConfig
    imbalance: ImbalanceConfig
    evaluation: EvaluationConfig
    runtime: RuntimeConfig

    def to_dict(self) -> dict:
        return {
            "project": asdict(self.project),
            "paths": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in asdict(self.paths).items()
            },
            "audio": asdict(self.audio),
            "labels": asdict(self.labels),
            "split": asdict(self.split),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "finetuning": asdict(self.finetuning),
            "imbalance": asdict(self.imbalance),
            "evaluation": asdict(self.evaluation),
            "runtime": asdict(self.runtime),
        }


def load_config(path: str | Path) -> QueenPipelineConfig:
    config_path = Path(path)
    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    project_root = Path(raw["project"]["project_root"]).resolve()

    return QueenPipelineConfig(
        project=ProjectConfig(
            name=raw["project"]["name"],
            description=raw["project"].get("description", ""),
            project_root=project_root,
        ),
        paths=PathsConfig(
            processed_dir=project_root / raw["paths"]["processed_dir"],
            splits_dir=project_root / raw["paths"]["splits_dir"],
            models_dir=project_root / raw["paths"]["models_dir"],
            normalization_stats_filename=raw["paths"]["normalization_stats_filename"],
            metrics_filename=raw["paths"]["metrics_filename"],
            checkpoint_filename=raw["paths"]["checkpoint_filename"],
        ),
        audio=AudioConfig(**raw["audio"]),
        labels=LabelsConfig(**raw["labels"]),
        split=SplitConfig(**raw["split"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        finetuning=FinetuningConfig(**raw["finetuning"]),
        imbalance=ImbalanceConfig(**raw["imbalance"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        runtime=RuntimeConfig(**raw["runtime"]),
    )
