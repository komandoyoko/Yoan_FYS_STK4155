from dataclasses import dataclass, asdict, field
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
PLOTS_DIR = PROJECT_ROOT / "plots"


@dataclass
class DataConfig:
    # Use all gamers
    gamer_ids: tuple = (1, 2, 3, 4, 5)

    # Use FULL data per gamer (None)
    max_hours_per_gamer: float | None = None   

    # Sequence construction (length in *samples*)
    seq_len: int = 200          # window length
    pred_len: int = 1           # unused for sleepiness, but keep for flexibility

    # doing sleepiness prediction
    label_type: str = "sleepiness"

    # Sliding windows around each annotation
    backward_minutes: float = 10.0          # how far back from annotation we look
    window_stride_fraction: float = 0.5     # 0.5 => 50% overlap

    # Train/val/test split
    train_frac: float = 0.7
    val_frac: float = 0.15

    # Filename patterns (still used by loader)
    ppg_pattern: str = "gamer{gamer_id}-ppg-*.csv"
    annotations_pattern: str = "gamer{gamer_id}-annotations.csv"

    normalize_signal: bool = True
    normalization_mode: str = "zscore"



@dataclass
class ModelConfig:
    model_type: str = "lstm"   # "rnn" or "gru" also fine
    input_size: int = 1
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    bidirectional: bool = False

    # 7 classes: sleepiness 1–7
    output_size: int = 7


@dataclass
class TrainingConfig:
    batch_size: int = 128
    num_epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    early_stopping_patience: int = 6
    print_every: int = 1
    save_best_model: bool = True
    seed: int = 42



@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self):
        d = asdict(self)
        d["PROJECT_ROOT"] = str(PROJECT_ROOT)
        d["DATA_DIR"] = str(DATA_DIR)
        d["CHECKPOINT_DIR"] = str(CHECKPOINT_DIR)
        d["PLOTS_DIR"] = str(PLOTS_DIR)
        return d



# This is what is imported from other files:
cfg = Config()

