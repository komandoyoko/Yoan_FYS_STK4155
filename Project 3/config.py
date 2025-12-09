# config.py

from dataclasses import dataclass, asdict, field
from pathlib import Path
import torch


# ---------- Paths ----------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
PLOTS_DIR = PROJECT_ROOT / "plots"

# Create dirs if they don't exist (safe to import everywhere)
for d in [CHECKPOINT_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ---------- Core configuration objects ----------

@dataclass
class DataConfig:
    # Which gamer´s data to use
    gamer_ids: tuple = (1, 2, 3, 4, 5)

    # How much data to use from each gamer
    max_hours_per_gamer: float = 4.0

    # Sequence construction
    seq_len: int = 200          # input sequence length (timesteps)
    pred_len: int = 1           # how many future steps to predict (1 = next-step)

    label_type: str = "sleepiness"

    # Train/val/test split (on sequences)
    train_frac: float = 0.7
    val_frac: float = 0.15      # remainder goes to test

    # File patterns (adjust if your filenames differ)
    ppg_pattern: str = "gamer{gamer_id}-ppg-*.csv"
    annotations_pattern: str = "gamer{gamer_id}-annotations.csv"

    # Normalization options
    normalize_signal: bool = True
    normalization_mode: str = "zscore"  # or "minmax"


@dataclass
class ModelConfig:
    model_type: str = "rnn"     # "rnn", "lstm", "gru"
    input_size: int = 1         # PPG is a single channel
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    bidirectional: bool = False

    # Output settings
    output_size: int = 7        # 1 for regression (next-step / score)


@dataclass
class TrainingConfig:
    num_epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

    # Logging / saving
    print_every: int = 10
    save_best_model: bool = True
    early_stopping_patience: int = 10

    # Reproducibility
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



# This is what you'll import from other files:
cfg = Config()

