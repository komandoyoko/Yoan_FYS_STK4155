from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV_FILENAME = "gamer2-ppg-2000-01-02.csv" 

# Full path to the raw CSV
RAW_CSV_PATH = DATA_DIR / RAW_CSV_FILENAME

@dataclass
class DataConfig:
    # Column names in the CSV
    timestamp_col: str = "timestamp"   # set to None if there is no timestamp
    ppg_col: str = "ppg"               # name of the PPG signal column

    sample_rate_hz: int = 64

    limit_hours: float = 4.0

    window_seconds: int = 30           # length of each sequence in seconds
    window_overlap_seconds: int = 0

    labeling_strategy: str = "dummy_time_based"

    num_classes: int = 3               # e.g. {0: alert, 1: medium, 2: tired}

    # Train/val/test split (fractions of windows)
    train_fraction: float = 0.6
    val_fraction: float = 0.2
    test_fraction: float = 0.2

    # Sanity check: we want fractions to add up to ~1
    def check_splits(self):
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Train/val/test fractions must sum to 1. Got {total}.")
        
@dataclass
class ModelConfig:
    # RNN type: "lstm" or "gru"
    rnn_type: str = "lstm"

    input_size: int = 1        # 1 feature: raw PPG value
    hidden_size: int = 64
    num_layers: int = 1
    bidirectional: bool = True

    # Fully connected head
    fc_hidden_size: int = 64
    dropout: float = 0.3

    # Output settings (for classification)
    num_classes: int = 3       # should match DataConfig.num_classes

@dataclass
class TrainConfig:
    batch_size: int = 64
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0       # L2 regularization
    gradient_clip: float = 5.0      # to avoid exploding gradients, set None to disable

    # Random seeds for reproducibility
    seed: int = 42

    # Device preference: "cuda" or "cpu"
    device: str = "cuda"  # will automatically fall back to CPU if CUDA is not available

@dataclass
class Config:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()

config = Config()

if __name__ == "__main__":
    # Quick manual check:
    print("Project root:", PROJECT_ROOT)
    print("Raw CSV path:", RAW_CSV_PATH)
    config.data.check_splits()
    print("Config OK.")
