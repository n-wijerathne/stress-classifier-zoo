"""Central paths and constants so every script works from any working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
# Default input: anonymised session-level table (no individual IDs). If a full-detail
# file exists at data/raw/animal_stress_data.csv it is used instead (it is gitignored).
RAW_FULL = DATA_DIR / "raw" / "animal_stress_data.csv"
PUBLIC_INPUT = DATA_DIR / "processed" / "animal_stress_data.csv"
MODEL_READY = DATA_DIR / "processed" / "model_ready.csv"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

SEED = 42
N_SPLITS = 5          # stratified, day-grouped CV folds on the development set
TEST_SPLITS = 7       # one of 7 day-grouped folds (~14%) is held out as the final test set
N_REPEATS = 10        # repeated CV (different fold shuffles) for stability
ALPHA = 0.05


def default_input() -> Path:
    return RAW_FULL if RAW_FULL.exists() else PUBLIC_INPUT
