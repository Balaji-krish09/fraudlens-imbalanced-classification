"""Central config: paths and constants shared across notebooks."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = "Class"
SCALE_COLS = ["Time", "Amount"]   # V1-V28 are already PCA-scaled
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

for d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)
