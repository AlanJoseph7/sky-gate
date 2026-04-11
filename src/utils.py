from pathlib import Path
import numpy as np

# -----------------------------
# PROJECT STRUCTURE
# FIX #5: use pathlib for robust path resolution
# utils.py lives at: project_root/src/utils.py
# -----------------------------
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"   # FIX #3
LOG_DIR   = BASE_DIR / "logs"     # FIX #4

# -----------------------------
# SHARED CONSTANTS (FIX #6, #7)
# Single source of truth for the whole pipeline.
# Import these in every file instead of redefining locally:
#   from utils import FEATURE_COLUMNS, RANDOM_SEED
# -----------------------------
RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "altitude",
    "velocity",
    "heading",
    "distance_km",
    "climb_rate",
    "acceleration",
    "heading_change",
    "speed_consistency"
]

# -----------------------------
# PATH HELPERS
# -----------------------------
def get_data_path(filename: str) -> str:
    """Return absolute path to a file inside the data/ directory."""
    return str(DATA_DIR / filename)

def get_model_path(filename: str) -> str:
    """Return absolute path to a file inside the models/ directory."""
    return str(MODEL_DIR / filename)

def get_log_path(filename: str) -> str:
    """Return absolute path to a file inside the logs/ directory."""
    return str(LOG_DIR / filename)

# -----------------------------
# DIRECTORY SETUP
# FIX #2: no trailing separator via get_data_path("")
# FIX #3: models/ directory provisioned
# FIX #4: logs/ directory provisioned
# -----------------------------
def ensure_data_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "plots").mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# KERAS MODEL LOADING
# Supports older tf_keras versions by stripping unsupported InputLayer keys.
# -----------------------------

def load_keras_model_compatible(filepath: str, **kwargs):
    """Load a Keras model while ignoring unsupported InputLayer config keys."""
    from tf_keras.models import load_model
    from tf_keras.layers import InputLayer

    class CompatibleInputLayer(InputLayer):
        @classmethod
        def from_config(cls, config):
            config = config.copy()
            config.pop("optional", None)
            return super().from_config(config)

    kwargs.setdefault("custom_objects", {})
    kwargs["custom_objects"].update({"InputLayer": CompatibleInputLayer})
    return load_model(filepath, **kwargs)

# -----------------------------
# SEQUENCE CREATION
# FIX #1: removed naive create_sequences() — it ignored aircraft grouping
# and would bleed sequences across aircraft boundaries if used.
# Use create_sequences_per_aircraft() in lstm_autoencoder.py instead.
# -----------------------------