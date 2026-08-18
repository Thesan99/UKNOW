"""
UKNOW — shared configuration: paths, HF environment, dataset registry.

Kept torch-free on purpose so it can be imported BEFORE torch / diffusers.
setup_hf_env() must run before the first HuggingFace import for the cache and
download env vars to take effect.

PATHS ARE NOT TIED TO ANY PERSONAL MACHINE.
All data lives under DATA_ROOT, which defaults to ./data next to this repo and
can be pointed anywhere with the UKNOW_DATA environment variable:

    export UKNOW_DATA=/scratch/uknow_data      # or wherever you keep large data

Nothing in this repo hardcodes a home directory; a fresh clone works as-is.
"""

import os
from pathlib import Path

# Repo root = the folder this file lives in.
PROJECT_ROOT = Path(__file__).resolve().parent

# Large, gitignored data tree. Override with UKNOW_DATA; defaults to ./data.
DATA_ROOT = Path(os.environ.get("UKNOW_DATA", PROJECT_ROOT / "data")).resolve()

# --- Derived locations (each script makes the dirs it writes) ---
CACHE_DIR     = DATA_ROOT / "hf_cache"
PROMPTS_DIR   = DATA_ROOT / "prompts"
REAL_ROOT     = DATA_ROOT / "dataset_real"
FINAL_ROOT    = DATA_ROOT / "dataset_final"
FINETUNED_DIR = DATA_ROOT / "finetuned_sdxl_detector"

METADATA_REAL = DATA_ROOT / "metadata_real.csv"
MANIFEST      = DATA_ROOT / "manifest.csv"
BASELINE_CSV  = DATA_ROOT / "baseline_results.csv"
FINETUNE_CSV  = DATA_ROOT / "finetune_results.csv"

# Pretrained detector fine-tuned in this project (head-only).
MODEL_ID = "Organika/sdxl-detector"


def setup_hf_env():
    """Set HF cache + download env vars and raise the open-file limit.

    Must be called before importing torch / diffusers / huggingface_hub.
    Returns the cache directory as a string.
    """
    # Sequential downloads instead of the parallel Xet backend, which opens too
    # many sockets on shared servers and crashes with:
    #   zmq.error.ZMQError: Too many open files
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    try:
        import resource
        _soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    except Exception as e:  # non-fatal
        print("could not raise open-file limit (non-fatal):", e)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(CACHE_DIR)
    os.environ["HF_HUB_CACHE"] = str(CACHE_DIR)
    # Optional: export HF_TOKEN=hf_... to lift anonymous rate limits.
    # Never hardcode a token here -- this file is public.
    return str(CACHE_DIR)


# Human-readable topics used for prompt + real-image collection. This is the
# as-built 8-topic set. The last three are information-integrity categories;
# see NOTES.md for the open question about their prompt scaffold and real source.
TOPICS = [
    "wildfire",
    "flood",
    "protest crowd",
    "war-torn destroyed buildings",
    "plane crash debris",
    "documents",
    "fake news",
    "memes",
]

# In-distribution synthetic generators. All SDXL-family fine-tunes; each writes
# its own versioned dataset. The out-of-distribution FLUX set is a later stage.
GENERATORS = {
    "realvisxl_v4": {
        "model_id": "SG161222/RealVisXL_V4.0",
        "dataset_dir": "datasetV1",
        "metadata_csv": "metadata_sdxlV1.csv",
    },
    "realvisxl_v5": {
        "model_id": "SG161222/RealVisXL_V5.0",
        "dataset_dir": "datasetV2",
        "metadata_csv": "metadata_sdxlV2.csv",
    },
    "juggernaut_v9": {
        "model_id": "RunDiffusion/Juggernaut-XL-v9",
        "dataset_dir": "datasetV3",
        "metadata_csv": "metadata_sdxlV3.csv",
    },
}

# Wikimedia Commons requires a descriptive User-Agent with a real contact, or it
# may return HTTP 403. This is a public project URL, not a secret; override with
# UKNOW_CONTACT if you fork.
CONTACT = os.environ.get("UKNOW_CONTACT", "https://github.com/Thesan99")
USER_AGENT = f"UKNOW-RealImageCollector/1.0 (contact: {CONTACT})"
