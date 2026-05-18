import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-score-matching-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ]
}

# Rolling windows (days)
WINDOWS = [63, 252, 504, 1008, 2016]

# Score network hyperparameters
HIDDEN_DIM = 128
N_LAYERS = 3
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 100
NOISE_SCALE = 0.1   # for denoising score matching

# Langevin sampling parameters
LANGEVIN_STEPS = 10
LANGEVIN_STEP_SIZE = 0.1

TOP_N = 3
