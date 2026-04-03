"""
config.py — All hyperparameters and paths for the BFI pipeline.
"""

from pathlib import Path

# ─── Project root ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ─── Dataset paths (override with env vars or edit here) ─────────────────────
DATA_ROOTS = {
    "CHB_MIT":   Path("data/raw/chb-mit"),          # D1
    "TUH":       Path("data/raw/tuh_eeg_seizure"),   # D2
    "SIENA":     Path("data/raw/siena"),             # D3
    "SPARCNET":  Path("data/raw/sparcnet"),          # D4
    "ICARE":     Path("data/raw/i-care"),            # D5
    "ECAMS":     Path("data/raw/e-cam-s"),           # D6
    "SAH_PROS":  Path("data/raw/sah_prospective"),   # D7
    "SAH_AUTO":  Path("data/raw/sah_automated"),     # D8
}

# SPaRCNet CSV (pre-extracted features)
SPARCNET_FEATURES_CSV = DATA_ROOTS["SPARCNET"] / "sparcnet_features.csv"

# E-CAM-S clinical labels
ECAMS_LABELS_CSV = DATA_ROOTS["ECAMS"] / "E_CAM_S_Data.csv"

# SAH master spreadsheet
SAH_MASTER_XLSX = DATA_ROOTS["SAH_PROS"] / "sah_data_MASTER_FOR_ANALYSIS_v5.xlsx"

# ─── Standard montage ────────────────────────────────────────────────────────
STANDARD_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
    "C3",  "C4",  "Cz",
    "T3",  "T4",  "T5",  "T6",
    "P3",  "P4",  "Pz",
    "O1",  "O2",
]
N_CHANNELS = len(STANDARD_CHANNELS)   # 19

# ─── Preprocessing ───────────────────────────────────────────────────────────
TARGET_SFREQ        = 256        # Hz
BANDPASS_LOW        = 0.5        # Hz
BANDPASS_HIGH       = 45.0       # Hz
BANDPASS_ORDER      = 4
NOTCH_FREQS         = [50.0, 60.0]   # Hz
AMP_THRESHOLD_UV    = 500.0      # µV
FLATLINE_THRESH_S   = 5.0        # seconds of flatline → bad segment
ZSCORE_WINDOW_S     = 60.0       # seconds for sliding z-score

# ─── Segmentation ────────────────────────────────────────────────────────────
WINDOW_S            = 10         # seconds per window
OVERLAP_S           = 5          # seconds overlap (step = 5 s)
SEQ_LEN             = 36         # L windows per sequence  (6 min of data)
ENTROPY_HISTORY_S   = 15 * 60   # 15 min trailing history for entropy slope
INSTAB_HISTORY_S    = 30 * 60   # 30 min trailing history for trend

# ─── Feature extraction ──────────────────────────────────────────────────────
# Spectral bands (Hz)
BANDS = {
    "delta": (1.0,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0,  13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# Coherence channel pairs (names)
COHERENCE_PAIRS = [
    ("F3", "P3"), ("F4", "P4"), ("F3", "F4"),
    ("P3", "P4"), ("Fz", "Pz"),
]

# Permutation entropy params
PE_ORDER  = 5
PE_DELAY  = 1

# Sample entropy params
SE_M  = 2
SE_R  = 0.2   # r = SE_R * std(signal)

# DFA
DFA_MIN_WIN = 10
DFA_MAX_WIN = int(WINDOW_S * TARGET_SFREQ // 2)

# Network connectivity threshold (proportional)
NETWORK_THRESHOLD = 0.2   # keep top 20% of wPLI edges

# Left / right hemisphere channel indices in STANDARD_CHANNELS
LEFT_CHANNELS  = ["Fp1", "F3", "F7", "C3", "T3", "P3", "T5", "O1"]
RIGHT_CHANNELS = ["Fp2", "F4", "F8", "C4", "T4", "P4", "T6", "O2"]

# ─── Pattern groupings (indices into STANDARD_CHANNELS) ──────────────────────
# Pattern 1 – IACF: alpha coherence + graph metrics
PATTERN1_FEATURES = [
    "alpha_coh_FpO", "alpha_coh_FO", "alpha_coh_IH",
    "global_efficiency", "modularity",
    "posterior_clustering", "frontal_clustering",
]

# Pattern 2 – Hemispheric FC/CP/PO coherence + asymmetry
PATTERN2_FEATURES = [
    "alpha_coh_FC_aff", "alpha_coh_CP_aff", "alpha_coh_PO_aff",
    "coherence_asymmetry", "hemispheric_density",
]

# Pattern 3 – Thalamocortical: FC/CP/FP coherence + frontal
PATTERN3_FEATURES = [
    "alpha_coh_FC", "alpha_coh_CP", "alpha_coh_FP",
    "frontal_theta_ratio", "frontal_centrality", "network_variance",
]

# Global features
GLOBAL_FEATURES = [
    "spectral_slope", "broadband_power", "full_entropy", "overall_instability",
]

# ─── Model architecture ──────────────────────────────────────────────────────
D_PROJ     = 64     # projection dim per branch
D_HIDDEN   = 128    # BiLSTM hidden dim per direction
N_LAYERS   = 2      # BiLSTM layers
DROPOUT    = 0.3
BRANCH_DROPOUT = 0.2

# ─── Training ────────────────────────────────────────────────────────────────
LEARNING_RATE    = 1e-3
LR_MIN           = 1e-6
WEIGHT_DECAY     = 1e-4
BATCH_SIZE       = 32
MAX_EPOCHS       = 100
PATIENCE         = 10     # early stopping on val AUROC
GRAD_CLIP        = 1.0
AUX_LAMBDA       = 0.3    # weight for auxiliary BCE losses
NUM_CLASSES      = 3      # stable / pre-collapse / imminent
VALIDATION_FRAC  = 0.15   # patient-level within each training fold

# BFI scoring
BFI_PRE_WEIGHT  = 0.5
BFI_IMM_WEIGHT  = 1.0

# ─── Alarm logic ─────────────────────────────────────────────────────────────
BFI_ALARM_THRESHOLD   = 70    # BFI ≥ 70
ALARM_SUSTAINED_S     = 120   # must be sustained for 120 s

# ─── Evaluation ──────────────────────────────────────────────────────────────
BOOTSTRAP_ITERS  = 1000
ECE_BINS         = 10
FDR_ALPHA        = 0.01
SHAM_REPEATS     = 10

# ─── Optuna ──────────────────────────────────────────────────────────────────
OPTUNA_TRIALS    = 50
OPTUNA_TIMEOUT   = 3600  # seconds

# ─── Misc ────────────────────────────────────────────────────────────────────
RANDOM_SEED      = 42
NUM_WORKERS      = 4
