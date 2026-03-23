"""
DEAP emotion recognition pipeline (subject-independent evaluation).

Loads preprocessed DEAP EEG trials (or simulates data when no files are present),
extracts differential entropy (DE) features per frequency band, and compares
a random train/test split (optimistic, subject leakage) with GroupKFold by
subject (scientifically appropriate for generalization). Intended as a minimal,
reproducible reference for BCI / affective computing coursework and research.
"""

import glob
import os
import pickle
import warnings

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Config ──────────────────────────────────────────────────────
DEAP_DIR = os.getenv("DEAP_PATH", os.path.join("data", "DEAP"))
FS = 128
N_CHANNELS = 32
VALENCE_THRESHOLD = 4.5
BASELINE_SAMPLES = 3 * FS  # 384 samples
TRIAL_SAMPLES = 60 * FS  # 7680 samples (after baseline removal)

FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 14),
    "beta": (14, 31),
    "gamma": (31, 45),
}


# ── Filters ──────────────────────────────────────────────────────
def butter_bandpass(data, low, high, fs=FS, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1)


# ── DE for one window, one channel ───────────────────────────────
def de_segment(x):
    s = np.std(x)
    return 0.0 if s < 1e-10 else 0.5 * np.log(2 * np.pi * np.e * s**2)


# ── Feature extraction: DE per (channel, band) ───────────────────
def extract_de(eeg):
    """
    eeg: (32, T) - already preprocessed
    Returns: (32 * 5,) = 160 DE features
    """
    n_ch, T = eeg.shape
    win = FS  # 1-second windows
    n_wins = T // win
    feats = np.zeros((n_ch, len(FREQ_BANDS)), dtype=np.float32)

    for b_i, (_, (lo, hi)) in enumerate(FREQ_BANDS.items()):
        band = butter_bandpass(eeg, lo, hi)  # (32, T)
        de_wins = np.zeros((n_wins, n_ch), dtype=np.float32)
        for w in range(n_wins):
            seg = band[:, w * win : (w + 1) * win]  # (32, 128)
            de_wins[w] = [de_segment(seg[c]) for c in range(n_ch)]
        feats[:, b_i] = de_wins.mean(axis=0)

    return feats.flatten()  # (160,)


# ── Load real DEAP data ───────────────────────────────────────────
def load_deap(data_dir):
    """Load and extract features subject-by-subject to save RAM."""
    files = sorted(glob.glob(os.path.join(data_dir, "s*.dat")))
    if not files:
        return None, None, None
    X_list, y_list, subj_list = [], [], []
    for i, fp in enumerate(files):
        with open(fp, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        eeg = d["data"][:, :N_CHANNELS, :].astype(np.float32)  # (40, 32, 8064)
        labels = d["labels"][:, 0]  # valence
        # Extract features immediately, then discard raw EEG
        feats = build_feature_matrix(eeg)
        X_list.append(feats)
        y_list.append(labels)
        subj_list.extend([i + 1] * len(labels))
        del eeg, d  # free RAM immediately
        print(f"  Subject {i+1:02d}: 40 trials | features extracted")
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y, np.array(subj_list)


# ── Simulate DEAP data (memory-efficient) ────────────────────────
def simulate_deap(n_subjects=32, n_trials=40, seed=42):
    print("\nNo DEAP .dat files found - running on SIMULATED data.")
    print("Results demonstrate the pipeline; real metrics need real data.\n")
    rng = np.random.default_rng(seed)
    total = n_subjects * n_trials
    T = TRIAL_SAMPLES + BASELINE_SAMPLES  # 8064
    X = np.zeros((total, N_CHANNELS, T), dtype=np.float32)
    y = np.zeros(total, dtype=np.float32)
    subjects = np.zeros(total, dtype=int)
    t_vec = np.linspace(0, T / FS, T)
    idx = 0
    for s in range(n_subjects):
        scale = 0.8 + 0.4 * rng.random()
        for _t in range(n_trials):
            valence = rng.uniform(1, 9)
            y[idx] = valence
            subjects[idx] = s + 1
            boost = 1 + 0.3 * (valence / 9)
            sig = (rng.standard_normal((N_CHANNELS, T)) * scale).astype(np.float32)
            f_a = rng.uniform(9, 12)
            sig += (boost * 0.5 * np.sin(2 * np.pi * f_a * t_vec)).astype(np.float32)
            X[idx] = sig
            idx += 1
    print(
        f"  Simulated {total} trials | {n_subjects} subjects | "
        f"shape {X.shape}"
    )
    return X, y, subjects


# ── Preprocessing (applied trial-by-trial to save memory) ────────
def preprocess_one(trial):
    """trial: (32, 8064) -> (32, 7680) after baseline removal + bandpass"""
    # Baseline correction
    baseline = trial[:, :BASELINE_SAMPLES].mean(axis=1, keepdims=True)
    trial = trial - baseline
    # Broadband bandpass 1–45 Hz
    trial = butter_bandpass(trial, 1, 45)
    # Remove baseline segment
    return trial[:, BASELINE_SAMPLES:]  # (32, 7680)


# ── Build classifier pipeline ────────────────────────────────────
def build_clf():
    """
    StandardScaler + sklearn GradientBoostingClassifier (tree boosting;
    API similar in spirit to XGBoost but implemented in scikit-learn).
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.07,
                    max_depth=4,
                    subsample=0.8,
                    min_samples_leaf=5,
                    random_state=42,
                ),
            ),
        ]
    )


# ── Full feature extraction pass ─────────────────────────────────
def build_feature_matrix(X_raw):
    print("  Extracting DE features (trial-by-trial)...")
    N = X_raw.shape[0]
    n_features = N_CHANNELS * len(FREQ_BANDS)  # 160
    X_feat = np.zeros((N, n_features), dtype=np.float32)
    for i in range(N):
        proc = preprocess_one(X_raw[i])
        X_feat[i] = extract_de(proc)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{N} trials processed")
    print(f"  Feature matrix: {X_feat.shape}")
    return X_feat


# ── Method 1: Random Split ────────────────────────────────────────
def eval_random(X, y, seed=42):
    print("\n" + "=" * 58)
    print("  METHOD 1: Random Train/Test Split (80/20)")
    print("=" * 58)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    clf = build_clf()
    clf.fit(X_tr, y_tr)
    y_p = clf.predict(X_te)
    acc = accuracy_score(y_te, y_p)
    bal = balanced_accuracy_score(y_te, y_p)
    print(f"\n  Train: {len(y_tr)}  |  Test: {len(y_te)}")
    print(f"  Accuracy         : {acc*100:.2f}%")
    print(f"  Balanced Accuracy: {bal*100:.2f}%")
    print("\n  Classification Report:")
    print(
        classification_report(
            y_te,
            y_p,
            target_names=["Low Valence", "High Valence"],
            digits=4,
        )
    )
    print(f"  Confusion Matrix:\n{confusion_matrix(y_te, y_p)}")
    return acc, bal


# ── Method 2: GroupKFold ──────────────────────────────────────────
def eval_groupkfold(X, y, groups):
    n_subj = len(np.unique(groups))
    k = min(10, n_subj)
    print("\n" + "=" * 58)
    print(f"  METHOD 2: GroupKFold  (k={k}, grouped by subject)")
    print("=" * 58)
    gkf = GroupKFold(n_splits=k)
    fold_acc, fold_bal = [], []
    all_true, all_pred = [], []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        clf = build_clf()
        clf.fit(X[tr], y[tr])
        y_p = clf.predict(X[te])
        a = accuracy_score(y[te], y_p)
        b = balanced_accuracy_score(y[te], y_p)
        fold_acc.append(a)
        fold_bal.append(b)
        all_true.extend(y[te])
        all_pred.extend(y_p)
        test_subjs = np.unique(groups[te])
        print(
            f"  Fold {fold+1:2d} | Subjects {test_subjs} | "
            f"Acc={a*100:.2f}%  BalAcc={b*100:.2f}%"
        )

    mean_a, std_a = np.mean(fold_acc), np.std(fold_acc)
    mean_b, std_b = np.mean(fold_bal), np.std(fold_bal)
    print(f"\n  Mean Accuracy         : {mean_a*100:.2f}% +/- {std_a*100:.2f}%")
    print(f"  Mean Balanced Accuracy: {mean_b*100:.2f}% +/- {std_b*100:.2f}%")
    print("\n  Aggregate Classification Report:")
    print(
        classification_report(
            all_true,
            all_pred,
            target_names=["Low Valence", "High Valence"],
            digits=4,
        )
    )
    print(f"  Aggregate Confusion Matrix:\n{confusion_matrix(all_true, all_pred)}")
    return mean_a, std_a, mean_b, std_b


# ── Discussion ────────────────────────────────────────────────────
def discussion(rand_acc, gkf_acc, gkf_std):
    diff = (rand_acc - gkf_acc) * 100
    print("\n" + "=" * 58)
    print("  DISCUSSION: Random split vs subject-independent CV")
    print("=" * 58)
    print(
        f"""
  Random split (one 80/20 holdout)     : {rand_acc*100:.2f}%
  GroupKFold mean (by subject)         : {gkf_acc*100:.2f}% +/- {gkf_std*100:.2f}%
  Single-holdout minus GroupKFold mean : {diff:+.2f} percentage points

  INTERPRETATION
  --------------
  On real DEAP recordings, a random trial split often looks better than
  subject-wise CV because train and test both contain trials from the same
  people (subject leakage). The classifier can exploit idiosyncratic EEG
  traits instead of valence alone.

  On the built-in simulation, numbers can flip or look similar: labels and
  signals are synthetic, and one random 80/20 draw is a high-variance
  estimate. The methodological point is unchanged: GroupKFold (or LOSO)
  answers "how well does this work for new subjects?" - the question that
  matters for BCIs and affective interfaces.

  1. SUBJECT LEAKAGE (random split)
     Trials from the same person can appear in train AND test. EEG carries
     subject-specific structure; the model may learn people, not emotion.

  2. NON-IID TRIALS
     Trials from one subject are correlated; random splitting breaks the
     i.i.d. assumption used by many textbook ML setups.

  3. GROUPKFOLD
     Each test fold holds out entire subjects. Use this (or LOSO) for
     reporting generalization to new users.

  PRACTICAL INSIGHT
  -----------------
  Compare subject-independent scores across papers and models. If you only
  tune on a random split, expect optimistic performance versus deployment
  on unseen subjects; domain adaptation and subject-aware normalization are
  active research directions.
"""
    )


# ── Main ──────────────────────────────────────────────────────────
def main():
    print("=" * 58)
    print("  DEAP Emotion Recognition Pipeline")
    print("  Binary Valence Classification (High / Low)")
    print("=" * 58)

    # 1. Load + extract features (memory-efficient, subject by subject)
    print("\n[1/2] Loading & extracting DE features per subject...")
    print(f"  DEAP data directory: {os.path.abspath(DEAP_DIR)}")
    X_feat, y_cont, subjects = load_deap(DEAP_DIR)
    if X_feat is None:
        n_subj = int(os.getenv("DEAP_SIM_SUBJECTS", "32"))
        n_tr = int(os.getenv("DEAP_SIM_TRIALS", "40"))
        X_raw, y_cont, subjects = simulate_deap(n_subjects=n_subj, n_trials=n_tr)
        X_feat = build_feature_matrix(X_raw)
        del X_raw

    # Binarise labels
    y_bin = (y_cont >= VALENCE_THRESHOLD).astype(int)
    print(f"\n  Feature matrix: {X_feat.shape}")
    print(
        f"  Labels -> Low: {(y_bin==0).sum()}  High: {(y_bin==1).sum()}  "
        f"(threshold={VALENCE_THRESHOLD})"
    )

    # 2. Evaluate
    print("\n[2/2] Evaluating...")
    rand_acc, rand_bal = eval_random(X_feat, y_bin)
    gkf_acc, gkf_std, _, _ = eval_groupkfold(X_feat, y_bin, subjects)

    # 3. Discussion
    discussion(rand_acc, gkf_acc, gkf_std)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        main()
