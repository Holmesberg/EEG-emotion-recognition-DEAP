# DEAP Emotion Recognition: Subject-Independent Valence Classification

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Repository:** [github.com/Holmesberg/EEG-emotion-recognition-DEAP](https://github.com/Holmesberg/EEG-emotion-recognition-DEAP)

Binary **valence** (high vs low) classification on the [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) EEG using **differential entropy (DE)** features and a gradient boosting classifier. The pipeline contrasts a random train/test split (subject leakage) with **GroupKFold** by subject—the correct framing for **generalization to new users**.

## Key innovation

**Random split vs subject-wise cross-validation.** A random split on EEG trials often mixes the same subject in train and test; the model can exploit subject-specific signals and report **inflated** accuracy on real DEAP data. **GroupKFold** keeps each test fold subject-disjoint, so metrics reflect **new-subject** performance—the relevant benchmark for BCI and affective computing.

**Example (built-in simulation, default 32 subjects × 40 trials, seed 42):**

| Method | Accuracy | Notes |
|--------|----------|--------|
| Random 80/20 split (one run) | **94.14%** | Single holdout; not subject-independent |
| GroupKFold (k=10, by subject) | **95.92% ± 1.65%** (mean ± std across folds) | Subject-independent estimate |

On the same pipeline, a **small** simulation (e.g. `DEAP_SIM_SUBJECTS=8`, `DEAP_SIM_TRIALS=10`) can show a **large** gap (e.g. random split 100% vs GroupKFold ~81%) because of small-sample variance and the toy generative model—**use real DEAP files for publication-grade numbers**. The qualitative lesson remains: always report **who** is in train vs test.

## Quick start

```bash
pip install -r requirements.txt
python DEAP_pipeline.py
```

With no `.dat` files under `data/DEAP/`, the script runs **simulated** EEG (default: 32 subjects × 40 trials). To use real DEAP preprocessed files:

1. Obtain the DEAP dataset (see [official page](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)).
2. Place preprocessed `s*.dat` files in a directory.
3. Point the pipeline at that directory:

```bash
set DEAP_PATH=C:\path\to\your\deap\preprocessed
python DEAP_pipeline.py
```

(On Linux/macOS: `export DEAP_PATH=/path/to/deap/preprocessed`.)

Optional: faster smoke tests (smaller simulated study):

```bash
set DEAP_SIM_SUBJECTS=8
set DEAP_SIM_TRIALS=10
python DEAP_pipeline.py
```

## Technical details

| Item | Description |
|------|-------------|
| **Dataset** | DEAP: 32-channel EEG, 32 subjects, 40 trials each (when using real files); 63 s per trial after baseline removal in this script. |
| **Labels** | Continuous valence; binarized at **4.5** (high ≥ 4.5). |
| **Preprocessing** | Per-channel baseline subtraction (first 3 s), 1–45 Hz bandpass, then 60 s of usable data. |
| **Features** | DE in delta, theta, alpha, beta, gamma → **160** features per trial. |
| **Classifier** | `StandardScaler` + `GradientBoostingClassifier` (fixed `random_state=42`). |
| **Evaluation** | (1) Stratified random 80/20. (2) **GroupKFold** grouped by subject. |

More depth: [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Why subject-independent validation matters

EEG is **not** i.i.d. across trials: trials cluster by person, session, and physiology. A **random** trial split lets the model see subjects in training that reappear in test, which **does not** answer “how well does this work for a **new** user?” GroupKFold and leave-one-subject-out are standard tools for that question.

## Requirements

See `requirements.txt` (`numpy`, `scipy`, `scikit-learn`).

## Author

**Aly Nasser** — built for [g.tec](https://www.gtec.at/) BCI Spring School 2026.

If you use this repo in coursework or a portfolio, cite the DEAP dataset and link this repository.

## License

MIT — see [LICENSE](LICENSE).
