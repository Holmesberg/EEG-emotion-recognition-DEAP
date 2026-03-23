# Methodology (DEAP valence pipeline)

## Differential entropy (DE)

For a short window of band-pass filtered EEG \(x(t)\), differential entropy under a Gaussian assumption is

\[
h = \tfrac{1}{2} \log(2\pi e \sigma^2)
\]

where \(\sigma^2\) is the variance of the window. The implementation uses one-second windows (128 samples at 128 Hz) per channel, averages DE across windows, and repeats for each frequency band.

## Frequency bands

Features use five bands (Hz): delta 1–4, theta 4–8, alpha 8–14, beta 14–31, gamma 31–45. These are common choices in affective EEG work; boundaries are not universal—report them whenever you compare to other studies.

## Preprocessing (per trial)

1. Baseline removal: subtract the mean of the first 3 s (384 samples) per channel.
2. Broadband filter: Butterworth bandpass 1–45 Hz on the full trial.
3. Discard the baseline segment; classify on the remaining 60 s (7680 samples) at 128 Hz.

## Features

Per trial: 32 channels \(\times\) 5 bands = **160** DE values (one scalar per channel-band after averaging over windows).

## Classifier

`StandardScaler` + `GradientBoostingClassifier` (scikit-learn) with fixed `random_state=42` for reproducibility. This is a standard tree boosting baseline—not XGBoost/LightGBM, but comparable in role (nonlinear decision boundaries on tabular features).

## Evaluation

- **Random 80/20 split**: stratified by binary label. **Not** subject-independent; included to show how leakage can distort conclusions.
- **GroupKFold** by subject ID: `k = min(10, n_subjects)`. Each fold holds out disjoint subjects; the reported mean and standard deviation across folds approximate performance for **new** users.

## Subject-independent vs subject-dependent

- **Subject-dependent**: train and test on the same pool of people (e.g. random trial split). Often optimistic for "will this work on a new person?"
- **Subject-independent**: never test on a subject seen in training for that fold. Appropriate for BCI and real-world deployment narratives.

## References

- Koelstra et al., "DEAP: A Database for Emotion Analysis using Physiological Signals," IEEE TPAMI 2011.
- Pedregosa et al., "Scikit-learn: Machine Learning in Python," JMLR 2011.
