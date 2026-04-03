# BFI Pipeline

This is a pipeline for predicting seizure risk from EEG using a Brain Fragility Index (BFI) score — a number from 0 to 100 that reflects how close a patient's brain activity looks to a pre-seizure state.

---

## What it does

Raw EEG goes in. A score comes out. Everything in between is:

1. **Preprocessing** — resample to 256 Hz, bandpass filter (0.5–45 Hz), notch filter at 50/60 Hz, reject channels with amplitude spikes or flatlines, z-score normalize per channel
2. **Feature extraction** — slice EEG into 10-second windows and compute four families of features:
   - *Spectral* — band power (delta/theta/alpha/beta/gamma), spectral slope, theta/alpha ratio
   - *Complexity* — permutation entropy, sample entropy, detrended fluctuation analysis
   - *Coordination* — coherence between electrode pairs, weighted phase-lag index, phase-amplitude coupling
   - *Network* — graph metrics (global efficiency, modularity, clustering) built from coherence matrices
3. **Sequence modeling** — stack 36 consecutive windows into a sequence and run it through a BiLSTM. Four separate branch encoders first project each feature family into a shared 64-dim space, then a bidirectional LSTM reads the sequence, and a pattern-conditioned attention layer collapses it into a single vector. A linear head outputs 3-class probabilities: *stable*, *pre-event*, *imminent seizure*
4. **BFI score** — a weighted sum of those three probabilities: `BFI = 100 × (0.5 × P(pre-event) + 1.0 × P(imminent))`. If it stays above 70 for 2 minutes, an alarm fires
5. **Evaluation** — leave-one-site-out cross-validation across 8 datasets, bootstrapped AUROC, calibration (ECE + Brier), alarm lead time and false positive rate, sham permutation test, ablation study

---

## Datasets

The pipeline loads from these folders (configure paths in `bfi_pipeline/config.py`):

| Key | Dataset |
|---|---|
| `CHB_MIT` | CHB-MIT scalp EEG (PhysioNet) |
| `TUH` | TUH EEG Seizure Corpus |
| `SIENA` | Siena Scalp EEG |
| `SPARCNET` | SPARCNET |
| `ICARE` | I-CARE (post-cardiac arrest) |
| `ECAMS` | E-CAM-S |
| `SAH_PROS` | SAH prospective cohort |
| `SAH_AUTO` | SAH automated |

All EEG gets remapped to the standard 10-20 montage (19 channels: Fp1, Fp2, F3, F4, ...) before anything else happens.

---

## Files

```
bfi_pipeline/
  config.py              all hyperparameters and paths in one place
  run_pipeline.py        main entrypoint — runs all 7 stages in order

  data/
    loader.py            reads EDF files for each dataset
    harmonize.py         remaps channel names to standard 10-20
    splits.py            leave-one-site-out fold generation

  preprocess/
    filter.py            resampling + bandpass + notch
    artifact.py          amplitude/flatline rejection + z-score normalization
    segment.py           slice into windows and sequences

  features/
    spectral.py          band power, spectral slope
    complexity.py        permutation entropy, sample entropy, DFA
    coordination.py      coherence, wPLI, phase-amplitude coupling
    network.py           graph metrics from coherence matrices
    instability.py       rolling variance, overall instability index
    extract.py           runs all four families and packages them

  model/
    bfi_model.py         the full BFIModel — branches + BiLSTM + attention + heads
    bilstm.py            stacked bidirectional LSTM
    branches.py          four parallel branch encoders
    attention.py         pattern-conditioned multi-head attention
    losses.py            cross-entropy + auxiliary branch losses

  train/
    trainer.py           training loop, early stopping, gradient clipping
    cross_val.py         leave-one-site-out cross-validation
    hyperopt.py          Optuna hyperparameter search

  evaluate/
    metrics.py           bootstrapped AUROC, ECE, Brier score
    alarm.py             alarm logic — lead time, false positive rate
    stats.py             Wilcoxon tests with FDR correction for feature discrimination
    ablation.py          feature family ablation study
    sensitivity.py       first-event-only analysis + sham permutation test

quickstart_chbmit.py     standalone script — loads CHB-MIT, extracts features, runs stats. No model training needed
train_bilstm_chbmit.py   standalone BiLSTM trained directly on CHB-MIT (simpler version)
train_xgb_*.py           XGBoost baselines for individual datasets
```

---

## Running it

**Full pipeline:**
```bash
python bfi_pipeline/run_pipeline.py
```

Add `--skip-hyperopt` to skip the Optuna search and use the defaults from `config.py`. Add `--fast` for a quick sanity check with reduced iterations.

**Just CHB-MIT with no model training:**
```bash
python quickstart_chbmit.py
```

This loads CHB-MIT, extracts spectral/complexity/instability features on pre-seizure and control windows, and runs Wilcoxon tests with FDR correction to see which features discriminate pre-event from stable EEG.

**XGBoost baselines:**
```bash
python train_xgb_chbmit.py
python train_xgb_siena.py
# etc.
```

---

## Output

Results go to `bfi_pipeline/results/paper_metrics.json`. It has:
- Per-site AUROC with 95% bootstrap CI
- Pooled AUROC, ECE, Brier score
- Per-condition breakdown
- Alarm logic stats (mean lead time, FP/h)
- Feature discrimination results
- Sham permutation test results
- Ablation results

---

## Acknowledgments

Code development assisted by Claude (Anthropic). All experimental design, data collection, and scientific interpretation by Ketav Karthikeyan.
