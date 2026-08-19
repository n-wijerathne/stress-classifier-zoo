# Stress Classifier — Zoo Animals (Feeding Schedule & Keeper Routine Focus)

**Module:** IT41043 — Intelligent Systems, Horizon Campus (2026)
**Milestone:** Milestone 2 — Methodology and Data Description

**Individual focus:** E.N. Sandeepani Wijerathne (ITBIN-2313-0128) — *The Role of Feeding
Schedules and Keeper Routines in Predicting Stress-Indicative Behaviours in Captive Zoo Animals*

**Group partner:** D.B. Senarathna (ITBIN-2313-0105) — visitor-interaction & environmental-conditions focus

## Project Summary

This project investigates whether a supervised machine learning classifier trained on
feeding-schedule and keeper-routine variables (meal timing, inter-meal interval, diet
variety, keeper proximity, keeper arrival time) can detect stress-indicative behaviours
in captive zoo animals more accurately than a baseline classifier trained on general
movement/trajectory features alone. Full methodology, dataset description, model
architecture, baseline definition, and evaluation plan are documented in the
Milestone 2 report (not included in this repo for size/format reasons — see the
module submission portal).

## ⚠️ Current Data Status

**Real field data has not yet been collected.** Zoo access and the required
institutional/ethical approvals (Section 2.1 of the methodology) are still pending.

To allow the pipeline below to be built, tested, and demonstrated ahead of approval,
`src/generate_synthetic_data.py` produces a **synthetic dataset** that follows the
schema and statistical assumptions described in the methodology (assumptions are
documented in the script itself and grounded in the cited literature). **Synthetic
output must never be presented as real collected data.** Once real data collection is
complete, it will replace the synthetic file at the same path
(`data/processed/synthetic_session_bins.csv`) and the rest of the pipeline
(`preprocess.py` onward) will run unchanged.

## Project Status

- [x] Research gap, question, and scope finalised (Milestone 1)
- [x] Preprocessing, feature engineering, baseline, tabular-model, and evaluation
      pipeline implemented and tested against synthetic data
- [ ] Zoo access / ethical approval obtained
- [ ] Real data collection
- [ ] LSTM branch trained and compared on real data
- [ ] Final results (Milestone 4)

## Repository Structure

```
stress-classifier-zoo/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                        # untouched real logs (gitignored — never committed)
│   └── processed/                  # synthetic + (later) real session-bin tables
├── diagrams/
│   └── architecture_diagram.svg    # system architecture diagram (vector)
├── src/
│   ├── generate_synthetic_data.py  # synthetic data generator (see warning above)
│   ├── preprocess.py               # cleaning, encoding, split-key preparation
│   ├── features.py                 # feature engineering for all 3 model branches
│   ├── baseline_model.py           # movement/trajectory-only Random Forest
│   ├── models_tabular.py           # Branch A: Random Forest + SVM
│   ├── models_lstm.py              # Branch B: LSTM-FCN sequence model
│   └── evaluate.py                 # day-grouped stratified CV + significance tests
├── notebooks/
│   └── exploratory_analysis.ipynb  # (to be added)
└── tests/
    └── test_preprocess.py
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Pipeline (on synthetic data)

```bash
# 1. Generate synthetic data (stand-in until real data is approved/collected)
python3 src/generate_synthetic_data.py

# 2. Preprocess
python3 src/preprocess.py

# 3. Run cross-validated evaluation: baseline vs. proposed tabular model
python3 src/evaluate.py

# 4. (Optional, requires `pip install tensorflow`) sanity-check the LSTM architecture
python3 src/models_lstm.py
```

## Data & Ethics Note

Raw zoo logs (once collected) will **not** be committed to this repository, per the
ethical considerations in Section 2.1 of the methodology (keeper anonymisation, animal
welfare, institutional data-sharing terms). Only anonymised, processed session-bin
tables — or, until then, clearly-labelled synthetic data — are stored under
`data/processed/`.

## Authors

- E.N. Sandeepani Wijerathne (ITBIN-2313-0128)
- D.B. Senarathna (ITBIN-2313-0105)
