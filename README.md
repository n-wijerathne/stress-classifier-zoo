# Stress Classifier - Zoo Animals (Feeding Schedule & Keeper Routine Focus)

**Module:** IT41043 - Intelligent Systems, Horizon Campus (2026)
**Research title:** Machine Learning-Based Identification of Stress Behaviours in Zoo Animals
**Individual focus:** E.N. Sandeepani Wijerathne (ITBIN-2313-0128) - *The Role of Feeding Schedules and Keeper Routines in Predicting Stress-Indicative Behaviours in Captive Zoo Animals*
**Group partner:** D.B. Senarathne (ITBIN-2313-0105)

## Project summary

**Research question.** Can a supervised machine-learning classifier trained on feeding-schedule and
keeper-routine variables (inter-meal interval, diet variety, keeper proximity, keeper arrival and
meal timing) identify stress-indicative behaviours in captive zoo animals more accurately than a
baseline classifier trained on general movement/trajectory features alone? The question is
falsifiable: the analysis reports the result at alpha = 0.05 whether or not the difference is significant.

**About the project.** *Machine Learning-Based Identification of Stress Behaviours in Zoo Animals*
applies established classifiers (Random Forest, SVM, gradient boosting, logistic regression, MLP) to
a new feature domain - feeding and keeper-routine data - using **primary data we collected ourselves**
at the National Zoological Gardens, Dehiwala, from Asian elephants (*Elephas maximus*) and toque
macaques (*Macaca sinica*). Every model is compared with a realistic movement-based baseline under
identical day-grouped cross-validation. The partner study covers visitor and environmental variables.

## Project status

- [x] Research gap, question and scope finalised (Milestone 1)
- [x] Methodology and data description (Milestone 2)
- [x] **Real data collected** (190 animal-sessions, 28 observation days, 2026-04-01 to 2026-05-25)
- [x] Preprocessing, feature engineering, baseline, tabular models and evaluation pipeline run on the **real data**
- [x] Day-grouped cross-validation, significance tests, ablation, permutation importance, error analysis
- [x] Final Random Forest model trained and saved (`models/`)
- [ ] Paper draft (Milestone 3) and final paper (Milestone 4)

> The synthetic-data generator from the Milestone 2 repository has been **removed**. All results
> below come from the real field data.

## Key results (real data)

Primary metric: F1 on the stress-present class. Day-grouped, stratified 5-fold CV on 24 development
days (162 sessions, 32% stress-positive); 4 further days (28 sessions) were held out as a test set.
Full tables are in `results/`.

| Model (proposed feature set) | F1 | Precision | Recall | AUC-ROC |
|---|---|---|---|---|
| **Random Forest** | **0.691 ± 0.084** | 0.767 ± 0.080 | 0.635 ± 0.108 | 0.854 ± 0.064 |
| Gradient Boosting | 0.633 ± 0.141 | 0.657 ± 0.161 | 0.613 ± 0.129 | 0.826 ± 0.074 |
| SVM (RBF) | 0.529 ± 0.075 | 0.492 ± 0.083 | 0.593 ± 0.139 | 0.691 ± 0.067 |
| Logistic Regression | 0.511 ± 0.093 | 0.506 ± 0.187 | 0.558 ± 0.148 | 0.726 ± 0.101 |
| MLP | 0.461 ± 0.124 | 0.520 ± 0.145 | 0.425 ± 0.138 | 0.644 ± 0.071 |

**Baseline vs proposed (Random Forest, same folds):** baseline F1 = 0.325, proposed F1 = 0.677 over
10 repeated CV runs; the proposed model was better in 10/10 repeats. Paired t-test on the 5 folds
p = 0.003; Nadeau-Bengio corrected test on repeated CV p < 0.001.

**Ablation (RF, 10x repeated CV).**

| Feature set | F1 | AUC-ROC |
|---|---|---|
| Movement only | 0.360 ± 0.097 | 0.567 ± 0.091 |
| Baseline (movement + context) | 0.325 ± 0.093 | 0.542 ± 0.090 |
| Baseline + `time_to_keeper_arrival_min` only | 0.675 ± 0.132 | 0.849 ± 0.087 |
| Proposed (movement + context + all feeding/keeper features) | 0.677 ± 0.147 | 0.844 ± 0.082 |
| Proposed + rolling stereotypy rate | 0.668 ± 0.132 | 0.858 ± 0.068 |

**What this means - and what it does not.**
- The gain comes almost entirely from **one feature, time to keeper arrival**. Adding it alone to the baseline reproduces the full proposed model's F1.
- The model finds stress mainly when the keeper's arrival is imminent (mean 3.4 min for correctly detected stress sessions vs 19.5 min for missed ones), so stress occurring far from keeper arrival is largely missed (overall miss rate 37%; see `results/error_analysis.csv`).
- Held-out test days (28 sessions, 9 stress-positive): proposed F1 0.50 (precision 1.00, recall 0.33, AUC 0.81) vs baseline F1 0.47 (AUC 0.54); McNemar p = 0.51. This set is far too small to be conclusive, so **cross-validation is the primary evidence**.
- Rolling stereotypy rate is reported only as an extended set because it is behaviour-derived and may overlap with how the label was assigned.

## Differences from the Milestone 2 methodology (and why)

| Milestone 2 plan | What the real data allowed |
|---|---|
| ~4,000 five-minute bins | **190 session-level rows** (one per animal-session); the session table is the unit of analysis |
| Branch B: LSTM-FCN on windowed sequences | **Not trained** - there are no 5-minute sequences to window. An MLP is included as a neural comparison model |
| SMOTE on training folds | **Class weights only** (`class_weight="balanced"`); SMOTE is not applied |
| Baseline = movement features only | Baseline also receives species/enclosure/session, so the comparison isolates the feeding/keeper features. A movement-only run is in the ablation |
| 70:15:15 split | ~86:14 day-grouped development/test split (one of 7 folds) plus 5-fold CV inside the development days, because the dataset is small |
| Individual ID one-hot encoded | Individual IDs are not used and are removed from the public file |

## Repository structure

```
stress-classifier-zoo/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── README.md                   # data dictionary
│   ├── raw/                        # full-detail logs (gitignored - never committed)
│   └── processed/animal_stress_data.csv   # real, anonymised session-level data
├── diagrams/architecture_diagram.svg      # system architecture (vector)
├── src/
│   ├── config.py                   # paths and constants
│   ├── preprocess.py               # validation, cleaning, derived features, day-group key
│   ├── features.py                 # feature groups + scikit-learn preprocessor
│   ├── baseline_model.py           # movement/context-only Random Forest
│   ├── models_tabular.py           # RF, SVM, GB, LR, MLP pipelines
│   ├── evaluate.py                 # day-grouped CV, statistics, ablation, error analysis
│   ├── train_final.py              # fit + save final model
│   └── predict.py                  # score new sessions
├── notebooks/exploratory_analysis.ipynb
├── results/                        # tables and figures produced by evaluate.py
├── models/                         # saved Random Forest + model card
└── tests/                          # pytest suite (preprocessing, features, leakage checks)
```

## Setup

Tested with **Python 3.12** (scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4, scipy 1.17.1,
statsmodels 0.15.0, matplotlib 3.10.8). Older versions down to Python 3.10 should work (see `requirements.txt`).

```bash
git clone https://github.com/n-wijerathne/stress-classifier-zoo.git
cd stress-classifier-zoo
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the results

```bash
python src/preprocess.py     # validate + engineer features -> data/processed/model_ready.csv
python src/evaluate.py       # all experiments -> results/   (about 1-2 minutes)
python src/train_final.py    # fit and save the final Random Forest -> models/
python -m pytest tests/      # run the test suite
```

Score new sessions (same columns as `data/processed/animal_stress_data.csv`, label column optional):

```bash
python src/predict.py --input new_sessions.csv --output predictions.csv
```

All random seeds are fixed (`src/config.py`, seed 42), so results are reproducible. Saved model files
are scikit-learn-version specific; re-run `train_final.py` if loading fails.

## Validation design (no leakage)

- **Splits are by observation day**, never by row: no day appears in both training and test data (checked by assertions and tests).
- Stratified day-grouped 5-fold CV on development days; a separate held-out test set is not used for model selection.
- Scaling and one-hot encoding are fitted **inside each training fold only** (scikit-learn `Pipeline`).
- Class imbalance is handled with class weights.
- Statistics: Shapiro-Wilk then paired t-test / Wilcoxon on fold F1; 10x repeated CV with the Nadeau-Bengio corrected t-test; McNemar on the held-out days.

## Data & ethics

- **Data source:** primary data collected by the authors at the National Zoological Gardens, Dehiwala, with the permission of zoo management: scan-sampled behavioural observations plus keeper/feeding logs. No animals were approached, fed or interacted with, and no physiological sampling was done.
- **Raw logs are not stored in this repository.** `data/raw/` is gitignored. The repository contains only the processed, anonymised session-level table in `data/processed/`.
- **No keeper names** are recorded in the data, and **individual animal identifiers are removed** from the public file because Asian elephants are IUCN-Endangered; results are reported at species/group level only.
- No visitor or environmental data are included here (partner study).

## Limitations

Small dataset (190 sessions, 28 days) from one zoo and two species; a small held-out test set; the
effect is dominated by one feature; labels come from manual behavioural coding. Results should be read as
a proof of concept, not a deployable welfare tool.

## Contributions

<!-- EDIT BEFORE PUSHING: replace with the actual split of work. -->
| Member | Contribution |
|---|---|
| E.N. Sandeepani Wijerathne (ITBIN-2313-0128) | Field data collection; *(add: coding, analysis, writing ...)* |
| D.B. Senarathne (ITBIN-2313-0105) | Field data collection; *(add: coding, analysis, writing ...)* |

## References

The full reference list is in the Milestone 2 report; the paper reference list will be added at Milestone 4.

## License

MIT - see [LICENSE](LICENSE).
