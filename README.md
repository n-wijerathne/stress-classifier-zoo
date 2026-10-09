# Stress Classifier — Zoo Animals (Feeding Schedule & Keeper Routine Focus)

**Module:** IT41043 — Intelligent Systems, Horizon Campus (2026)  
**Milestone:** Milestone 2 — Methodology and Data Description  

### **Authors & Individual Focus**
* **E.N. Sandeepani Wijerathne (ITBIN-2313-0128)** — *The Role of Feeding Schedules and Keeper Routines in Predicting Stress-Indicative Behaviours in Captive Zoo Animals*
* **D.B. Senarathna (ITBIN-2313-0105)** — *Visitor Interaction & Environmental Conditions Focus*

---

## 📌 Project Summary
This project investigates whether a supervised machine learning classifier trained on **feeding-schedule** and **keeper-routine** variables (meal timing, inter-meal interval, diet variety, keeper proximity, keeper arrival time) can detect stress-indicative behaviours in captive zoo animals more accurately than a baseline classifier trained on general movement/trajectory features alone.

---

## ⚠️ Current Data Status
Real field data collection is pending institutional and ethical approval (Section 2.1 of the methodology). To build, validate, and demonstrate the machine learning pipeline end-to-end, `src/generate_synthetic_data.py` produces a synthetic dataset (`data/processed/synthetic_session_bins.csv`) grounded in behavioral domain literature.

Once real data collection is completed, real logs will replace the synthetic file, and the preprocessing and model evaluation pipeline (`src/preprocess.py` onward) will execute unchanged.

---

## 🚦 Project Status & Roadmap
- [x] **Milestone 1**: Research gap, question, and scope finalized
- [x] **Milestone 2**: Preprocessing, feature engineering, baseline, tabular models, and evaluation pipeline implemented on synthetic data
- [ ] Zoo access & ethical approval obtained
- [ ] Real data collection completed
- [ ] LSTM branch trained & benchmarked on real data
- [ ] **Milestone 4**: Final results & paper submission

---

## 📁 Repository Structure
```text
stress-classifier-zoo/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                        # Untouched real logs (gitignored)
│   └── processed/                  # Session-bin tabular data
├── diagrams/
│   └── architecture_diagram.svg    # System architecture diagram
├── src/
│   ├── generate_synthetic_data.py  # Synthetic data generator
│   ├── preprocess.py               # Cleaning, encoding & split-key preparation
│   ├── features.py                 # Feature engineering for all 3 branches
│   ├── baseline_model.py           # Movement/trajectory-only Random Forest
│   ├── models_tabular.py           # Branch A: Random Forest & SVM
│   ├── models_lstm.py              # Branch B: LSTM-FCN sequence model
│   └── evaluate.py                 # 5-fold CV & statistical significance tests
├── notebooks/
│   └── exploratory_analysis.ipynb  # Interactive EDA & visualization notebook
└── tests/
    └── test_preprocess.py          # Unit tests for preprocessing pipeline
```

---

## ⚙️ Installation & Setup

### **1. Clone the Repository**
```bash
git clone https://github.com/n-wijerathne/stress-classifier-zoo.git
cd stress-classifier-zoo
```

### **2. Create & Activate Virtual Environment**
* **On macOS/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
* **On Windows:**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 🏃 Running the Pipeline

### **Step 1: Generate Synthetic Session Data**
```bash
python src/generate_synthetic_data.py
```

### **Step 2: Preprocess Data**
```bash
python src/preprocess.py
```

### **Step 3: Run Model Benchmarking & Statistical Validation**
```bash
python src/evaluate.py
```

### **Step 4: (Optional) Run Deep Learning LSTM Architecture**
```bash
python src/models_lstm.py
```

### **Step 5: Run Unit Tests**
```bash
pytest tests/
```

---

## 📊 Experimental Results Summary

| Model Branch | Feature Set | Model Algorithm | F1-Score | Statistical Test vs. Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Trajectory / Movement Only | Random Forest | **35.93%** | Baseline |
| **Branch A (Proposed)** | Movement + Schedule + Keeper | **Random Forest** | **68.34%** | **Paired $t$-test $p = 0.0365$ ($p < 0.05$)** |
| **Branch A** | Movement + Schedule + Keeper | SVM (RBF) | **26.55%** | $p = 0.1240$ |
| **Branch B** | Sequence Feed & Routine | LSTM-FCN / MLP | **45.69%** | **McNemar Test $p = 0.0002$ ($p < 0.001$)** |

---

## 🔒 Data & Ethics Note
Raw zoo logs will not be committed to this repository per the ethical considerations in Section 2.1 of the methodology (keeper anonymization, animal welfare, institutional terms). Only anonymized, processed session-bin tables are stored under `data/processed/`.

---

## 👥 Authors
* **E.N. Sandeepani Wijerathne** (ITBIN-2313-0128) — *Horizon Campus*
* **D.B. Senarathna** (ITBIN-2313-0105) — *Horizon Campus*
