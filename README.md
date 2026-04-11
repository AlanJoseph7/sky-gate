# ✈️ SkyGate — ADS-B Anomaly Detection System

Aviation Cybersecurity · ADS-B · Anomaly Detection

> **"ADS-B signals are unencrypted and unauthenticated — anyone can inject fake aircraft into the sky. SkyGate detects them."**

---

## 🎯 What Is This?

SkyGate is a **real-time aviation cybersecurity system** that detects spoofed and anomalous aircraft behaviour in ADS-B (Automatic Dependent Surveillance–Broadcast) data using a three-layer AI/ML detection pipeline.

ADS-B is the system commercial aircraft use to broadcast their position. Because it has **no authentication or encryption**, attackers can:
- Inject fake aircraft into radar displays
- Teleport real aircraft to false positions
- Spoof impossible flight behaviour to confuse air traffic control

SkyGate is a **detection system** — it identifies these attacks using rules, machine learning, and deep learning working together. Models are trained on a combination of **live ADS-B feeds** (via OpenSky Network and adsb.lol) and **synthetic anomaly-injected data**, and the system supports **real-time monitoring** over Indian airspace.

---

## 🏆 Results

| Detector | Precision | Recall | F1 Score |
|---|---|---|---|
| Rule-Based | 0.9055 | 0.7625 | **0.8279** |
| Isolation Forest | 0.4631 | 0.6693 | 0.5474 |
| Autoencoder (DL) | 0.5601 | 0.4580 | 0.5039 |
| 🏅 **Final Ensemble** | **0.7097** | **0.8307** | **0.7654** |

> The ensemble catches **83% of all anomalies** while maintaining **71% precision** — outperforming every individual detector on recall.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   SKYGATE PIPELINE                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  [1] generator.py  →  ADS-B Data Ingestion              │
│         ↓              Live feed: OpenSky + adsb.lol     │
│                        Synthetic: anomaly injection      │
│                        6 anomaly types · mixed dataset   │
│                                                          │
│  [2] features.py   →  Feature Engineering               │
│         ↓              climb rate · acceleration         │
│                        heading change · distance         │
│                                                          │
│  [3] rules.py      →  Rule-Based Detection              │
│         ↓              Physics threshold violations      │
│                        F1: 0.83 · Precision: 0.91       │
│                                                          │
│  [4] isolation_    →  Isolation Forest (ML)             │
│      forest.py ↓      Unsupervised · trained on normals  │
│                        F1: 0.55 · Recall: 0.67          │
│                                                          │
│  [5] lstm_model.py →  Feedforward Autoencoder (DL)      │
│         ↓              Reconstruction error detection    │
│                        F1: 0.50 · Bottleneck: 2 neurons  │
│                                                          │
│  [6] detect.py     →  Weighted Ensemble                 │
│         ↓              Dynamic F1-based weights          │
│                        F1: 0.77 · Recall: 0.83          │
│                                                          │
│  [7] evaluation.py →  Full Metrics Report               │
│  [8] visualize.py  →  4 Diagnostic Plots                │
│  [9] live_monitor  →  Real-Time ADS-B Feed              │
│                        OpenSky + adsb.lol · 30s interval │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 🚨 Anomaly Types Detected

| Anomaly | Type | Detector |
|---|---|---|
| **Altitude Jump** | Sudden ±5000–10000ft spike | Rules + IF |
| **Teleportation** | Impossible lat/lon jump | Rules + IF |
| **Speed Spike** | Velocity exceeds physical limits | Rules + IF |
| **Gradual Speed Drift** | Compounding acceleration over 10 steps | Autoencoder |
| **Altitude Oscillation** | Rapid ±1500–2500ft oscillation pattern | Autoencoder |
| **Heading Oscillation** | Zigzag ±20–45° heading pattern | Autoencoder |

---

## 🧠 Technical Stack

```
Language     : Python 3.11
ML           : scikit-learn (Isolation Forest, MinMaxScaler)
Deep Learning: TensorFlow 2.16 + tf-keras (Feedforward Autoencoder)
Data         : pandas, numpy
Live Feed    : OpenSky Network API · adsb.lol API (Indian airspace)
Visualisation: matplotlib
Environment  : CPU-only · 16GB RAM · No GPU required · Zero cost
```

---

## 📁 Project Structure

```
SkyGate/
│
├── src/
│   ├── generator.py          # ADS-B data ingestion (live + synthetic)
│   ├── features.py           # Feature engineering pipeline
│   ├── rules.py              # Rule-based anomaly detection
│   ├── isolation_forest.py   # Isolation Forest detector
│   ├── lstm_model.py         # Feedforward Autoencoder detector
│   ├── detect.py             # Weighted ensemble combiner
│   ├── evaluation.py         # Metrics and evaluation
│   ├── visualize.py          # Plot generation
│   ├── main.py               # Single entry point
│   └── utils.py              # Shared constants and path helpers
│
├── data/
│   ├── adsb_raw.csv          # Raw ADS-B data (live + synthetic)
│   ├── adsb_features.csv     # Engineered features
│   ├── adsb_rules.csv        # After rule detection
│   ├── adsb_if.csv           # After Isolation Forest
│   ├── adsb_lstm.csv         # After Autoencoder
│   ├── adsb_final.csv        # Final ensemble results
│   └── plots/                # Generated visualisations
│
├── models/                   # Saved model files
├── logs/                     # Training and live monitoring logs
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/yourusername/skygate.git
cd skygate
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. Configure API credentials

Add your credentials to `utils.py` or a `.env` file:

```python
OPENSKY_USERNAME = "your_opensky_username"
OPENSKY_PASSWORD = "your_opensky_password"
# adsb.lol is open — no key required
```

### 3. Run the full pipeline (live + synthetic)

```bash
cd src
python main.py
```

That's it. One command runs all steps sequentially — fetching live ADS-B data, injecting synthetic anomalies for training, running the detection pipeline, and printing a final summary table.

### 4. Run live monitoring mode

```bash
cd src
python main.py --live
```

Fetches real ADS-B data over Indian airspace every 30 seconds and runs the full detection pipeline continuously.

### 5. Run individual steps

```bash
cd src
python generator.py          # Fetch live data + generate synthetic data
python features.py           # Engineer features
python rules.py              # Rule-based detection
python isolation_forest.py   # Isolation Forest
python lstm_model.py         # Autoencoder
python detect.py             # Ensemble
python evaluation.py         # Evaluate
python visualize.py          # Plot
```

---

## 📊 Visual Output

The pipeline generates 4 plots saved to `data/plots/`:

| Plot | Description |
|---|---|
| `trajectory.png` | Flight paths with anomalies marked in red |
| `lstm_error.png` | Reconstruction error over time with threshold line |
| `error_distribution.png` | Normal vs anomaly error histogram — shows model separation |
| `detector_comparison.png` | Anomaly counts per detector vs ground truth |

---

## 🔍 How the Ensemble Works

Each detector votes with a weight proportional to its F1 score:

```python
weight_rule = F1_rule / (F1_rule + F1_IF + F1_autoencoder)
weight_IF   = F1_IF   / (F1_rule + F1_IF + F1_autoencoder)
weight_AE   = F1_AE   / (F1_rule + F1_IF + F1_autoencoder)

final_score = weight_rule * rule_anomaly +
              weight_IF   * if_anomaly   +
              weight_AE   * lstm_anomaly

# Best threshold found via F1 sweep over [0.1, 1.0]
final_anomaly = final_score >= best_threshold
```

This means **if the autoencoder degrades, it automatically loses voting power**. The ensemble is self-correcting.

---

## 💡 Key Design Decisions

**Why use both live and synthetic data?**
Live ADS-B data from OpenSky and adsb.lol provides real-world normal flight behaviour over Indian airspace, ensuring the models learn genuine traffic patterns. Synthetic anomalies are then injected into this real data for supervised training and evaluation, since real labelled attack data is not publicly available.

**Why a feedforward autoencoder instead of LSTM?**
The LSTM autoencoder was initially used but produced an anomaly/normal reconstruction ratio of ~0.97x — effectively random. The feedforward autoencoder evaluates each row independently, making anomalous feature combinations directly visible as reconstruction error spikes. Final ratio: 1.55x.

**Why dynamic F1 weights in the ensemble?**
Hardcoded weights (e.g. 0.4/0.3/0.3) don't adapt to actual detector performance. Dynamic weights mean the ensemble automatically favours whichever detector is strongest on the current dataset.

**Why clip outliers before scaling?**
Anomaly values (e.g. teleport: 500km/step) dominated the MinMaxScaler range, collapsing all normal variation to near zero. Clipping to 4σ of normal data preserves normal feature variation while keeping anomaly signal intact.

---

## 🎤 Interview Summary

> *"I built SkyGate — a real-time ADS-B anomaly detection system for aviation cybersecurity. ADS-B signals are unauthenticated, making them vulnerable to spoofing attacks. The system ingests live flight data from the OpenSky Network and adsb.lol APIs over Indian airspace, and uses a three-layer pipeline: physics-based rule detection, unsupervised Isolation Forest, and a feedforward autoencoder trained on normal flight data. Synthetic anomalies are injected into real ADS-B data for model training and evaluation. The three detectors are combined using a weighted ensemble where weights are derived dynamically from each detector's F1 score. The final system achieves 0.83 recall and 0.77 F1, catching 83% of injected anomalies including teleportation, speed spikes, and subtle multi-step flight pattern drift."*

---

## 📋 Requirements

```
numpy==1.26.4
pandas==2.2.2
matplotlib==3.9.0
scikit-learn==1.5.0
tensorflow-cpu==2.16.1
tf-keras==2.16.0
keras==3.3.3
requests==2.31.0
```

---

## ⚠️ Constraints & Properties

- ✅ Zero cost — OpenSky and adsb.lol are free APIs
- ✅ Fully local — runs on CPU, no GPU required
- ✅ Real-time capable — live monitoring over Indian airspace
- ✅ Reproducible — seeded RNG throughout (`RANDOM_SEED = 42`)
- ✅ Modular — each pipeline step is independently runnable

---
