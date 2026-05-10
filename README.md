# 🔍 System Failure Pattern Analysis
### Phase 4 & 5 — Rule Evaluation & Visualization

> **Association Rule Mining applied to system log data to discover failure patterns,
> measure their predictive strength, and communicate findings through publication-quality charts.**

---

## 📌 Project Overview

Modern distributed systems (e.g. OpenStack, cloud infrastructure) generate thousands of log events per second. When failures occur, they rarely happen in isolation — a cascade of related errors typically precedes a critical outage.

This project applies **Association Rule Mining** (ARM) to system logs to automatically discover these cascades. It answers the core question:

> *"When events A and B appear together in a log window, does failure C reliably follow?"*

The pipeline treats each **time-windowed log session** as a *transaction* and each **error/event** as an *item* — exactly like a market-basket analysis, but for system failures instead of shopping carts.

---

## 🧠 Core Concepts

### What is Association Rule Mining?

| Concept | Plain English | Example |
|---|---|---|
| **Transaction** | One log time-window (e.g. 60-second slice) | All events between 14:00:00 – 14:01:00 |
| **Item** | A single error/event in that window | `nova.compute.manager::ERROR` |
| **Itemset** | A combination of events that co-occur | `{disk_io::CRITICAL, memory_leak::ERROR}` |
| **Rule** | IF [itemset] → THEN [another itemset] | `{disk_io} → {nova.compute crash}` |

### Evaluation Metrics

#### 📊 Support
> *"How common is this combination across all sessions?"*

$$\text{Support}(A \Rightarrow B) = \frac{\text{sessions containing both A and B}}{\text{total sessions}}$$

- `0.05` = this pattern appears in 5% of all log windows
- Higher support = more frequent failure pattern

#### 🎯 Confidence
> *"Given that A occurred, how often does B follow?"*

$$\text{Confidence}(A \Rightarrow B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}$$

| Value | Interpretation |
|---|---|
| ≥ 0.90 | Near-certain — treat as a hard alarm |
| ≥ 0.75 | Strong — highly actionable rule |
| ≥ 0.60 | Moderate — worth monitoring |
| ≥ 0.50 | Marginal — minimum useful threshold |
| < 0.50 | Weak — discard |

#### 🚀 Lift
> *"Is this pattern real, or just random coincidence?"*

$$\text{Lift}(A \Rightarrow B) = \frac{\text{Confidence}(A \Rightarrow B)}{\text{Support}(B)}$$

| Value | Interpretation |
|---|---|
| > 3.0 | Very strong real pattern — highly non-random |
| > 2.0 | Strong positive association |
| > 1.2 | Acceptable — above the random baseline |
| ≈ 1.0 | Random co-occurrence — rule is useless |
| < 1.0 | Negative association — A suppresses B |

#### ⚖️ Leverage
> *"How many extra co-occurrences beyond statistical independence?"*

$$\text{Leverage} = \text{Support}(A \cup B) - \text{Support}(A) \times \text{Support}(B)$$

- Positive leverage confirms the rule is non-trivial
- Large positive value = strong non-random association

#### 🔒 Conviction
> *"How often would the rule be wrong if A and B were independent?"*

$$\text{Conviction} = \frac{1 - \text{Support}(B)}{1 - \text{Confidence}(A \Rightarrow B)}$$

- Higher = more reliable prediction
- ∞ means the rule has 100% confidence

---

## 📁 Project Structure

```
System-Failure-Pattern-Analysis/
│
├── data/
│   ├── raw_logs.csv              ← raw OpenStack log file
│   └── transactions.csv          ← preprocessed log sessions
│
├── src/
│   ├── 00_mock_data.py           ← [TEST] generates mock CSVs for testing
│   ├── 01_generate_logs.py       ← Phase 1: load / generate dataset
│   ├── 02_preprocess.py          ← Phase 2: convert logs → transactions
│   ├── 03_apriori.py             ← Phase 3a: Apriori algorithm
│   ├── 04_fpgrowth.py            ← Phase 3b: FP-Growth algorithm
│   ├── 05_evaluate.py            ← Phase 4: confidence & lift analysis  ★
│   └── 06_visualize.py           ← Phase 5: charts & output             ★
│
├── output/
│   ├── association_rules.csv     ← input to Phase 4/5
│   ├── frequent_itemsets.csv     ← input to Phase 5
│   ├── evaluation_report.csv     ← full ranked & annotated rule table
│   ├── top_rules.csv             ← top-N rules by composite score
│   ├── anomaly_rules.csv         ← high-lift anomaly rules
│   └── charts/
│       ├── 00_dashboard.png      ← 4-panel summary dashboard
│       ├── 01_support_confidence_scatter.png
│       ├── 02_lift_distribution.png
│       ├── 03_top_rules_heatmap.png
│       ├── 04_quality_pie.png
│       ├── 05_itemset_frequency_bar.png
│       ├── 06_confidence_lift_violin.png
│       ├── 07_rule_network.png
│       └── 08_metric_correlation.png
│
├── config.py                     ← all thresholds and paths
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration (`config.py`)

All tunable parameters live in `config.py`. The most important ones for Phase 4/5:

| Parameter | Default | Description |
|---|---|---|
| `MIN_SUPPORT` | `0.02` | Minimum support to keep an itemset |
| `MIN_CONFIDENCE` | `0.50` | Minimum confidence to keep a rule |
| `MIN_LIFT` | `1.20` | Minimum lift to keep a rule |
| `TOP_N_PATTERNS` | `20` | Number of top rules to report/visualise |
| `ANOMALY_THRESHOLD` | `2.50` | Lift value above which a rule is flagged as anomalous |

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
# Always set UTF-8 on Windows to avoid encoding errors
$env:PYTHONIOENCODING="utf-8"      # PowerShell
# or: set PYTHONIOENCODING=utf-8   # CMD

python src/01_generate_logs.py
python src/02_preprocess.py
python src/03_apriori.py
python src/04_fpgrowth.py
python src/05_evaluate.py
python src/06_visualize.py
```

### 3. Quick test (no full pipeline needed)

If you only want to test Phase 4 and 5, generate mock data first:

```bash
python src/00_mock_data.py   # creates realistic fake CSVs
python src/05_evaluate.py
python src/06_visualize.py
```

---

## 📄 Phase 4 — Evaluation (`05_evaluate.py`)

### What it does

1. **Loads** `output/association_rules.csv` (output of Apriori / FP-Growth)
2. **Computes extra metrics** not always provided by `mlxtend`:
   - Leverage, Conviction, Zhang's Metric
3. **Tags every rule** with a quality tier: `⭐ Excellent`, `✔ Good`, `~ Acceptable`, `✗ Weak`
4. **Filters** rules below `MIN_CONFIDENCE` and `MIN_LIFT` thresholds
5. **Ranks** by composite score: `lift × confidence`
6. **Prints a detailed plain-English interpretation** for each top rule
7. **Flags anomaly rules** whose lift exceeds `ANOMALY_THRESHOLD`
8. **Saves** three output CSVs

### Sample terminal output

```
╔══════════════════════════════════════════════════════════════════════╗
║           PHASE 4 — ASSOCIATION RULE EVALUATION                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  LIFT > 1.0  → events co-occur MORE than expected by chance          ║
║  LIFT = 1.0  → purely random, no real association                    ║
╚══════════════════════════════════════════════════════════════════════╝

  ┌──────────────────────────────────────────────────────────────────────┐
  │  Rule #1    ⭐ Excellent                                              │
  │  IF   [ keystonemiddleware.auth_token::ERROR ]                       │
  │  THEN [ nova.scheduler.manager::CRITICAL ]                           │
  ├──────────────────────────────────────────────────────────────────────┤
  │  Support    : 0.1167  (11.7% of all log sessions)                   │
  │  Confidence : 0.7900                                                 │
  │    79% confidence → Strong prediction. Reliable in 3 of 4 sessions. │
  │  Lift       : 9.5526                                                 │
  │    Lift = 9.55 → Extremely strong — co-occurs 9.6× more than chance │
  │    Leverage = 0.0778 → Strongly non-trivial.                        │
  └──────────────────────────────────────────────────────────────────────┘
```

### Output files

| File | Description |
|---|---|
| `evaluation_report.csv` | All rules with all metrics + quality label |
| `top_rules.csv` | Top-N rules ranked by score |
| `anomaly_rules.csv` | Rules with lift ≥ `ANOMALY_THRESHOLD` |

---

## 📊 Phase 5 — Visualization (`06_visualize.py`)

### What it does

Reads the evaluation outputs and generates **9 publication-quality charts** with a consistent dark theme.

### Chart gallery

| Chart | File | What it shows |
|---|---|---|
| **Dashboard** | `00_dashboard.png` | 4-panel summary of all key results |
| **Scatter** | `01_support_confidence_scatter.png` | Support vs Confidence, bubble size/colour = lift |
| **Lift Histogram** | `02_lift_distribution.png` | Distribution of lift values with KDE and anomaly line |
| **Heatmap** | `03_top_rules_heatmap.png` | Metric intensity across top-N rules |
| **Quality Pie** | `04_quality_pie.png` | Donut chart of rule quality tiers |
| **Itemset Frequency** | `05_itemset_frequency_bar.png` | Most frequent itemsets by support |
| **Violin Plot** | `06_confidence_lift_violin.png` | Confidence & lift distribution per quality tier |
| **Rule Network** | `07_rule_network.png` | Directed graph of antecedent → consequent relationships |
| **Correlation** | `08_metric_correlation.png` | Pearson correlation matrix of all numeric metrics |

### Optional dependencies

| Library | Used for | Install |
|---|---|---|
| `seaborn` | Violin plots, KDE overlays | `pip install seaborn` |
| `networkx` | Rule network graph | `pip install networkx` |

> Both are optional — all other charts are generated even without them.

---

## 📦 Requirements

```
pandas
numpy
matplotlib
mlxtend          # association rule mining (Apriori, FP-Growth)
seaborn          # optional — enhanced visualizations
networkx         # optional — rule network graph
```

Install everything:

```bash
pip install -r requirements.txt
```

---

## 🔬 Methodology Summary

```
Raw Logs
   │
   ▼
Time-Window Segmentation (60s windows)
   │
   ▼
Transaction Matrix  [ session × event ] binary matrix
   │
   ├──► Apriori Algorithm  ──┐
   │                         ├──► Frequent Itemsets ──► Association Rules
   └──► FP-Growth Algorithm ─┘
                                       │
                                       ▼
                              Phase 4: Evaluate
                         (confidence, lift, leverage,
                          conviction, Zhang's metric)
                                       │
                                       ▼
                              Phase 5: Visualize
                         (9 charts + 3 output CSVs)
```

---

## 📚 References

- Agrawal, R., & Srikant, R. (1994). *Fast Algorithms for Mining Association Rules.* VLDB.
- Zaki, M. J. (2000). *Scalable Algorithms for Association Mining.* IEEE TKDE.
- Raschka, S. (2018). *MLxtend: Providing machine learning and data science utilities.* JOSS.
- He, S., et al. (2020). *Loghub: A Large Collection of System Log Datasets.* IEEE ISSRE.

---

<div align="center">

**System Failure Pattern Analysis**  
*Data Mining Project — Association Rule Mining on System Logs*

</div>
