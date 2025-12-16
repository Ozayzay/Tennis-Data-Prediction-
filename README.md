# Tennis Match Outcome Prediction (ATP) — End-to-End Data Science Pipeline

**Author:** Ozafa Yousuf Mahmood  
**Goal:** Predict the winner of professional tennis matches using **only pre-match information**, with a strong focus on **data leakage prevention**, **time-aware validation**, and **feature engineering** that reflects how tennis performance evolves over time.

This repository is designed as a **portfolio-grade** project for **Data Analyst / Data Scientist** roles: it demonstrates practical work across data cleaning, missing-data strategy, time-series evaluation, feature engineering, model training/tuning, ensembling, and unsupervised analysis.

---

## Highlights (Why this project is worth reading)

- **Leakage-safe pipeline design:** I aggressively removed post-match variables and engineered every historical feature using a strict **GET → STORE → UPDATE** pattern to ensure *no match can “predict itself.”*
- **Time-aware evaluation:** Used **chronological splits** and moved tuning to **TimeSeriesSplit** after identifying why standard k-fold CV can leak future information in time-dependent sports data.
- **Domain-driven feature engineering:** Built career stats, surface-conditioned stats, head-to-head rivalry features, and “recent form” using efficient hash-table and deque-based methods.
- **Practical modeling iteration:** Baselines → engineered features → retraining strategy to reduce cold-start → **ATP-only training/testing** to reduce sparsity → hyperparameter tuning → ensembles.
- **Best model:** **Stacking ensemble with RF meta-model** achieved **66.1%** test accuracy.
- **Unsupervised + anomaly work:** PCA/KMeans clustering, silhouette analysis, cluster profiling, and Isolation Forest anomaly detection with careful interpretation (distinguishing “real anomalies” vs expected data artifacts).

---

## Results (Test Accuracy)

### Before Feature Engineering (baseline models)
These runs used cleaned pre-match features and chronological evaluation:

- **Decision Tree:** 62.0%  
- **Random Forest:** 61.5%  
- **Logistic Regression:** 62.6%  
- **XGBoost:** 62.2%

### After Feature Engineering + ATP-only Train/Test + RF GridSearchCV
Key improvement came from reducing sparsity by focusing on ATP matches and properly tuning RF:

- **Random Forest (GridSearchCV + time-aware):** **65.58%** *(+1.5% absolute)*  
- **Logistic Regression:** 65.13%  
- **XGBoost:** 63.56%  
- **Decision Tree:** 61.30%

### Ensembles
- **Voting Ensemble (DT + RF + LR + XGB):** ~65.0%  
- **Stacking Ensemble (meta = Random Forest):** **66.1%** *(best overall)*

---

## Problem Statement

Given a match between **Player 1** and **Player 2**, predict whether Player 1 wins using only information that is available **before the match starts**.

This is harder than it looks:
- Tennis performance is **time-dependent** (rankings, form, surfaces).
- Many high-signal variables are only known **after** the match (scores, minutes, break points, etc.) and must be removed to avoid leakage.
- A large portion of tennis data is sparse (qualifiers, newcomers, surface history).

---

## Data Overview

The raw dataset includes:
- A large **Matches** table (hundreds of thousands of rows, many columns),
- Supporting tables such as **Players** and **Rankings**.

I treated this as a realistic prediction setting: the model must work even when a player has limited history, and must generalize across seasons.

---

## Methodology Overview

### 1) Data Cleaning & Preprocessing (Leakage-first mindset)
Key steps:
- **Dropped post-match / non-deployable columns** (anything unavailable pre-match).
- **Missing data strategy based on meaning:**
  - If missingness is tiny and essentially random (e.g., very small age missingness), I dropped those rows.
  - If missingness is meaningful (e.g., missing rank because a player is unranked), I **imputed with a sentinel** (e.g., max-rank + 1 style value) **and added a missingness flag** so the model can learn “unranked-ness” explicitly.
- Converted the dataset from winner/loser format into a consistent **Player1 vs Player2** structure.
  - This makes the pipeline model-friendly and enables symmetric features like *rank_diff, age_diff,* etc.

### 2) Baseline Modeling (Pre-feature-engineering)
I trained a baseline suite to test whether “better models” alone would solve the problem:
- Decision Tree
- Random Forest (with diagnostic tuning: max_features, depth, number of trees)
- Logistic Regression (with scaling pipeline)
- XGBoost

Outcome: performance was clustered around the low 60s. This validated the core hypothesis:
> The bottleneck is **feature signal**, not model choice.

---

## Feature Engineering (The core technical contribution)

### Design Principle: **Point-in-time correctness**
All historical features are computed as they would exist *before* each match.

To guarantee this, every feature update follows the same pattern:

1. **GET** historical stats for both players  
2. **STORE** them into the current match row  
3. **UPDATE** the stats using the current match result  

If you UPDATE first, you leak the result into the feature itself.

### Implementation Choices (Performance + clarity)
- Used **hash tables (Python dicts)** keyed by player IDs and matchups for O(1) lookup.
- Used **deques** for rolling “recent form” windows.
- Verified chronological ordering by tournament date and enforced monotonic time behavior.

### Engineered Feature Groups

#### A) “Difference” Features (quick, high-signal)
- rank_difference  
- age_difference  
- height_difference  

Why: many models benefit when the relative advantage is explicit rather than implied.

#### B) Career Performance Features (per player + diff)
For each player:
- career_matches, career_wins, career_win_rate  
Plus:
- career_win_rate_diff (P1 − P2)

Cold-start handling:
- If no history: default win_rate = **0.5** (neutral prior)
- Also store match counts so the model understands confidence level.

#### C) Surface-Specific Performance (Hard / Clay / Grass + diff)
For each surface, per player:
- matches, wins, win_rate  
Plus diffs for each surface.

Important discovery: **grass is extremely sparse** in historical context.
- ~95% of players have **0 grass matches** in the tracked history window.
This explains why surface features can be valuable but often default-heavy.

#### D) Head-to-Head (H2H) Rivalry Features
Tracked rivalry stats using a canonical matchup key:
- `key = tuple(sorted([player1_id, player2_id]))`

Features:
- h2h_matches
- h2h_wins for each player
- h2h win rates
- h2h_win_rate_diff

Scale note: tens of thousands of unique rivalries, and many matches are first-time meetings (H2H = 0).

Validation example:
- Verified known matchups (e.g., Federer vs Nadal) to ensure rivalry counting logic is correct.

#### E) Recent Form (Last 20 Matches)
For each player, tracked last N results using a deque (N=20):
- recent_matches (0–20)
- recent_form (win rate in last 20)
- recent_form_diff

Why “Last N Matches” (not “Last N Days”):
- Match frequency varies wildly by player, tournament type, and season.
- Matches provide a consistent sample size and reduce calendar-based bias.

---

## Retraining Strategy (Solving sparsity + cold-start)

After engineering, initial retraining showed only modest gains and sometimes more overfitting. I treated this as a signal-quality problem, not a failure.

Key refinements:

### 1) Reduce cold-start impact via time windowing
- Built history from earlier years (to accumulate stats),
- Trained and tested on later years where features are populated.

### 2) Train/test on ATP matches only (while still using broader history to build stats)
Qualification/lower-tier matches increase sparsity and distort the prediction target.
Restricting training/testing to ATP matches made engineered features more usable and improved results meaningfully.

---

## Hyperparameter Tuning (GridSearchCV + Time-Aware CV)

### Why time-aware tuning matters
Standard k-fold CV can produce overly optimistic or inconsistent estimates in time-dependent data because the model may train on “future” matches relative to a validation fold.

During tuning, I found that:
- Dataset ordering and split logic can materially affect outcomes.
- Time-series validation better matches the real task: **predict the future from the past.**

### What I did
- Used **GridSearchCV** (and moved toward **RandomizedSearchCV** when grid size became expensive).
- Switched evaluation to **TimeSeriesSplit** to preserve temporal ordering during tuning.
- Tuned Random Forest parameters (depth, trees, max_features, etc.) and selected configs based on time-aware validation behavior.

---

## Ensembling (Voting vs Stacking)

### Voting Ensemble
Combined DT + RF + LR + XGB to reduce variance.
Result: ~65% (comparable to the tuned RF).

### Stacking Ensemble (Best Model)
I implemented a **StackingClassifier** with:
- Base learners: DT, RF, LR (scaled pipeline), XGB
- Meta learner: **Random Forest**
- `passthrough=True` so the meta-model sees both base predictions and original features.

Why stacking wins:
- Voting combines models with fixed logic.
- Stacking **learns** how to weight each model differently across feature space.

**Best test result:** **66.1%**

---

## Unsupervised Analysis (Clustering)

### PCA + KMeans
I explored whether match contexts naturally cluster without using the label.

Key outcomes:
- Silhouette scores were generally low (weak global separation).
- Some K=2 solutions were imbalanced, suggesting KMeans was capturing edge cases rather than strong natural groupings.
- Cluster profiling (feature means per cluster) revealed meaningful patterns:
  - Clusters often reflected **experience gaps** (career matches) and **surface**.

### “Mid-project fix”: Carpet surface encoding
I caught a surface-encoding issue and added the missing category.
Observation:
- Carpet matches are extremely rare (~0.4%), so adding it can destabilize K=4 clustering.
This was a good example of using unsupervised diagnostics to find feature representation issues.

---

## Anomaly Detection (Isolation Forest)

Goal: detect unusual matches and verify whether they are real anomalies or explainable artifacts.

What I found:
- Many “rank anomalies” were not real anomalies—they were expected outcomes of **unranked-player rank imputation** (sentinel rank values).
- Some flagged matches were explained by entry mechanics:
  - Wildcards (WC), Qualifiers (Q), and Direct Acceptance behavior in weaker draws can create extreme rank differences.
- I **kept** these rows because they reflect real tournament structure.
- I did identify a small number of genuine physical outliers (e.g., extreme height values) and removed them.

This section demonstrates judgment: anomaly detection is only useful if you interpret it correctly.

---

## Repository Structure (Typical)
- `EDA_and_preprocessing.ipynb` — cleaning, exploratory analysis, leakage-safe preprocessing
- `feature_engineering.ipynb` — engineered features (career/surface/H2H/recent form)
- `ozafa_modeling_and_evaluation.ipynb` — baselines, retraining, evaluation
- `ozafa_clustering_analysis_based_on_matches.ipynb` — PCA/KMeans, clustering diagnostics, cluster profiling
- (Optional) `df_tail.csv` — sample output snapshot / debug artifact

---

## How to Run (Setup & Reproducibility)

1) **Clone**
```bash
git clone https://github.com/Ozayzay/Tennis-Data-Prediction-
cd Tennis-Data-Prediction-
```

2) Create environment
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\\Scripts\\activate   # Windows
```

3) Install dependencies

If you have a requirements.txt:
```bash
pip install -r requirements.txt
```

If not, install core packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost jupyter
```

4) Run notebooks
```bash
jupyter notebook
```

