# Tennis Match Outcome Prediction (ATP) — End-to-End Data Science Pipeline

**Author:** Ozafa Yousuf Mahmood  
**Goal:** Predict the winner of professional tennis matches using **only pre-match information**, with a strong focus on **data leakage prevention**, **time-aware validation**, and **feature engineering** that reflects how tennis performance evolves over time.

This repository is designed as a portfolio-grade project for Data Analyst / Data Scientist roles: it demonstrates practical work across data cleaning, missing-data strategy, time-series evaluation, feature engineering, model training/tuning, ensembling, and unsupervised analysis.

## Highlights (Why this project is worth reading)

- Leakage-safe pipeline: removed post-match variables; engineered historical features with a strict GET → STORE → UPDATE pattern to ensure point-in-time correctness.
- Time-aware evaluation: chronological splits; tuning with TimeSeriesSplit to avoid temporal leakage.
- Domain-driven features: career stats, surface-conditioned stats, head-to-head rivalry features, recent form via deque.
- Practical iteration: baselines → engineered features → ATP-only training/testing to reduce sparsity → hyperparameter tuning → ensembles.
- Best model: Stacking ensemble with Random Forest meta-model achieved 66.1% test accuracy.
- Unsupervised + anomaly work: PCA/KMeans clustering and Isolation Forest with careful interpretation.

## Results (Test Accuracy)

### Before Feature Engineering

- Decision Tree: 62.0%
- Random Forest: 61.5%
- Logistic Regression: 62.6%
- XGBoost: 62.2%

### After Feature Engineering + ATP-only Train/Test + RF GridSearchCV

- Random Forest (time-aware tuning): 65.58%
- Logistic Regression: 65.13%
- XGBoost: 63.56%
- Decision Tree: 61.30%

### Ensembles

- Voting Ensemble (DT + RF + LR + XGB): ~65.0%
- Stacking Ensemble (meta = Random Forest): 66.1% (best)

### Quick Results Table

| Model                          | Test Accuracy |
| ------------------------------ | ------------- |
| Decision Tree                  | 61.30%        |
| Random Forest (tuned)          | 65.58%        |
| Logistic Regression (scaled)   | 65.13%        |
| XGBoost                        | 63.56%        |
| Voting Ensemble (DT+RF+LR+XGB) | ~65.0%        |
| Stacking (meta Random Forest)  | 66.10%        |

## Problem Statement

Given a match between Player 1 and Player 2, predict whether Player 1 wins using only pre-match information.

Challenges:

- Tennis performance is time-dependent (rankings, form, surfaces).
- Many high-signal variables are post-match and must be removed to avoid leakage.
- Data sparsity for qualifiers/newcomers and certain surfaces.

## Data Overview

Raw inputs include ATP match results, player info, and rankings over multiple seasons. The pipeline treats prediction realistically: models must work even with limited player history and across seasons.

## Methodology

### 1) Data Cleaning & Preprocessing

- Dropped post-match and non-deployable columns (scores, minutes, in-match stats, seeds, etc.).
- Missing data strategy:
  - Tiny, random missingness: drop rows.
  - Meaningful missingness: impute sentinel (e.g., max-rank + 1) and add a flag (e.g., unranked indicator).
- Converted winner/loser tables to Player1 vs Player2 format to enable symmetric features like rank_diff, age_diff.

### 2) Baseline Modeling

Trained DT, RF, LR (with scaling), XGB to validate whether models alone solve performance; results clustered in low 60s, confirming feature signal is the bottleneck.

### 3) Feature Engineering (Point-in-time correct)

Design principle: compute features as they exist before each match using GET → STORE → UPDATE order.

Implementation:

- Hash tables (dicts) for per-player lookups and canonical H2H keys.
- Deques for rolling recent form windows (last 20 matches).
- Chronological ordering via tournament date; validations for monotonic updates.

Engineered groups:

- Difference features: rank_difference, age_difference, height_difference.
- Career performance: career_matches, career_wins, career_win_rate per player; plus diffs; neutral prior 0.5 for no history.
- Surface-specific: per-surface matches, wins, win_rate (hard/clay/grass; carpet included as explicit category); plus diffs.
- Head-to-head: h2h_matches, per-player h2h_win_rate, h2h_win_rate_diff via canonical matchup keys.
- Recent form: recent_form over last 20 matches per player; recent_form_diff. Chose Last N Matches over Last N Days to avoid calendar bias and ensure consistent sample size.

### 4) Retraining Strategy

- Reduce cold-start via time windowing: accumulate history in earlier years; train/test in later years.
- Train/test on ATP matches only to reduce sparsity; still use broader history to build stats.

### 5) Hyperparameter Tuning

- GridSearchCV (and RandomizedSearchCV when needed) with TimeSeriesSplit for time-aware tuning.
- Tuned RF depth, trees, max_features; selected based on chronological validation.

### 6) Ensembling

- Voting: DT + RF + LR + XGB achieved ~65%.
- Stacking: base models as above; meta model = Random Forest; passthrough=True so meta sees base predictions and original features → best at 66.1%.

### 7) Unsupervised Analysis

- PCA + KMeans across k; silhouette analysis and cluster profiling.
- Clusters often reflect experience gaps and surface; global separation is weak (low silhouette), which is expected in mixed tennis contexts.
- Mid-project fix: added carpet surface encoding to ensure complete surface representation (rare but explicit category).

### 8) Anomaly Detection

- Isolation Forest on numeric pre-match features.
- Many “rank anomalies” were explained by unranked sentinel values and entry types (WC/Q/DA); retained as realistic tournament structure.
- Removed a small number of genuine physical outliers.

## Repository Structure

- notebooks/EDA_and_preprocessing.ipynb — cleaning, exploratory analysis, leakage-safe preprocessing
- notebooks/feature_engineering.ipynb — engineered features (career/surface/H2H/recent form)
- notebooks/ozafa_modeling_and_evaluation.ipynb — baselines, retraining, evaluation, ensembles
- notebooks/ozafa_clustering_analysis_based_on_matches.ipynb — PCA/KMeans, clustering diagnostics and profiling
- data/raw — source CSVs (ATP matches, players, rankings)
- data/processed — model-ready CSVs (e.g., matches_with_engineered_features.csv)
- results — feature importance, figures, and model artifacts

## How to Run (Setup & Reproducibility)

### Prerequisites

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Reproduce Modeling

1. Ensure processed dataset exists: data/processed/matches_with_engineered_features.csv
2. Open notebooks/ozafa_modeling_and_evaluation.ipynb and run sequentially.
3. Optional: adjust year splits in the notebook’s split section for different validation windows.

### Quick Start (CLI)

Use VS Code’s notebook runner or run cells sequentially. To reproduce results quickly, focus on `notebooks/ozafa_modeling_and_evaluation.ipynb` after ensuring `data/processed/matches_with_engineered_features.csv` exists.

1. Clone

```bash
git clone https://github.com/Ozayzay/Tennis-Data-Prediction-
cd Tennis-Data-Prediction-
```

## Results & Figures

- Feature importance CSV: results/baseline_dt_feature_importance.csv (rank_difference is dominant in baseline DT).
- Clustering and sparsity/completeness visualizations can be exported from the notebooks to results/figures.
  - If specific figure files are finalized, add their filenames here for showcase.

## Future Work

- ELO or Glicko-style dynamic ratings integrated into feature set.
  Source: Jeff Sackmann ATP dataset (matches, players, rankings).
- Player-specific fatigue/injury proxies; travel schedules.
- Tournament-level features (draw strength, altitude, indoor/outdoor) and richer contextual variables.
- Calibrated probabilities and betting-odds comparison.
- Dropped post-match and non-deployable columns (scores, minutes, in-match stats) to prevent leakage.
- Removed high-missingness or redundant fields (e.g., seeds) to reduce noise and improve robustness.

## License

This is a personal portfolio project. Please do not redistribute raw data files that may be subject to external licensing.

---

---

---

## 🚀 Project Structure

```

### Quick Reproduce (Best Model)

1) Open `notebooks/ozafa_modeling_and_evaluation.ipynb`
2) Run cells:
   - Load data (Step 1)
   - Filter ATP-only and split by `tourney_date` (Steps 1–3)
   - Train base models (Step 4)
   - Train Voting (Step 5) and Stacking (Step 6)
   - Evaluate (Step 7) → prints accuracy table; best stacking ≈ 66.1%

Outputs: accuracy summary, predictions for each split, and `results/baseline_dt_feature_importance.csv`.
project/
├── data/
│   ├── raw/                                    # Original CSV files from ATP database
│   └── processed/                              # Cleaned and transformed data
│       ├── matches_cleaned_columns_dropped.csv # After removing leakage features

Example figures to include for recruiters:
- Correlation heatmap (from `EDA_and_preprocessing.ipynb`)
- PCA + KMeans cluster plot (from `ozafa_clustering_analysis_based_on_matches.ipynb`)
- Top-15 feature importance bar chart (from `ozafa_modeling_and_evaluation.ipynb`)
│       ├── matches_final_with_player_context.csv   # Encoded with IDs/names
│       └── matches_final_without_player_context.csv # Model-ready dataset
├── notebooks/
│   ├── EDA_and_preprocessing.ipynb            # Data cleaning and preprocessing
│   ├── 02_feature_engineering.ipynb           # Feature creation and selection
│   └── 03_modeling_and_evaluation.ipynb       # Model training and evaluation
├── results/
<details>
<summary>Deep Dive: EDA, Cleaning, and Pipeline Details</summary>

This section collapses extended tables and explanations so the main README stays skimmable.

#### Summary of Preprocessing Pipeline
- Load 2014–2024 ATP matches (Main + Challenger/Qual), players, rankings
- Drop leakage (scores, minutes, in-match stats) and identifiers
- Clean missing values (rank sentinel + flag, country-median heights, direct entries)
- Reframe to Player1/Player2 with balanced target via random flip
- One-hot encode categorical features; control dimensionality
- Save model-ready dataset in `data/processed/`

#### Saved Datasets
- `matches_cleaned_columns_dropped.csv` — after dropping leakage columns
- `matches_final_with_player_context.csv` — encoded with player context
- `matches_final_without_player_context.csv` — model-ready dataset

#### Notes on Encoding Consistency
- For linear models: used `drop_first=True` to avoid multicollinearity
- For clustering diagnostics: kept full one-hot to interpret surfaces (including carpet)

</details>
│   └── models/                                # Trained model files
├── README.md                                  # This file (serves as final report)
├── requirements.txt                           # Python dependencies
└── .gitignore                                 # Git ignore rules
```

**Note:** We're using a notebook-focused approach with clear separation of concerns: preprocessing → feature engineering → modeling.

---

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Installation

1. **Clone the repository**

```bash
git clone [your-repo-url]
cd project
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download the data**
   - Place CSV files in `data/raw/`
   - Files can be obtained from [Jeff Sackmann's GitHub](https://github.com/JeffSackmann/tennis_atp)

---

## 📈 Methodology

### 1. Data Preprocessing & EDA

#### **A. Data Loading**

**Loaded Data Sources**:

- **ATP Players** (`atp_players.csv`): 65,989 players with biographical data
  - Key observation: 27% missing DOB, 90% missing height
  - Strategy: Will impute/handle missing values when merging with match data
- **ATP Rankings** (`atp_rankings_current.csv`): 92,341 ranking records

  - Date range: 2024-01-01 to 2024-12-30
  - Weekly snapshots of ATP rankings
  - No missing values

- **Match Data**: Combined Main Tour + Challenger/Qualification matches
  - Years: 2014-2024 (11 years)
  - Total matches loaded: 124,429

#### **B. Exploratory Data Analysis**

**Initial Data Quality Assessment**:

- Visualized ranking distributions (right-skewed, most players ranked 200-1000)
- Visualized ranking points distributions (exponential decay - top players have disproportionately more points)
- Analyzed missing value patterns across all features
- Key finding: Futures matches had ~40% missing match statistics → excluded from dataset

**Missing Value Analysis** (% of total data):

| Feature                             | Missing %  | Notes                                         |
| ----------------------------------- | ---------- | --------------------------------------------- |
| `w_ace`, `w_df`, etc. (match stats) | ~40%       | Futures matches lack statistics               |
| `winner_seed`, `loser_seed`         | 54%, 73%   | Only for seeded players                       |
| `winner_entry`, `loser_entry`       | 84%, 73%   | NaN means "Direct" entry                      |
| `winner_ht`, `loser_ht`             | 4.3%, 9.0% | Height data incomplete                        |
| `winner_rank`, `loser_rank`         | 0.8%, 3.3% | Unranked players (no points in last 52 weeks) |

---

#### **C. Feature Removal - Preventing Information Leakage**

**Dropped 25 columns** (reduced from 49 → 24 columns):

| Category             | Columns Dropped                                                    | Reason                                       |
| -------------------- | ------------------------------------------------------------------ | -------------------------------------------- |
| **Seed columns**     | `winner_seed`, `loser_seed`                                        | Redundant with ATP rank                      |
| **Identifiers**      | `tourney_id`, `match_num`, `tourney_name`                          | Unique row IDs; not generalizable            |
| **Match outcomes**   | `score`, `minutes`                                                 | Known only after match (information leakage) |
| **Match statistics** | `w_ace`, `w_df`, `w_svpt`, `w_1stIn`, `w_1stWon`, `l_*` (18 total) | In-match stats; not predictive features      |

**Remaining features** (24 columns):

- Player attributes: `id`, `name`, `hand`, `ht`, `age`, `ioc`, `rank`, `entry`
- Match context: `surface`, `tourney_level`, `round`, `best_of`, `draw_size`, `tourney_date`

---

#### **D. Data Cleaning - Handling Missing Values**

Applied domain-driven imputation strategies:

| Feature                                      | Missing %  | Strategy                                                               | Outcome                                                                    |
| -------------------------------------------- | ---------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **`surface`**                                | 0.04%      | Dropped rows                                                           | -53 rows (negligible)                                                      |
| **`winner_entry`/`loser_entry`**             | 84%/73%    | Imputed with `'Direct'`                                                | No flag needed (not missing at random - NaN explicitly means direct entry) |
| **`winner_ht`/`loser_ht`**                   | 4.3%/9.0%  | 1. Imputed with **country median height**<br>2. Dropped remaining NaNs | -165 rows after imputation                                                 |
| **`winner_age`, `loser_age`, `hand`, `ioc`** | <0.1% each | Dropped rows                                                           | -114 rows (minimal data loss)                                              |
| **`winner_rank`/`loser_rank`**               | 0.8%/3.1%  | Imputed with **max_rank + 1 (2258)**<br>Created `*_rank_imputed` flags | 975 winners, 3851 losers flagged                                           |
| **`*_rank_points`**                          | 0.8%/3.3%  | Dropped columns                                                        | Weak correlation (-0.4) with rank; rank is more informative                |

**Final Clean Dataset**: **124,097 matches** (99.7% retention rate)

**Rationale for Rank Imputation**:

- Players without rankings haven't won points in the past 52 weeks (per ATP rules)
- Imputing with `max_rank + 1` treats them as "worse than the worst-ranked player"
- Flag column alerts model: "this rank is uncertain"

---

#### **E. Data Reframing - Winner/Loser → Player1/Player2**

**Problem**: Original data format creates information leakage

- Each row: `winner_*` vs `loser_*` → model learns "person in winner column always wins"

**Solution**: Randomly assign players to `player1` or `player2`

**Implementation**:

1. Set random seed (`np.random.seed(89)`) for reproducibility
2. Create `flip` column: random binary (0 or 1) for each match
3. Use **vectorized approach** (`np.where`) for speed:
   - When `flip=1`: winner → player1, loser → player2
   - When `flip=0`: loser → player1, winner → player2
4. Target variable: `player1_won = flip`

**Result**:

- Perfectly balanced dataset: **50.16% class 1**, **49.84% class 0**
- No information leakage: model treats both players symmetrically
- Fast execution: vectorized operations (seconds vs minutes with `iterrows()`)

**New Column Structure**:

- `player1_*`: rank, rank_imputed, hand, height, age, ioc, entry
- `player2_*`: rank, rank_imputed, hand, height, age, ioc, entry
- `player1_won`: binary target (0 or 1)
- Match context: `surface`, `tourney_level`, `round`, `best_of`, `draw_size`, `tourney_date`

---

#### **F. Dimensionality Control - Dropping Country Codes**

**Dropped**: `player1_ioc`, `player2_ioc` (country codes)

**Rationale**:

- Each has ~150 unique values → would create **300+ sparse binary columns**
- **Curse of dimensionality**: Too many features relative to samples
- **Overfitting risk**: Model might memorize country patterns instead of learning tennis dynamics
- **Computational cost**: Slower training and inference

**Future consideration**: Can revisit in feature engineering as:

- Grouped by region (Europe, Americas, Asia, etc.)
- Top-N countries only
- "Home advantage" feature (player from tournament's host country)

---

#### **G. Categorical Encoding - One-Hot Encoding**

Applied `pd.get_dummies(drop_first=True)` to **7 categorical features**:

| Feature         | Unique Values | Description                                                                     |
| --------------- | ------------- | ------------------------------------------------------------------------------- |
| `player1_hand`  | 3             | R (Right), L (Left), U (Unknown)                                                |
| `player2_hand`  | 3             | R, L, U                                                                         |
| `player1_entry` | ~6            | Direct, Q (Qualifier), WC (Wild Card), LL (Lucky Loser), PR (Protected Ranking) |
| `player2_entry` | ~6            | Same as above                                                                   |
| `surface`       | 4             | Hard, Clay, Grass, Carpet                                                       |
| `tourney_level` | 5             | G (Grand Slam), M (Masters), A (ATP Tour), C (Challenger), D (Davis Cup)        |
| `round`         | ~8            | F (Final), SF (Semi-Final), QF, R16, R32, R64, R128, RR                         |

**Result**: **23 → 81 columns** (manageable expansion)

**Why `drop_first=True`?**

- Avoids **multicollinearity** (perfect correlation between features)
- Example: If surface has 3 values (Hard, Clay, Grass) and we create 3 binary columns:
  - If `surface_Clay=0` and `surface_Grass=0`, then it must be Hard
  - Third column is redundant (perfectly predictable from the other two)
- Dropping one category creates **reference category** (baseline for model interpretation)

---
