# ATP Tennis Match Outcome Prediction

**CMPT 459 Data Mining - Fall 2025**  
**Instructor:** Martin Ester | **TA:** Ethan Do

---

## 👥 Team Members

- [Your Name Here]
- [Teammate Name Here]

---

## 📋 Project Overview

This project analyzes ATP tennis match data to predict match outcomes using machine learning and data mining techniques. We explore relationships between player statistics, surface types, and match results to build an accurate prediction model.

### 🎯 Research Questions

1. **Match Outcome Prediction**: Given player and match features, can we accurately predict who will win in a head-to-head match?

2. **Feature Importance**: Which factors are the most predictive of match outcomes?

3. **Custom ELO System**: Does a custom Elo-based rating system provide more accurate predictions than the official ATP/WTA ranking points system?

---

## 📊 Dataset

**Source:** [Jeff Sackmann's ATP Tennis Database](https://github.com/JeffSackmann/tennis_atp/tree/master)

### Dataset Statistics

| Table        | Rows      | Columns | Description                            |
| ------------ | --------- | ------- | -------------------------------------- |
| **Matches**  | 124,097\* | 77      | Match results with detailed statistics |
| **Rankings** | 92,341    | 4       | Current ATP rankings (2024)            |
| **Players**  | 65,989    | 8       | Player biographical information        |

\*After preprocessing and cleaning (originally 124,429 matches from 2014-2024)

### Data Scope Decision

**Temporal Range**: 2014-2024 (11 years)

- **Training Data**: 2014-2022 (9 years)
- **Validation Data**: 2023 (1 year)
- **Test Data**: 2024 (1 year)

**Match Types Included**:

- ✅ Main Tour matches (~30,000 matches)
- ✅ Challenger/Qualification matches (~94,000 matches)
- ❌ Futures matches (excluded due to poor data quality - missing match statistics)
- ❌ Pre-2014 data (excluded to focus on modern tennis dynamics)

### Key Features

- **Match Context**: Surface (Clay/Hard/Grass/Carpet), draw size, tournament level, date, round, best of (3 or 5 sets)
- **Player Attributes**: Rankings, age, height, handedness, entry type (Direct/Qualifier/Wild Card)
- **Target Variable**: `player1_won` (binary: 0 or 1)

### Rationale for Dataset Selection

We selected this dataset because:

- **Rich Features**: Extensive contextual information including player rankings, surfaces, and tournament details
- **Large Sample Size**: 124,000+ matches provide robust training data
- **Real-World Application**: Tennis analytics is a growing field with practical applications in sports betting and coaching
- **Domain Relevance**: Clean binary outcomes, balanced data, and well-documented features make it ideal for supervised learning
- **Feature Engineering Opportunities**: Surface-specific performance, head-to-head records, and ELO ratings can be derived
- **Temporal Validation**: Chronological data allows for realistic time-based train/test splits

---

## 🚀 Project Structure

```
project/
├── data/
│   ├── raw/                                    # Original CSV files from ATP database
│   └── processed/                              # Cleaned and transformed data
│       ├── matches_cleaned_columns_dropped.csv # After removing leakage features
│       ├── matches_final_with_player_context.csv   # Encoded with IDs/names
│       └── matches_final_without_player_context.csv # Model-ready dataset
├── notebooks/
│   ├── EDA_and_preprocessing.ipynb            # Data cleaning and preprocessing
│   ├── 02_feature_engineering.ipynb           # Feature creation and selection
│   └── 03_modeling_and_evaluation.ipynb       # Model training and evaluation
├── results/
│   ├── figures/                               # Visualizations and plots
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

#### **H. Final Cleanup - Removing Player Identifiers**

**Dropped**: `player1_id`, `player2_id`, `player1_name`, `player2_name`

**Reason**: Prevent model from memorizing specific players (overfitting)

**Final Model-Ready Dataset**: **77 features** × **124,097 matches**

---

### **Summary of Preprocessing Pipeline**

| Stage              | Action                                                   | Result                          |
| ------------------ | -------------------------------------------------------- | ------------------------------- |
| **Load**           | Combined Main Tour + Challenger/Qual matches (2014-2024) | 124,429 matches, 49 columns     |
| **Drop leakage**   | Removed match outcomes, statistics, identifiers          | 124,429 matches, 24 columns     |
| **Clean**          | Imputed/dropped missing values                           | 124,097 matches, 24 columns     |
| **Reframe**        | Winner/loser → player1/player2 (balanced target)         | 124,097 matches, 25 columns     |
| **Drop countries** | Removed `player1_ioc`, `player2_ioc`                     | 124,097 matches, 23 columns     |
| **Encode**         | One-hot encoded 7 categorical features                   | 124,097 matches, 81 columns     |
| **Drop IDs**       | Removed player IDs and names                             | 124,097 matches, **77 columns** |

---

### **Saved Datasets**

| File                                       | Purpose                           | Columns |
| ------------------------------------------ | --------------------------------- | ------- |
| `matches_cleaned_columns_dropped.csv`      | After dropping leakage columns    | 24      |
| `matches_final_with_player_context.csv`    | Encoded but with player IDs/names | 81      |
| `matches_final_without_player_context.csv` | **Model-ready dataset**           | **77**  |

---

### 2. Feature Engineering

[To be completed in `02_feature_engineering.ipynb`]

**Planned Features**:

- Rank difference (`player1_rank - player2_rank`)
- Age difference (`player1_age - player2_age`)
- Height difference (`player1_ht - player2_ht`)
- Surface-specific win rates (historical performance per surface)
- Head-to-head records (historical matchup records)
- Recent form/momentum features
- Optional: Custom ELO rating system

**Planned Feature Selection Techniques**:

- Correlation analysis (heatmap to identify multicollinearity)
- Mutual Information scores
- Model-based feature importance (Random Forest)
- Recursive Feature Elimination (RFE)

---

### 3. Modeling & Evaluation

#### **Data Split Strategy**

- **Training**: 2014-2022 (104,874 matches, 84.5%)
- **Validation**: 2023 (10,663 matches, 8.6%)
- **Test**: 2024 (8,560 matches, 6.9%)
- **Note**: `tourney_date` dropped from features (used only for splitting)

---

#### **Baseline Model 1: Decision Tree (No Feature Engineering)** ✅

**Purpose**: Establish baseline performance before feature engineering

**Configuration**:

- Algorithm: DecisionTreeClassifier
- Parameters: `max_depth=10`, `random_state=42`
- Features: 75 (preprocessed features only)

**Results**:

| Metric        | Training | Validation | Test   |
| ------------- | -------- | ---------- | ------ |
| **Accuracy**  | 65.96%   | 62.68%     | 62.14% |
| **Precision** | 0.6518   | 0.6168     | 0.6136 |
| **Recall**    | 0.6555   | 0.6264     | 0.6208 |
| **F1-Score**  | 0.6536   | 0.6215     | 0.6172 |

**Key Findings**:

- Top 3 features: `player1_rank` (47.4%), `player2_rank` (36.1%), `player2_age` (4.2%)
- Overfitting gap: 3.28% (train-val) - acceptable
- Baseline accuracy: ~62% on unseen 2024 data

---

#### **Planned Models** (To be completed)

- Logistic Regression (linear baseline)
- Random Forest (ensemble method)
- Gradient Boosting (XGBoost/LightGBM)
- Support Vector Machine (SVM)
- k-Nearest Neighbors (k-NN)
- Models with engineered features (rank_diff, age_diff, etc.)

---

## 🔬 Key Results

[To be filled after modeling]

---

## 🧩 Challenges Encountered

### 1. Information Leakage Prevention

- **Challenge**: Original dataset had winner/loser format which would leak outcome information to the model
- **Solution**: Implemented random player position assignment (player1/player2) with reproducible seed, achieving perfect class balance (50.16% / 49.84%)

### 2. Missing Data Handling

- **Challenge**: Multiple features had missing values with different missingness patterns
- **Solution**: Applied domain-driven imputation:
  - `winner_entry`/`loser_entry`: NaN means "Direct" entry (domain knowledge)
  - Height: Imputed using country median
  - Rank: Imputed with max_rank + 1 for unranked players, added flag columns
  - Dropped rows with <0.1% missingness in critical features

### 3. Dimensionality vs. Information Trade-off

- **Challenge**: Country codes (IOC) would add 300+ sparse columns but contain potentially useful information
- **Solution**: Dropped for initial modeling to avoid curse of dimensionality; can revisit as grouped/aggregated features in feature engineering

### 4. Futures Match Data Quality

- **Challenge**: Futures matches (lower-tier tournaments) had ~40% missing match statistics
- **Solution**: Excluded Futures matches entirely; focused on Main Tour and Challenger/Qualification matches (2014-2024)

---

## 📚 Code Organization

### Notebook Structure

#### **1. `EDA_and_preprocessing.ipynb`** ✅ COMPLETED

**Sections**:

1. **Setup & Data Loading**

   - Import libraries (pandas, numpy, matplotlib, seaborn)
   - Load ATP matches, rankings, and player data
   - Initial data inspection and quality assessment

2. **Exploratory Data Analysis (EDA)**

   - Visualize ranking and points distributions
   - Analyze missing value patterns
   - Identify data quality issues (Futures matches)

3. **Feature Removal** (Information Leakage Prevention)

   - Drop seed columns (redundant with rank)
   - Drop identifiers (tourney_id, match_num, tourney_name)
   - Drop match outcomes (score, minutes)
   - Drop match statistics (w*\*, l*\* columns)

4. **Data Cleaning**

   - Impute missing `surface` (drop 53 rows)
   - Impute missing `winner_entry`/`loser_entry` with 'Direct'
   - Impute missing heights using country median
   - Drop rows with minimal missing values (<0.1%)
   - Impute missing ranks with max_rank + 1, create flag columns
   - Drop rank_points columns (weak correlation with rank)

5. **Data Reframing** (Winner/Loser → Player1/Player2)

   - Set random seed (89) for reproducibility
   - Create flip column for random player assignment
   - Vectorized implementation using `np.where()`
   - Create `player1_won` target variable
   - Result: Perfectly balanced classes (50.16% / 49.84%)

6. **Dimensionality Control**

   - Drop country codes (`player1_ioc`, `player2_ioc`)

7. **Categorical Encoding**

   - One-hot encode 7 categorical features
   - Use `drop_first=True` to avoid multicollinearity
   - Result: 23 → 81 columns

8. **Final Cleanup**
   - Drop player IDs and names
   - Save model-ready dataset (77 columns)

**Outputs**:

- `data/processed/matches_cleaned_columns_dropped.csv` (24 columns)
- `data/processed/matches_final_with_player_context.csv` (81 columns)
- `data/processed/matches_final_without_player_context.csv` (77 columns) ← **Model-ready**

---

#### **2. `02_feature_engineering.ipynb`** (Planned)

**Planned Sections**:

1. Load preprocessed data
2. Create derived features (rank/age/height differences)
3. Engineer historical features (surface win rates, head-to-head)
4. Feature selection (correlation, mutual information, RFE)
5. Save feature-engineered dataset

---

#### **3. `modeling_and_evaluation.ipynb`** ✅ BASELINE COMPLETED

**Completed Sections**:

1. Import libraries and load preprocessed data (77 features, 124,097 matches)
2. Train/validation/test split by date (2014-2022 / 2023 / 2024)
3. Baseline Decision Tree model training
4. Performance evaluation (accuracy, precision, recall, F1-score)
5. Confusion matrix and classification report
6. Feature importance analysis
7. Results summary

**Outputs**:

- `results/baseline_dt_feature_importance.csv` (feature rankings)
- Confusion matrix visualization
- Performance metrics across train/val/test sets

**Planned Additions**:

- Additional models (Random Forest, XGBoost, Logistic Regression)
- Models with feature-engineered datasets
- Hyperparameter tuning
- Model comparison and final selection

---

## 📊 Running the Analysis

### Start Jupyter Notebook

```bash
# Navigate to project directory
cd "/Users/oz/Desktop/459 Ester/project"

# Activate virtual environment (if using one)
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# Start Jupyter
jupyter notebook
```

Then open `notebooks/EDA_and_preprocessing.ipynb` and run cells sequentially.

### Saving Results

- **Figures**: Save plots to `results/figures/` using `plt.savefig()`
- **Processed Data**: Save cleaned data to `data/processed/` using `df.to_csv()`
- **Models**: Save trained models to `results/models/` using `joblib.dump()` or `pickle`

---

## 🎓 References

1. Sackmann, J. (2024). _ATP Tennis Database_. GitHub. https://github.com/JeffSackmann/tennis_atp
2. [Add any papers or resources you referenced]

---

## 📝 Project Deliverables Checklist

### ✅ Phase 1: Project Proposal (Due Sept 30, 2025)

- [x] Dataset selected and linked
- [x] Dataset statistics documented
- [x] Rationale provided
- [x] Research questions defined

### ✅ Phase 2: Data Preprocessing & EDA

- [x] Basic EDA completed (distributions, correlations)
- [x] Missing value handling (domain-driven imputation strategies)
- [x] Feature encoding (one-hot encoding with drop_first=True)
- [x] Data reframing (winner/loser → player1/player2)
- [x] Information leakage prevention (dropped match outcomes and statistics)
- [x] Class balancing (50.16% / 49.84% split via random assignment)
- [ ] Feature engineering (in progress)
- [ ] Dimensionality reduction (planned for modeling phase)

### ✅ Phase 3: Final Project Delivery

- [x] Baseline model trained (Decision Tree, 62.14% test accuracy)
- [ ] Feature engineering (rank_diff, age_diff, surface win rates)
- [ ] Multiple classification algorithms (3+ models)
- [ ] Hyperparameter tuning
- [ ] Model comparison and evaluation
- [ ] Clustering analysis (3+ algorithms)
- [ ] Outlier detection (3+ methods)
- [ ] Feature selection (mutual information, RFE)
- [ ] Final report (max 2 pages)
- [ ] Clean, well-documented code

---

## 📞 Contact

For questions or collaborations:

- [Your Email]
- [Teammate Email]

---

## 📜 License

This project is for academic purposes as part of CMPT 459 at Simon Fraser University.
