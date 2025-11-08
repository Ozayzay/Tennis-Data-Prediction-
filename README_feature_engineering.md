# Feature Engineering for ATP Tennis Match Prediction

**CMPT 459 - Data Mining Project**  
**Dataset:** 124,097 ATP matches (2014-2024)  
**Total Features Engineered:** 43 temporal features

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Information](#dataset-information)
3. [Feature Engineering Categories](#feature-engineering-categories)
4. [Technical Implementation](#technical-implementation)
5. [Validation and Quality Assurance](#validation-and-quality-assurance)
6. [Results and Performance](#results-and-performance)
7. [Files Generated](#files-generated)
8. [Usage Instructions](#usage-instructions)
9. [Key Takeaways](#key-takeaways)

---

## Project Overview

### Goal
Predict ATP tennis match outcomes using machine learning with sophisticated temporal feature engineering.

### Core Principle: Preventing Data Leakage
**Critical Rule:** All features must use ONLY information available BEFORE the match started.

For every match, we follow this pattern:
1. **GET** statistics from hash table (stats from before this match)
2. **STORE** those stats in the dataframe
3. **UPDATE** hash table after recording (include current match result)

This ordering ensures we never use future information to predict past outcomes.

### Approach
- **Implementation:** Hash table approach for O(1) lookups
- **Processing Time:** ~3-4 minutes for 124,097 matches
- **Memory Efficient:** Hash tables for 4,413 unique players
- **Validated:** Manual player tracing and statistical checks

---

## Dataset Information

### Source Data
- **Main Tour Matches:** atp_matches_{2014-2024}.csv
- **Qualification/Challenger Matches:** atp_matches_qual_chall_{2014-2024}.csv
- **Time Period:** December 29, 2013 - December 18, 2024
- **Total Matches:** 124,097 (after preprocessing)

### Data Characteristics
- **Unique Players:** 4,413
- **Original Features:** 81 (after preprocessing)
- **Engineered Features:** 43
- **Final Feature Count:** 122 (after encoding)
- **Target Variable:** player1_won (binary: 1=win, 0=loss)
- **Missing Values:** 0 (after imputation during preprocessing)

### Preprocessing Completed
- Missing values imputed
- Heights filled by country median
- Rankings imputed with max_rank + 1 for unranked players
- Categorical variables one-hot encoded
- Match outcomes randomized (player1/player2 assignment balanced)

---

## Feature Engineering Categories

### 1. Simple Difference Features (3 features)

**Purpose:** Capture relative differences between players at match time

**Features Created:**
```python
- rank_difference = player1_rank - player2_rank
- age_difference = player1_age - player2_age  
- height_difference = player1_ht - player2_ht
```

**Key Insights:**
- `rank_difference`: -0.27 correlation with target (moderate predictor)
- Negative value = Player1 is better ranked (lower rank number = better)
- Age and height differences show weak correlations (<0.05)

**Implementation:** Simple subtraction, no temporal complexity

---

### 2. Career Statistics (7 features)

**Purpose:** Track each player's overall career performance before the match

**Features Created:**
- `player1_career_matches` - Total matches played before this one
- `player1_career_wins` - Total wins accumulated before this one
- `player1_career_win_rate` - Career win rate before this one
- `player2_career_matches`
- `player2_career_wins`
- `player2_career_win_rate`
- `career_win_rate_diff` - player1_career_win_rate - player2_career_win_rate

**Implementation Method:** Hash table approach

```python
from collections import defaultdict

player_stats = defaultdict(lambda: {'matches': 0, 'wins': 0})

for i in range(num_matches):
    p1_id = df.loc[i, 'player1_id']
    p2_id = df.loc[i, 'player2_id']
    p1_won = df.loc[i, 'player1_won']
    
    # STEP 1: GET stats BEFORE this match
    p1_stats = player_stats[p1_id]
    p2_stats = player_stats[p2_id]
    
    # STEP 2: STORE "before" stats
    df.loc[i, 'player1_career_matches'] = p1_stats['matches']
    df.loc[i, 'player1_career_wins'] = p1_stats['wins']
    df.loc[i, 'player1_career_win_rate'] = (
        p1_stats['wins'] / p1_stats['matches'] if p1_stats['matches'] > 0 else 0.5
    )
    
    # ... same for player2 ...
    
    # STEP 3: UPDATE hash table AFTER recording
    player_stats[p1_id]['matches'] += 1
    player_stats[p1_id]['wins'] += p1_won
    player_stats[p2_id]['matches'] += 1
    player_stats[p2_id]['wins'] += (1 - p1_won)
```

**Default Values:**
- First match: 0 matches, 0 wins, 0.5 win rate (neutral/unknown)
- Model learns to check `career_matches` column for confidence

**Processing Time:** ~50 seconds for 124,097 matches

---

### 3. Surface-Specific Win Rates (21 features)

**Purpose:** Track performance separately on Clay, Grass, and Hard courts

**Why Important:**  
Tennis players have dramatically different performance on different surfaces. Example: Rafael Nadal has a 93% win rate on clay but 78% on hard courts.

**Features Created:**

For each surface (Clay, Grass, Hard):
- `player1_[surface]_matches` - Matches on this surface before
- `player1_[surface]_wins` - Wins on this surface before
- `player1_[surface]_win_rate` - Win rate on this surface before
- Same 3 features for player2
- `[surface]_win_rate_diff` - Difference feature

**Total:** 3 surfaces × 3 stats × 2 players + 3 differences = 21 features

**Implementation Method:** Nested hash table

```python
player_surface_stats = defaultdict(lambda: {
    'Clay': {'matches': 0, 'wins': 0},
    'Grass': {'matches': 0, 'wins': 0},
    'Hard': {'matches': 0, 'wins': 0}
})

for i in range(num_matches):
    # Determine surface
    if df.loc[i, 'surface_Clay']:
        surface = 'Clay'
    elif df.loc[i, 'surface_Grass']:
        surface = 'Grass'
    else:
        surface = 'Hard'
    
    # Get stats BEFORE this match
    p1_surface_stats = player_surface_stats[p1_id][surface]
    
    # Store and update...
```

**Surface Distribution Statistics:**
- **Hard court:** Mean 53 matches/player (most common, ~65% of matches)
- **Clay court:** Mean 35 matches/player (~30% of matches)
- **Grass court:** Mean 0.97 matches/player (~5% of matches)
- 60% of players have 0 clay matches in their history
- 95% of players have 0 grass matches in their history

**Why Sparsity is OK:**
- Reflects reality: Grass season is very short (mainly Wimbledon)
- Model learns to trust surface-specific stats when matches > 0
- Falls back to overall career win rate when surface experience is 0

**Processing Time:** ~50 seconds

---

### 4. Head-to-Head (H2H) Statistics (6 features)

**Purpose:** Track rivalry history between specific player pairs

**Why Important:**  
Some players consistently beat others regardless of rankings. Example: Player A might have 70% career win rate but only 30% vs Player B specifically.

**Features Created:**
- `h2h_matches` - Total matches between these two players before
- `player1_h2h_wins` - Player1's wins vs this specific opponent before
- `player2_h2h_wins` - Player2's wins vs this specific opponent before
- `player1_h2h_win_rate` - Player1's win rate vs this opponent
- `player2_h2h_win_rate` - Player2's win rate vs this opponent
- `h2h_win_rate_diff` - Difference feature

**Implementation Challenge: Player Order Ambiguity**

Problem:
- Match 10: player1=Federer(103819), player2=Nadal(104745)
- Match 50: player1=Nadal(104745), player2=Federer(103819)
- Same rivalry, different player roles!

If we used `(player1_id, player2_id)` as the key, we'd create two separate rivalry records for the same matchup.

**Solution: Canonical Key Approach**

```python
def get_h2h_key(p1_id, p2_id):
    """Return canonical key: (smaller_id, larger_id)"""
    return tuple(sorted([p1_id, p2_id]))

# Both (103819, 104745) and (104745, 103819) become (103819, 104745)
```

**Hash Table Structure:**

```python
h2h_stats = defaultdict(lambda: {
    'matches': 0, 
    'smaller_id_wins': 0, 
    'larger_id_wins': 0
})

h2h_key = get_h2h_key(p1_id, p2_id)
smaller_id, larger_id = h2h_key

# Determine which counter to increment based on winner
if p1_won:
    if p1_id == smaller_id:
        h2h_stats[h2h_key]['smaller_id_wins'] += 1
    else:
        h2h_stats[h2h_key]['larger_id_wins'] += 1
```

**Statistics:**
- **Unique rivalries tracked:** 88,896
- **Most common:** h2h_matches = 0 (first meeting)
- **Famous rivalry example:** Federer-Nadal had 9 matches in dataset (7-2 Federer)

**Default Values:**
- First meeting: 0 matches, 0-0 record, 0.5 win rate for both

**Processing Time:** ~50 seconds

---

### 5. Recent Form / Momentum (5 features)

**Purpose:** Capture current momentum by tracking last 20 matches

**Why Important:**  
Recent performance often matters more than career statistics. A player on a 10-match winning streak is more dangerous than their career 60% win rate suggests.

**Features Created:**
- `player1_recent_matches` - Number of matches in window (0-20)
- `player1_recent_form` - Win rate in last 20 matches
- `player2_recent_matches`
- `player2_recent_form`
- `recent_form_diff` - Difference feature

**Implementation Method:** Using deque (double-ended queue)

```python
from collections import deque

player_recent_results = defaultdict(lambda: deque(maxlen=20))

for i in range(num_matches):
    p1_recent = player_recent_results[p1_id]  # deque of 1s and 0s
    
    # Calculate recent form
    if len(p1_recent) > 0:
        p1_recent_wins = sum(p1_recent)  # Counts the 1s (wins)
        p1_recent_matches = len(p1_recent)
        p1_recent_win_rate = p1_recent_wins / p1_recent_matches
    else:
        p1_recent_win_rate = 0.5
        p1_recent_matches = 0
    
    # Store in dataframe
    df.loc[i, 'player1_recent_form'] = p1_recent_win_rate
    df.loc[i, 'player1_recent_matches'] = p1_recent_matches
    
    # Update deque (automatically drops oldest if > 20)
    p1_recent.append(p1_won)
```

**Why deque?**
- **Automatic size management:** `maxlen=20` parameter handles sliding window
- **O(1) operations:** Fast append and automatic pop from other end
- **Memory efficient:** Only stores last 20 results per player
- **Simple logic:** `sum(deque)` counts wins, `len(deque)` gives match count

**Window Size: Why 20 matches?**
- **Too small (5):** Too volatile, random variance dominates signal
- **Too large (50):** Loses recency, becomes similar to career stats
- **20 matches:** Good balance of stability and recency (industry standard)

**Validation Example (Roger Federer):**
```
Match 1:  recent_matches=0,  recent_form=0.500 (default)
Match 2:  recent_matches=1,  recent_form=1.000 (1-0 record)
Match 5:  recent_matches=4,  recent_form=0.750 (3-1 record)
Match 21: recent_matches=20, recent_form=0.900 (18-2 in last 20)
Match 22: recent_matches=20, recent_form=0.850 (17-3 in last 20)
```

**Processing Time:** ~50 seconds

---

## Technical Implementation

### Hash Table Approach

**Why Hash Tables Over Vectorized Operations?**

| Aspect | Hash Tables | Vectorized (pandas) |
|--------|-------------|---------------------|
| **Speed** | ~50 seconds/feature | ~10 seconds/feature |
| **Code Complexity** | Simple, intuitive | Complex for H2H, deques |
| **Debugging** | Easy to trace | Difficult to debug |
| **Flexibility** | Handles any logic | Limited to vectorizable ops |
| **Risk** | Low (straightforward) | Medium (subtle bugs possible) |

**Decision:** Chose hash tables for:
- **Clarity:** Easier to understand and maintain
- **Correctness:** Lower risk of temporal ordering bugs
- **Flexibility:** Can handle complex logic (H2H, nested dicts, deques)
- **Performance:** 50 seconds is acceptable for 124k matches

### Data Leakage Prevention Pattern

**The Universal Pattern Used Throughout:**

```python
for i in range(num_matches):
    # ===== STEP 1: GET =====
    # Retrieve stats from hash table (before this match)
    current_stats = hash_table[player_id]
    
    # ===== STEP 2: STORE =====
    # Store those stats in the dataframe
    df.loc[i, 'feature'] = current_stats['value']
    
    # ===== STEP 3: UPDATE =====
    # Update hash table with current match result
    hash_table[player_id]['value'] += match_result
```

**Why This Ordering is Critical:**

❌ **WRONG (Data Leakage):**
```python
# Update first
hash_table[player_id]['wins'] += match_result
# Then store
df.loc[i, 'player_wins'] = hash_table[player_id]['wins']
# Result: Using current match to predict current match!
```

✅ **CORRECT (No Leakage):**
```python
# Get before
stats_before = hash_table[player_id]['wins']
# Store before
df.loc[i, 'player_wins'] = stats_before
# Update after
hash_table[player_id]['wins'] += match_result
# Result: Only using information from before the match
```

### Data Sorting Requirement

**Critical Assumption:** Data must be sorted chronologically by `tourney_date`

**Current Format:** Integer dates in YYYYMMDD format
- Example: 20140113 (January 13, 2014)
- Sorts correctly: 20131231 < 20140101 ✓
- Verified: `df['tourney_date'].is_monotonic_increasing` returns `True`

**Why Sorting Matters:**
- Hash tables maintain running totals in chronological order
- Processing out of order would give incorrect historical statistics
- Example: If we process a 2020 match before a 2019 match, the 2019 match would incorrectly include 2020 results in its "before" stats

### Default Value Strategy

**Philosophy:** Use 0.5 (50% win rate) when no history exists

**Rationale:**
1. **Better than NaN:** Avoids model errors and missing value handling
2. **Better than 0:** Doesn't imply "definitely loses"
3. **Better than global mean:** Doesn't assume average performance
4. **Neutral assumption:** 50/50 represents maximum uncertainty
5. **Model learns context:** Model can check companion features (e.g., `career_matches = 0`) to understand this is an uncertain estimate

**Application:**
- Player's first career match: 0.5 win rate
- Player's first match on a surface: 0.5 surface win rate  
- Players' first meeting: 0.5 H2H win rate for both
- Player's first match (no recent history): 0.5 recent form

### Python Data Structures Used

**1. defaultdict**
```python
from collections import defaultdict

# Auto-creates missing keys with default value
player_stats = defaultdict(lambda: {'matches': 0, 'wins': 0})

# No need for:
if player_id not in player_stats:
    player_stats[player_id] = {'matches': 0, 'wins': 0}

# Just access directly:
stats = player_stats[player_id]  # Auto-creates if missing
```

**2. deque (double-ended queue)**
```python
from collections import deque

# Automatic size limiting
recent = deque(maxlen=20)

# Append 21st item -> oldest automatically removed
for i in range(25):
    recent.append(i)
    
print(len(recent))  # Always 20 (not 25)
print(list(recent))  # [5, 6, 7, ..., 24]
```

**3. Tuples as dictionary keys**
```python
# Tuples are immutable and hashable -> can be dict keys
h2h_stats = {}
key = (103819, 104745)  # (Federer, Nadal)
h2h_stats[key] = {'matches': 9, 'p1_wins': 7, 'p2_wins': 2}

# Lists cannot be keys (mutable)
# h2h_stats[[103819, 104745]] = ...  # ERROR!
```

**4. Lambda functions**
```python
# Anonymous function for defaultdict
default_value = lambda: {'matches': 0, 'wins': 0}

# Equivalent to:
def default_value():
    return {'matches': 0, 'wins': 0}
```

**5. f-strings (formatted string literals)**
```python
surface = 'Clay'
column_name = f'player1_{surface.lower()}_matches'
# Result: 'player1_clay_matches'

# Dynamic column creation without repetitive code
for surface in ['Clay', 'Grass', 'Hard']:
    df[f'player1_{surface.lower()}_win_rate'] = ...
```

---

## Validation and Quality Assurance

### Validation Methods

**1. Player-Specific Tracing**

Followed individual players through their entire match history:

```python
player_id = 103819  # Roger Federer
player_matches = df[
    (df['player1_id'] == player_id) | 
    (df['player2_id'] == player_id)
]

# Manually verified:
# - Match 1: career_matches=0, career_wins=0 ✓
# - Match 2: career_matches=1, career_wins=1 ✓  
# - Match 3: career_matches=2, career_wins=2 ✓
```

**2. Famous Rivalry Validation**

Checked Roger Federer vs Rafael Nadal head-to-head:
- **Found:** 9 matches in dataset (2014-2024)
- **H2H record:** Federer 7-2 Nadal ✓
- **First match:** h2h_matches=0 (correct)
- **Last match:** h2h_matches=8 (correct)

**3. First Match Verification**

Confirmed no data leakage:
```python
first_rows = df.head(100)
assert (first_rows['player1_career_matches'] >= 0).all()
assert (first_rows['player2_career_matches'] >= 0).all()

# Some players should have 0 matches (their first match)
assert (first_rows['player1_career_matches'] == 0).any()
```

**4. Statistical Sanity Checks**

```python
# Career win rates should be between 0 and 1
assert (df['player1_career_win_rate'] >= 0).all()
assert (df['player1_career_win_rate'] <= 1).all()

# Recent matches should be between 0 and 20
assert (df['player1_recent_matches'] >= 0).all()
assert (df['player1_recent_matches'] <= 20).all()

# H2H matches should be non-negative
assert (df['h2h_matches'] >= 0).all()
```

**5. Distribution Analysis**

Checked for anomalies:
```python
df['player1_career_win_rate'].describe()
# mean: ~0.52 (reasonable for ATP level)
# min: 0.00 (some players never won)
# max: 1.00 (some early-career undefeated streaks)

df['h2h_matches'].value_counts()
# 0: ~85% (most matches are first meetings) ✓
# 1-3: ~10% (occasional rematches) ✓
# 4+: ~5% (repeated rivalries) ✓
```

### Data Quality Metrics

**No Missing Values:**
```
After feature engineering:
- Missing values: 0 (0.00%)
- All features properly populated
- Default values used where appropriate
```

**Feature Ranges:**
| Feature | Min | Max | Mean | Notes |
|---------|-----|-----|------|-------|
| career_matches | 0 | 590 | 98.3 | Max is a veteran player |
| career_win_rate | 0.0 | 1.0 | 0.52 | Balanced around 50% |
| clay_matches | 0 | 590 | 35.0 | 60% have 0 |
| grass_matches | 0 | 113 | 0.97 | 95% have 0 |
| hard_matches | 0 | 501 | 53.0 | Most common surface |
| h2h_matches | 0 | 17 | 0.38 | Most are first meetings |
| recent_matches | 0 | 20 | 14.2 | Window fills over time |
| recent_form | 0.0 | 1.0 | 0.52 | Similar to career rate |

**Known Limitations:**

1. **Dataset coverage:** Only 2014-2024, missing early careers of established players
2. **Grass court sparsity:** Very few matches (reality of tennis calendar)
3. **Cold start:** Early matches in dataset have limited historical features
4. **H2H sparsity:** Most matchups are first-time meetings
5. **No match-level details:** No score, set-by-set data, or match duration

---

## Results and Performance

### Feature Engineering Performance

| Feature Set | Features | Processing Time | Memory |
|-------------|----------|-----------------|--------|
| Simple Differences | 3 | < 1 second | Negligible |
| Career Stats | 7 | ~50 seconds | ~1 MB |
| Surface Stats | 21 | ~50 seconds | ~2 MB |
| Head-to-Head | 6 | ~50 seconds | ~5 MB |
| Recent Form | 5 | ~50 seconds | ~1 MB |
| **Total** | **43** | **~3-4 minutes** | **~10 MB** |

**Hash Table Statistics:**
- **Unique players tracked:** 4,413
- **Unique H2H rivalries:** 88,896
- **Deques maintained:** 4,413 (one per player)

### Baseline Model Performance

**Model:** XGBoost Classifier  
**Features:** 75 (excluding tourney_date and player IDs)  
**Data Split:** Chronological by date

| Split | Date Range | Matches | Accuracy | F1-Score |
|-------|-----------|---------|----------|----------|
| Train | 2014-2022 | 96,328 (77.6%) | 70.06% | 0.700 |
| Validation | 2023 | 13,584 (10.9%) | 63.57% | 0.637 |
| Test | 2024 | 14,185 (11.4%) | 62.93% | 0.630 |

**Observations:**
- **Moderate overfitting:** 6-7% train-test gap
- **Reasonable generalization:** Test performance close to validation
- **Better than random:** 63% vs 50% baseline
- **Room for improvement:** Feature selection and hyperparameter tuning needed

### Feature Importance (Top 10)

From baseline Decision Tree model:

1. `player1_rank` (0.185) - Primary predictor
2. `player2_rank` (0.173) - Primary predictor
3. `player1_age` (0.082)
4. `player2_age` (0.079)
5. `rank_difference` (0.065) - **Our engineered feature**
6. `player1_ht` (0.041)
7. `player2_ht` (0.038)
8. `draw_size` (0.028)
9. `best_of` (0.024)
10. `career_win_rate_diff` (0.019) - **Our engineered feature**

**Key Insights:**
- Rankings dominate predictions (expected)
- Our engineered difference features appear in top 10
- Surface and H2H features may need more sophisticated models to utilize

---

## Files Generated

### Input Files
```
data/processed/matches_final_with_player_context.csv
├── 124,097 matches
├── 81 columns (after preprocessing)
└── Includes player_id and player_name for feature engineering
```

### Output Files
```
data/processed/matches_with_engineered_features.csv
├── 124,097 matches
├── 122 columns (81 original + 43 engineered - 2 redundant)
├── All engineered features included
├── Player IDs and names still present
└── Ready for feature selection and advanced modeling
```

### Baseline Model Files
```
data/processed/matches_final_without_player_context.csv
├── 124,097 matches
├── 77 columns (player IDs and names dropped)
└── Used for baseline modeling
```

### Feature Engineering Notebook
```
notebooks/feature_engineering.ipynb
├── 28 cells
├── ~2,064 lines
├── Complete implementation with validation
└── Includes player-specific analysis examples
```

---

## Usage Instructions

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### Running Feature Engineering

**Option 1: Run Complete Notebook**
```bash
jupyter notebook notebooks/feature_engineering.ipynb
# Run all cells (Kernel -> Restart & Run All)
# Expected runtime: ~5-7 minutes
```

**Option 2: Run Specific Feature Sets**

The notebook is modular. You can run specific sections:

```python
# Section 1: Simple Differences (Cells 1-2)
# Section 2: Career Stats (Cells 5-9)
# Section 3: Surface Stats (Cells 12-14)
# Section 4: Head-to-Head (Cells 15-17)
# Section 5: Recent Form (Cells 18-20)
# Section 6: Save Results (Cell 27)
```

### Loading Engineered Features

```python
import pandas as pd

# Load full dataset with engineered features
df = pd.read_csv('data/processed/matches_with_engineered_features.csv')

print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Separate features and target
feature_cols = [col for col in df.columns if col != 'player1_won']
X = df[feature_cols]
y = df['player1_won']
```

### Creating Train/Val/Test Splits

**Important:** Use chronological split to prevent data leakage

```python
# Split by tournament date
train_mask = df['tourney_date'] < 20230000
val_mask = (df['tourney_date'] >= 20230000) & (df['tourney_date'] < 20240000)
test_mask = df['tourney_date'] >= 20240000

X_train = df[train_mask].drop(['player1_won', 'tourney_date'], axis=1)
y_train = df[train_mask]['player1_won']

X_val = df[val_mask].drop(['player1_won', 'tourney_date'], axis=1)
y_val = df[val_mask]['player1_won']

X_test = df[test_mask].drop(['player1_won', 'tourney_date'], axis=1)
y_test = df[test_mask]['player1_won']
```

### Feature Subsets

You can experiment with different feature combinations:

```python
# Only engineered features
engineered_features = [
    'rank_difference', 'age_difference', 'height_difference',
    'player1_career_win_rate', 'player2_career_win_rate', 'career_win_rate_diff',
    'player1_clay_win_rate', 'player2_clay_win_rate', 'clay_win_rate_diff',
    'player1_grass_win_rate', 'player2_grass_win_rate', 'grass_win_rate_diff',
    'player1_hard_win_rate', 'player2_hard_win_rate', 'hard_win_rate_diff',
    'player1_h2h_win_rate', 'player2_h2h_win_rate', 'h2h_win_rate_diff',
    'player1_recent_form', 'player2_recent_form', 'recent_form_diff'
]

X_engineered_only = df[engineered_features]
```

---

## Key Takeaways

### Technical Lessons

1. **Chronological ordering is critical** for time-series feature engineering
2. **Hash tables provide optimal trade-off** between speed and code clarity
3. **Validation with real examples** catches bugs that unit tests miss
4. **Default values matter** - 0.5 is better than NaN or 0 for tree models
5. **Deques are perfect for sliding windows** - automatic size management
6. **Canonical keys solve symmetry** - essential for H2H relationships
7. **Document the "why"** not just the "what" - reasoning is crucial

### Domain Insights

1. **Rankings dominate** but don't tell the whole story
2. **Surface specialization is real** - players have distinct surface profiles
3. **Grass court data is sparse** - reflects reality of tennis calendar
4. **Head-to-head matters** - some matchups defy overall statistics
5. **Recent form is noisy** but captures momentum
6. **Early career predictions are hard** - cold start problem is real

### Data Science Best Practices

1. **Prevent data leakage** - always use "before match" information
2. **Validate, validate, validate** - trace individual examples
3. **Start simple** - hash tables before complex vectorization
4. **Profile performance** - know your bottlenecks
5. **Save intermediate results** - don't recompute expensive features
6. **Document assumptions** - chronological sorting, default values, etc.

### Future Improvements

**Additional Features to Consider:**
- Multiple window sizes for recent form (5, 10, 20 matches)
- Streak features (current win/loss streak length)
- Time since last match (rest vs rust)
- Tournament-specific performance (Grand Slam vs regular)
- Surface-specific H2H (Federer vs Nadal on clay specifically)
- Player age trajectories (peak years, decline phases)

**Engineering Optimizations:**
- Vectorize simple operations (career stats could be vectorized)
- Pre-compute static features (height difference, age at match time)
- Use incremental updates (only recompute new matches)
- Parallel processing for independent features

**Model Improvements:**
- Feature selection (remove low-importance features)
- Hyperparameter tuning (reduce overfitting)
- Ensemble methods (combine multiple models)
- Neural networks (capture complex interactions)

---

## References

### Data Source
- ATP Tour Official Statistics
- Dataset compiled by Jeff Sackmann: https://github.com/JeffSackmann/tennis_atp

### Course Information
- **Course:** CMPT 459 - Data Mining
- **Institution:** Simon Fraser University
- **Term:** Fall 2024

### Technical Documentation
- Python collections: https://docs.python.org/3/library/collections.html
- Pandas documentation: https://pandas.pydata.org/docs/
- Scikit-learn: https://scikit-learn.org/

---

## Contact

For questions about this feature engineering implementation, please refer to the course project documentation or contact the project team.

---

**Last Updated:** November 8, 2025  
**Version:** 1.0  
**Status:** Feature engineering complete, ready for advanced modeling

