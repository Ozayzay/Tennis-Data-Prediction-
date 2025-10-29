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
| **Matches**  | 948,000   | 81      | Match results with detailed statistics |
| **Rankings** | 3,240,000 | 4       | Historical ATP rankings                |
| **Players**  | 65,000    | 8       | Player biographical information        |

### Key Features

- **Match Context**: Tournament name, surface (Clay/Hard/Grass), draw size, tournament level, date
- **Player Attributes**: Rankings, ranking points, age, height, handedness, country
- **Match Statistics**: Aces, double faults, service points, break points (when available)
- **Historical Data**: Rankings history spanning multiple decades

### Rationale for Dataset Selection

We selected this dataset because:

- **Rich Features**: Extensive contextual information including player rankings, surfaces, and match statistics
- **Large Sample Size**: Nearly 1 million matches provide robust training data
- **Real-World Application**: Tennis analytics is a growing field with practical applications
- **Domain Relevance**: Clean binary outcomes, balanced data, and well-documented features make it ideal for supervised learning
- **Feature Engineering Opportunities**: Surface-specific performance, head-to-head records, and ELO ratings can be derived

---

## 🚀 Project Structure

```
project/
├── data/
│   ├── raw/                          # Original unmodified data (CSV files)
│   └── processed/                    # Cleaned and preprocessed data
├── notebooks/
│   └── EDA_and_preprocessing.ipynb   # Main analysis notebook
├── results/
│   ├── figures/                      # Visualizations and plots
│   └── models/                       # Trained model files
├── README.md                         # This file (serves as final report)
├── requirements.txt                  # Python dependencies
└── .gitignore                        # Git ignore rules
```

**Note:** We're using a notebook-focused approach for flexibility during the EDA phase. All analysis, preprocessing, feature engineering, and modeling will be done in Jupyter notebooks.

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

#### **Exploratory Data Analysis**

- ✅ Analyzed data distribution across 948,000 matches
- ✅ Visualized feature distributions using histograms and box plots
- ✅ Created correlation heatmaps to identify feature relationships
- ✅ Identified key challenges: missing match statistics (~40%), class balance

#### **Data Cleaning**

- Handled missing values using domain-appropriate strategies
- Normalized numerical features (rank, ranking points, age)
- Encoded categorical variables (surface, handedness, country)
- Created train/test split with temporal validation

#### **Feature Engineering**

- **Surface-Specific Win Rates**: Calculated win rates on Clay, Hard, and Grass surfaces
- **Custom ELO Rating**: Implemented Elo rating system (starting rating: 1500, K-factor: 32)
- **Head-to-Head Records**: Historical performance between specific player matchups
- **Recent Form**: Win rate over last 10 matches
- **Rank Differential**: Difference in ATP rankings and points

### 2. Clustering Analysis

#### **Algorithms Applied**

- **K-Means Clustering** (Player performance profiles)
- **DBSCAN** (Outlier-resistant grouping)
- **Hierarchical Clustering** (Dendrogram-based analysis)

#### **Evaluation Metrics**

- Silhouette Score: [To be filled]
- Calinski-Harabasz Index: [To be filled]
- Davies-Bouldin Index: [To be filled]

#### **Dimensionality Reduction for Visualization**

- PCA (2D and 3D scatter plots)
- t-SNE visualization of clusters

### 3. Outlier Detection

#### **Methods Used**

- **Isolation Forest**: [Results to be filled]
- **Local Outlier Factor (LOF)**: [Results to be filled]
- **Elliptic Envelope**: [Results to be filled]

#### **Analysis**

- Identified anomalous matches (upsets, unusual statistics)
- Determined whether outliers represent noise or important information
- Decision: [Keep/Remove outliers] based on analysis

### 4. Feature Selection

#### **Techniques Applied**

- **Recursive Feature Elimination (RFE)**: [Results]
- **Lasso Regression**: [Results]
- **Mutual Information**: [Results]

#### **Selected Features**

[List of top features to be filled after analysis]

#### **Impact Assessment**

- Model performance with all features: [Accuracy/F1-score]
- Model performance with selected features: [Accuracy/F1-score]
- Computational efficiency improvement: [Time reduction]

### 5. Classification

#### **Algorithms**

- **Random Forest**: [Hyperparameters, results]
- **Support Vector Machines (SVM)**: [Hyperparameters, results]
- **k-Nearest Neighbors (k-NN)**: [Hyperparameters, results]

#### **Train/Test Split**

- Training: 80% of data (temporal split)
- Testing: 20% of data
- Cross-validation: 5-fold or 10-fold

#### **Evaluation Metrics**

| Model         | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| ------------- | -------- | --------- | ------ | -------- | ------- |
| Random Forest | TBD      | TBD       | TBD    | TBD      | TBD     |
| SVM           | TBD      | TBD       | TBD    | TBD      | TBD     |
| k-NN          | TBD      | TBD       | TBD    | TBD      | TBD     |

### 6. Hyperparameter Tuning

#### **Grid Search Results**

- **Random Forest**: [Best parameters]
- **SVM**: [Best parameters]
- **k-NN**: [Best parameters]

#### **Performance Improvement**

- Before tuning: [Baseline accuracy]
- After tuning: [Improved accuracy]
- Impact discussion: [Analysis of how tuning affected performance]

---

## 🔬 Key Results

### Main Findings

1. **[Finding 1]**: [Description of first major insight]

   - Supporting evidence: [Metric/visualization]

2. **[Finding 2]**: [Description of second major insight]

   - Supporting evidence: [Metric/visualization]

3. **Custom ELO vs ATP Rankings**: [Comparison results]
   - Custom ELO accuracy: [X%]
   - ATP ranking accuracy: [Y%]
   - Conclusion: [Which performed better and why]

### Most Important Features

1. [Feature 1]: [Importance score / explanation]
2. [Feature 2]: [Importance score / explanation]
3. [Feature 3]: [Importance score / explanation]

### Visualizations

[Include key plots here or reference them in results/figures/]

---

## 🧩 Challenges Encountered

### 1. Missing Data

- **Challenge**: ~40% of match statistics (aces, double faults) were missing
- **Solution**: [Imputation strategy / feature engineering workaround]

### 2. Class Imbalance

- **Challenge**: [Describe any imbalance if found]
- **Solution**: [SMOTE / class weights / downsampling]

### 3. Feature Engineering Complexity

- **Challenge**: Computing historical features requires temporal ordering
- **Solution**: [Implemented time-aware feature engineering pipeline]

### 4. [Additional Challenge]

- **Challenge**: [Description]
- **Solution**: [How you addressed it]

---

## 📚 Code Organization

### Main Notebook: `EDA_and_preprocessing.ipynb`

This notebook contains all analysis organized in clear sections:

1. **Setup & Data Loading**

   - Import libraries
   - Load ATP matches, rankings, and player data
   - Initial data inspection

2. **Exploratory Data Analysis (EDA)**

   - Data distributions (histograms, box plots)
   - Correlation analysis (heatmaps)
   - Missing value analysis
   - Surface and player statistics

3. **Data Preprocessing**

   - Missing value imputation
   - Feature encoding (categorical variables)
   - Feature normalization/standardization
   - Train/test splitting

4. **Feature Engineering**

   - Surface-specific win rates
   - Custom ELO rating system
   - Head-to-head records
   - Recent form metrics

5. **Modeling & Evaluation**
   - Clustering analysis
   - Outlier detection
   - Feature selection
   - Classification models
   - Hyperparameter tuning
   - Results visualization

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

- [ ] Basic EDA completed (distributions, correlations)
- [ ] Missing value handling
- [ ] Feature encoding and normalization
- [ ] Data augmentation (if applicable)
- [ ] Dimensionality reduction (PCA/t-SNE)

### ✅ Phase 3: Final Project Delivery

- [ ] Clustering analysis (3+ algorithms)
- [ ] Outlier detection (3+ methods)
- [ ] Feature selection
- [ ] Classification (3+ algorithms)
- [ ] Hyperparameter tuning
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
