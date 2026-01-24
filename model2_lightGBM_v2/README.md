# Pantry-Aware Recipe Recommender System

This project builds a **pantry-aware recipe recommender** that balances three key objectives:

- **Taste** – will people actually like this?
- **Nutrition** – is it relatively healthy?
- **Sustainability** – how environmentally friendly is it?

The final system uses three LightGBM models (one classifier + two regressors) and a simple ranking layer to suggest recipes a user can cook with what they already have at home.

## Note

For all experiments and the final demo we use **`pipeline_classification.py`**.

`pipeline.py` is kept as the earlier version where taste was modeled as a regression target, and is mainly used as a baseline for comparison.

---

## 1. Data

We combine two public datasets:

### Food.com Recipes

- **231,637 recipes**
- **Fields used:**
  - `name`, `minutes`, `n_steps`, `n_ingredients`
  - `ingredient_text` (ingredient list as text)
  - `calories`, `fat`, `sat_fat`, `carbs`, `sugar`, `protein`, `sodium`
  - `avg_rating` (user rating 0–5)

This dataset is our source for:

- **Taste labels** (user ratings)
- **Recipe metadata** (time, number of steps, etc.)
- **Ingredient text** (for TF-IDF features and pantry coverage)

### Open Food Facts (OFF)

- **1,982,824 products total**
- We sub-sample:

  - **200,000 products** with a valid Nutri-Score (A–E, internally mapped to a numeric score −15 to 40)
  - **150,000 products** with a valid Eco-Score (0–100)

- **Fields used:**
  - Nutritional macros per 100g (energy, fat, carbs, sugar, protein, sodium, …)
  - `product_name` (used as a rough proxy for ingredient composition)

OFF is treated as an **external label source** that teaches the model what "healthy" and "sustainable" food looks like based on real Nutri-Score and Eco-Score annotations.

---

## 2. Problem Setup

The model is multi-task, but we train each target separately:

### Taste Model (Food.com)

- **Task:** Binary classification
- **Label:**
  - `GOOD (1)` if average rating ≥ 4.5
  - `NOT_RECOMMENDED (0)` otherwise
- **Interpretation:** "Is this recipe good enough to recommend at all?"

### Nutrition Model (OFF)

- **Task:** Regression
- **Target:** Nutri-Score numeric value (−15…40, lower is healthier)

### Sustainability Model (OFF)

- **Task:** Regression
- **Target:** Eco-Score (0…100, higher is better)

At recommendation time, all three predictions are combined into a single ranking score.

---

## 3. Feature Engineering

Across the three models we reuse the same ideas:

### 3.1 Hand-crafted Numeric Features

For both recipes and OFF products we map nutritional macros:

| Food.com column | OFF column           |
| --------------- | -------------------- |
| `calories`      | `energy-kcal_100g`   |
| `fat`           | `fat_100g`           |
| `sat_fat`       | `saturated-fat_100g` |
| `carbs`         | `carbohydrates_100g` |
| `sugar`         | `sugars_100g`        |
| `protein`       | `proteins_100g`      |
| `sodium`        | `sodium_100g`        |

For the **taste model** we also add:

- `minutes`
- `n_steps`
- `n_ingredients`

These capture rough recipe complexity and effort.

### 3.2 Text Features (TF-IDF + SVD)

We use bag-of-words on text:

- **Food.com:**

  - `ingredient_text` → TF-IDF with vocabulary size 5,000

- **OFF:**
  - `product_name` → TF-IDF with vocabulary size 5,000

Since 5,000-dim TF-IDF is too large, we apply **Truncated SVD** to reduce it to **100 components**.  
This gives us dense 100-dim vectors for ingredients/product names.

### 3.3 Final Feature Shapes

From the logs:

- **Taste (Food.com):**  
  macros (7) + meta (3) + TF-IDF SVD (100) → **110 features**

- **Nutrition (OFF):**  
  macros (7) + TF-IDF SVD (100) → **107 features**

- **Sustainability (OFF):**  
  macros (7) + TF-IDF SVD (100) → **107 features**

---

## 4. Models

All models are **LightGBM**:

- **Taste:** binary classification with `binary_logloss`
- **Nutrition:** regression with `rmse`
- **Sustainability:** regression with `rmse`

**Training:**

- 80/20 train/validation split
- Early stopping with up to 4,000 boosting rounds
- Validation monitoring every 200 iterations

---

## 5. Evaluation Results

### 5.1 Taste – Binary Classification

From `pipeline_classification.py`:

**Class split (threshold 4.5 stars):**

- `NOT_RECOMMENDED`: 37.8% (87,460 recipes)
- `GOOD`: 62.2% (144,177 recipes)

**Metrics (weighted):**

- **Accuracy:** 0.625
- **Precision:** 0.589
- **Recall:** 0.625
- **F1-score:** 0.520

**Confusion matrix (normalized):**

- For class `GOOD`:
  - **Recall ≈ 0.962** – we catch almost all truly good recipes.
- For class `NOT_RECOMMENDED`:
  - Recall is low (≈ 0.069), i.e., the model tends to err on the side of predicting `GOOD`.

**Interpretation:**  
This is intentional: in the recommender, taste acts as a **soft gate**, and we mostly care about **not missing good recipes**. False positives (borderline recipes predicted GOOD) are later filtered by nutrition/sustainability and the final ranking.

The plot `plots_classification/Taste_Binary_classification_evaluation.png` shows:

- Confusion matrix and normalized confusion matrix
- Weighted metrics bar chart
- Per-class precision/recall/F1 table

### 5.2 Nutrition – Regression (Nutri-Score)

On a **40,000-sample hold-out set:**

- **RMSE:** 2.46
- **MAE:** 1.56
- **R²:** 0.923

Even when we convert continuous predictions into Nutri-Score categories (A–E):

- **Exact category accuracy:** 82.4%
- **Within 1 category:** 99.4%

The evaluation plot (`plots_classification/Nutrition_evaluation.png`) shows:

- Actual vs predicted scatter with the y=x line
- Residual distribution centered around zero
- Key metric bar chart (RMSE, MAE, R²)
- Confusion matrix over Nutri-Score categories

**Conclusion:** The nutrition model is strong and behaves as a well-calibrated approximator of Nutri-Score.

### 5.3 Sustainability – Regression (Eco-Score)

On a **30,000-sample hold-out set:**

- **RMSE:** 13.29
- **MAE:** 9.33
- **R²:** 0.695

More important for recommendation is the **ranking quality:**

- **Spearman ρ:** 0.836
- **Kendall τ:** 0.660

So even if the absolute Eco-Score has some error, the **relative ordering** of products is mostly correct.

The plot (`plots_classification/Sustainability_evaluation.png`) includes:

- Actual vs predicted scatter
- Residual distribution
- Key metrics bar chart
- Error heatmap colored by absolute error
- Spearman and Kendall correlation bars

---

## 6. Why We Switched Taste from Regression to Classification

Originally (`pipeline.py`) taste was trained as a regression model on the 0–5 star rating:

```
Taste regression (with TF-IDF):
    RMSE = 0.9795
    R²   = 0.0189
```

An **R² of ~0.02** basically means:  
_"We explain only 2% of the variance in the ratings; the model is barely better than predicting the global average rating for every recipe."_

### Why did this happen?

There are two main reasons:

1. **Rating distribution is extremely skewed.**  
   Most recipes sit between 4.0 and 5.0 with a mean around 4.45 and a small standard deviation.  
   Trying to predict whether a recipe will be 4.3 vs 4.6 stars is almost impossible from ingredients alone.

2. **Ratings include a lot of noise we can't model.**  
   User reviews depend on:

   - how strictly they followed the instructions
   - substitutions they made
   - personal taste, presentation, etc.

   Our features (ingredients, macros, basic metadata) simply don't contain enough signal to support precise regression.

### The solution: Binary classification

However, for the recommender we don't actually need "4.37 vs 4.52".  
What we care about is:

**"Is this recipe good enough to recommend at all?"**

So we reframed the problem as:

- `GOOD` if rating ≥ 4.5
- `NOT_RECOMMENDED` otherwise

This gives us:

- **Cleaner labels** (less noise)
- A **decision boundary** that aligns with what users care about ("excellent" recipes)
- A **probability P(GOOD)** we can directly plug into the ranking function

Even though accuracy is only around **62.5%** (close to the 62.2% majority baseline), the **recall for GOOD recipes is ~96%**, which is exactly what we want for a quality gate: it **very rarely discards truly good recipes**.

---

## 7. Recommendation Layer

The last part of the pipeline turns model outputs into actual recipe suggestions.

### 7.1 Filtering by Pantry

Given a user pantry (set of ingredients), we:

1. **Keep only recipes with at least 5 ingredients**  
   (to avoid extremely small side dishes / sauces).

2. **Compute pantry coverage:**

   ```
   coverage = #{ingredients in pantry} / #{ingredients in recipe}
   ```

3. **Filter to recipes with coverage ≥ 80%.**

For the example pantry in `pipeline_classification.py`:

- 231,637 total recipes
- 169,978 recipes have ≥ 5 ingredients
- **55 recipes** pass the 80% pantry coverage filter

### 7.2 Scoring

For each candidate recipe we compute:

- `taste_score` = predicted probability of GOOD
- `nutrition_score` = predicted Nutri-Score (lower is better)
- `sustainability_score` = predicted Eco-Score (higher is better)

Then we normalize them to [0, 1] and combine:

```python
final_score = (
    0.4 * norm_taste +
    0.3 * norm_nutrition +
    0.3 * norm_sustainability
)
```

**Weights** reflect the idea that taste should matter slightly more than the other two, but nutrition and sustainability still have a strong influence.

The **top-K recipes** are sorted by `final_score` and printed together with:

- Taste prediction (GOOD / NOT_RECOMMENDED + probability)
- Predicted Nutri-Score
- Predicted Eco-Score
- Pantry coverage
- Missing ingredients list

### 7.3 Recommendation Metrics

For the classification pipeline, on a demo pantry:

**Precision@K** (treating "GOOD and rating ≥ 4.5" as relevant):

- P@5 = 60%
- P@10 = 80%

**nDCG@K** (ranking quality based on true rating within the top K):

- nDCG@5 ≈ 0.95
- nDCG@10 ≈ 0.96

**Multi-objective improvement vs random baseline** (averaged over 100 random samples):

| Metric                                    | Baseline | Our Method | Improvement |
| ----------------------------------------- | -------- | ---------- | ----------- |
| Nutrition (Nutri-Score, lower better)     | 18.87    | 9.65       | **+9.23**   |
| Sustainability (Eco-Score, higher better) | 38.51    | 47.40      | **+8.89**   |
| Taste (avg rating, higher better)         | 4.55     | 4.72       | **+0.17**   |

**Conclusion:**  
Compared to random recipes that match the pantry, our recommender:

- Picks recipes that are **healthier and more sustainable**
- Keeps taste **at least as good**, slightly better on average

All of these metrics are visualized in  
`plots_classification/recommendation_metrics_evaluation.png`.

---

## 8. How to Run

### 8.1 Setup

```bash
git clone <this-repo>
cd model2_lightGBM_v2

python -m venv venv
source venv/bin/activate        # on macOS / Linux
# .\venv\Scripts\activate       # on Windows

pip install -r requirements.txt   # if available
# or manually install:
# pip install lightgbm pandas numpy scikit-learn matplotlib seaborn
```

Make sure the Food.com and OFF CSV files are in the expected `data/` folder (paths are set near the top of `pipeline_classification.py`).

### 8.2 Run the Classification Pipeline (Final Version)

```bash
python pipeline_classification.py
```

This will:

1. Load and preprocess datasets
2. Train the three models (taste classifier + 2 regressors)
3. Save model artifacts under `artifacts/models/`
4. Generate evaluation plots under `plots_classification/`
5. Save per-recipe predictions to `data/recipes_with_predictions_classification.csv`
6. Run a pantry-based recommendation demo and save the top-K to `data/top_recommendations_classification.csv`

### 8.3 Optional: Run the Regression Baseline

```bash
python pipeline.py
```

This is the older version where taste is modeled via regression.  
We mainly keep it for **comparison and ablation**.

---

## 9. Key Visualizations

All plots are saved to `plots_classification/`:

1. **`Taste_Binary_classification_evaluation.png`**

   - Confusion matrices (raw + normalized)
   - Weighted metrics bar chart
   - Per-class precision/recall/F1 table

2. **`Nutrition_evaluation.png`**

   - 2×3 grid showing:
     - Actual vs predicted scatter
     - Residual distribution
     - RMSE/MAE/R² bars
     - Nutri-Score category confusion matrix
     - Category accuracy bars (exact + within-1)
     - Summary table

3. **`Sustainability_evaluation.png`**

   - 2×3 grid showing:
     - Actual vs predicted scatter
     - Residual distribution
     - RMSE/MAE/R² bars
     - Spearman/Kendall correlation bars
     - Error heatmap by score range
     - Summary table

4. **`recommendation_metrics_evaluation.png`**
   - 4-subplot figure:
     - Precision@K bars (K=5, 10)
     - nDCG@K bars (K=5, 10)
     - Multi-objective comparison (baseline vs our method)
     - Improvement bar chart (nutrition/eco/taste)

---

## 10. Limitations and Future Work

### Known Limitations

1. **Text Domain Mismatch**  
   The sustainability model is trained on OFF `product_name` (e.g., "Organic Almond Milk") but applied to Food.com `ingredient_text` (e.g., "almond milk, honey, oats").  
   Despite this mismatch, the R² of 0.695 and high ranking correlations (Spearman ρ = 0.836) suggest the model generalizes reasonably well, likely due to:

   - Shared food vocabulary across domains
   - Macro nutrients as primary features (text as secondary)
   - Semantic similarity in TF-IDF space

2. **Taste Classification Accuracy**  
   The 62.5% accuracy is only slightly above the 62.2% majority baseline.  
   However, the **96% recall for GOOD recipes** is what matters for our use case—we prioritize not missing good recipes over perfect precision.

3. **Ingredient Filtering**  
   The ≥5 ingredients filter is a heuristic to exclude simple recipes. This threshold may need adjustment based on user feedback.

### Future Improvements

- **Fine-tune text features:** Train a domain-specific embedding model on food text
- **Add user personalization:** Incorporate dietary restrictions, allergies, taste preferences
- **Improve sustainability predictions:** Use actual ingredient-level LCA data if available
- **Hyperparameter optimization:** Grid search over LightGBM parameters and SVD components
- **A/B testing:** Validate multi-objective weights (0.4/0.3/0.3) with real users

---

## 11. Implementation Architecture

This section explains how the code is actually structured and what each major component does.

### 11.1 Overall Pipeline Flow

```
┌─────────────────┐
│  Load Raw Data  │  ← Food.com recipes CSV + OFF products CSV
└────────┬────────┘
         ↓
┌─────────────────┐
│  Preprocessing  │  ← Clean nulls, filter valid samples, create labels
└────────┬────────┘
         ↓
┌─────────────────┐
│ Feature Engine  │  ← Extract macros + TF-IDF + SVD
└────────┬────────┘
         ↓
┌─────────────────┐
│  Train Models   │  ← 3 separate LightGBM models (taste, nutrition, sustainability)
└────────┬────────┘
         ↓
┌─────────────────┐
│  Evaluate       │  ← Generate metrics + confusion matrices + plots
└────────┬────────┘
         ↓
┌─────────────────┐
│ Predict All     │  ← Apply models to Food.com recipes
│  Recipes        │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Recommendation  │  ← Filter by pantry → Score → Rank → Return top-K
└─────────────────┘
```

### 11.2 Data Loading and Preprocessing

**Function:** `load_and_preprocess_data()`

**What it does:**

1. **Load Food.com recipes** (`RAW_RECIPES` CSV)

   - Keeps only rows with valid `avg_rating` (taste label)
   - Cleans nutritional columns (calories, fat, protein, etc.)
   - **Creates binary taste label:**
     ```python
     food_df['taste_class'] = (food_df['avg_rating'] >= 4.5).astype(int)
     # 1 = GOOD, 0 = NOT_RECOMMENDED
     ```

2. **Load OFF products** (`OFF_CSV`)
   - Sub-samples 200K products with valid `nutriscore_score`
   - Sub-samples 150K products with valid `ecoscore_score`
   - Cleans nutritional columns (per 100g macros)
   - Keeps `product_name` as text feature

**Output:**

- `food_df` → 231K recipes with taste labels
- `nutrition_df` → 200K OFF products with Nutri-Score
- `sustainability_df` → 150K OFF products with Eco-Score

### 11.3 Feature Engineering Pipeline

**Function:** `extract_features(df, text_column, vectorizer=None, svd=None)`

**Input:** DataFrame + text column name (either `ingredient_text` or `product_name`)

**Steps:**

1. **Extract macro features** (7 columns):

   ```python
   macro_features = ['calories', 'fat', 'sat_fat', 'carbs',
                     'sugar', 'protein', 'sodium']
   X_macro = df[macro_features].fillna(0)
   ```

2. **For taste model only**, add metadata (3 columns):

   ```python
   meta_features = ['minutes', 'n_steps', 'n_ingredients']
   X_meta = df[meta_features].fillna(0)
   ```

3. **Text vectorization (TF-IDF):**

   ```python
   vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
   X_text_sparse = vectorizer.fit_transform(df[text_column])
   # Result: (n_samples, 5000) sparse matrix
   ```

4. **Dimensionality reduction (SVD):**

   ```python
   svd = TruncatedSVD(n_components=100, random_state=42)
   X_text_dense = svd.fit_transform(X_text_sparse)
   # Result: (n_samples, 100) dense matrix
   ```

5. **Concatenate all features:**
   ```python
   X_final = np.hstack([X_macro, X_meta, X_text_dense])
   # Taste: 7 + 3 + 100 = 110 features
   # Nutrition/Sustainability: 7 + 100 = 107 features
   ```

**Why TF-IDF + SVD?**

- TF-IDF captures "which words appear in this text"
- 5,000 vocab is too large → SVD compresses to 100 dims
- 100 dims is small enough for LightGBM but still captures semantic patterns

**Output:**

- Feature matrix `X` (numpy array)
- Fitted `vectorizer` and `svd` objects (saved for later prediction)

### 11.4 Model Training

**Function:** `train_lightgbm_model(X_train, y_train, X_val, y_val, task='regression')`

**What it does:**

1. **Split data 80/20:**

   ```python
   X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
   ```

2. **Set LightGBM parameters based on task:**

   ```python
   if task == 'classification':
       params = {
           'objective': 'binary',
           'metric': 'binary_logloss',
           'boosting_type': 'gbdt',
           'num_leaves': 31,
           'learning_rate': 0.05,
           'feature_fraction': 0.9,
           'verbose': -1
       }
   else:  # regression
       params = {
           'objective': 'regression',
           'metric': 'rmse',
           'boosting_type': 'gbdt',
           'num_leaves': 31,
           'learning_rate': 0.05,
           'feature_fraction': 0.9,
           'verbose': -1
       }
   ```

3. **Train with early stopping:**
   ```python
   model = lgb.train(
       params,
       train_data,
       num_boost_round=4000,
       valid_sets=[val_data],
       callbacks=[lgb.early_stopping(stopping_rounds=200),
                  lgb.log_evaluation(period=200)]
   )
   ```
   - Stops if validation metric doesn't improve for 200 rounds
   - Monitors every 200 iterations

**Output:**

- Trained LightGBM model
- Best iteration number (e.g., "stopped at iteration 1523")

**Three separate training runs:**

1. **Taste model** (classification)
   - Trained on Food.com with `taste_class` labels
   - Features: macros + meta + ingredient TF-IDF
2. **Nutrition model** (regression)
   - Trained on OFF with `nutriscore_score` targets
   - Features: macros + product_name TF-IDF
3. **Sustainability model** (regression)
   - Trained on OFF with `ecoscore_score` targets
   - Features: macros + product_name TF-IDF

### 11.5 Evaluation and Metrics

**Function:** `evaluate_and_plot_<model_type>(...)`

**For Classification (Taste):**

```python
# Get predictions
y_pred_proba = model.predict(X_val)  # probabilities [0, 1]
y_pred_class = (y_pred_proba >= 0.5).astype(int)  # binary labels

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred_class)
precision, recall, f1 = classification_report(...)
confusion_matrix = confusion_matrix(y_true, y_pred_class)

# Plot confusion matrix + metrics bar chart
```

**For Regression (Nutrition/Sustainability):**

```python
# Get predictions
y_pred = model.predict(X_val)

# Calculate metrics
rmse = sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# For Nutri-Score: convert to categories and check accuracy
categories_true = [nutriscore_to_category(score) for score in y_true]
categories_pred = [nutriscore_to_category(score) for score in y_pred]
exact_accuracy = accuracy_score(categories_true, categories_pred)

# For Eco-Score: calculate ranking correlation
spearman_rho = spearmanr(y_true, y_pred)[0]
kendall_tau = kendalltau(y_true, y_pred)[0]

# Plot actual vs predicted + residuals + metrics
```

### 11.6 Prediction on All Recipes

**Function:** Applied in `main()` after training

**What it does:**

1. **Extract features from all Food.com recipes:**

   ```python
   # Use the SAME vectorizer/svd fitted during training
   X_all_recipes = extract_features(
       food_df,
       text_column='ingredient_text',
       vectorizer=taste_vectorizer,  # reuse!
       svd=taste_svd                 # reuse!
   )
   ```

2. **Apply all three models:**

   ```python
   food_df['pred_taste_proba'] = taste_model.predict(X_all_recipes)
   food_df['pred_nutrition'] = nutrition_model.predict(X_all_recipes)
   food_df['pred_sustainability'] = sustainability_model.predict(X_all_recipes)
   ```

3. **Save to CSV:**
   ```python
   food_df.to_csv('data/recipes_with_predictions_classification.csv')
   ```

**Key point:** We reuse the **same TF-IDF vocabulary and SVD transformation** from training.  
This ensures consistency: if "chicken" was token #347 during training, it's still #347 now.

### 11.7 Recommendation System

**Function:** `recommend_recipes_multi_objective(...)`

**Input:**

- All recipes with predictions
- User pantry: `['chicken', 'rice', 'garlic', 'onion', ...]`
- Minimum coverage threshold: 0.8 (80%)
- Top-K: 10

**Steps:**

1. **Filter by ingredient count:**

   ```python
   recipes = recipes[recipes['n_ingredients'] >= 5]
   # Removes simple side dishes
   ```

2. **Calculate pantry coverage for each recipe:**

   ```python
   def pantry_coverage(recipe_ingredients, user_pantry):
       recipe_set = set(recipe_ingredients.split(', '))
       pantry_set = set(user_pantry)
       available = len(recipe_set & pantry_set)
       return available / len(recipe_set)

   recipes['coverage'] = recipes['ingredient_text'].apply(
       lambda x: pantry_coverage(x, user_pantry)
   )
   ```

3. **Filter by coverage threshold:**

   ```python
   candidates = recipes[recipes['coverage'] >= 0.8]
   # Only keep recipes where user has ≥80% of ingredients
   ```

4. **Normalize scores to [0, 1]:**

   ```python
   # Taste: already a probability (0-1)
   norm_taste = candidates['pred_taste_proba']

   # Nutrition: lower is better, so invert
   norm_nutrition = 1 - (candidates['pred_nutrition'] - min_nutri) / (max_nutri - min_nutri)

   # Sustainability: higher is better
   norm_sustainability = (candidates['pred_sustainability'] - min_eco) / (max_eco - min_eco)
   ```

5. **Calculate final score:**

   ```python
   final_score = (
       0.4 * norm_taste +
       0.3 * norm_nutrition +
       0.3 * norm_sustainability
   )
   ```

6. **Sort and return top-K:**
   ```python
   top_k = candidates.nlargest(10, 'final_score')
   return top_k[['name', 'pred_taste_proba', 'pred_nutrition',
                 'pred_sustainability', 'coverage', 'final_score']]
   ```

**Output:** Top 10 recipes ranked by multi-objective score

### 11.8 Key Design Decisions

**1. Why separate models instead of multi-task learning?**

- Different datasets: Food.com (taste) vs OFF (nutrition/sustainability)
- Different tasks: classification vs regression
- Easier to debug and tune each model independently

**2. Why reuse vectorizer/svd instead of re-fitting?**

- Ensures consistency: training vocabulary = prediction vocabulary
- Prevents "unknown token" issues
- Standard practice in ML pipelines

**3. Why 0.4/0.3/0.3 weights?**

- Taste is slightly more important (people won't eat healthy food they hate)
- But nutrition and sustainability still have strong influence
- Weights are tunable hyperparameters

**4. Why 80% pantry coverage?**

- Balance between "cookable with what I have" and "enough candidates"
- Too high (90%+) → very few matches
- Too low (50%) → too many missing ingredients

### 11.9 File Interactions

```
pipeline_classification.py  ←── Main entry point
    ↓
    ├── Loads: data/RAW_recipes.csv (Food.com)
    ├── Loads: data/OFF.csv (Open Food Facts)
    ↓
    ├── Calls: load_and_preprocess_data()
    ├── Calls: extract_features()  (3 times: taste, nutrition, sustainability)
    ├── Calls: train_lightgbm_model()  (3 times)
    ├── Calls: evaluate_and_plot_*()  (3 times)
    ↓
    ├── Saves: artifacts/models/taste_classifier.txt
    ├── Saves: artifacts/models/nutrition_regressor.txt
    ├── Saves: artifacts/models/sustainability_regressor.txt
    ├── Saves: artifacts/models/taste_vectorizer.pkl
    ├── Saves: artifacts/models/taste_svd.pkl
    ├── (similar for nutrition/sustainability)
    ↓
    ├── Saves: plots_classification/*.png (evaluation plots)
    ├── Saves: data/recipes_with_predictions_classification.csv
    ↓
    └── Calls: recommend_recipes_multi_objective()
        └── Saves: data/top_recommendations_classification.csv
```

### 11.10 What Each Major Function Learns

| Function                       | Input                       | Learns                                                                          | Output                           |
| ------------------------------ | --------------------------- | ------------------------------------------------------------------------------- | -------------------------------- |
| `TfidfVectorizer.fit()`        | Text corpus                 | Word importance weights (IDF values)                                            | Vocabulary mapping + IDF weights |
| `TruncatedSVD.fit()`           | TF-IDF matrix               | Principal components (latent semantic dimensions)                               | 100 semantic axes                |
| `lgb.train()` (taste)          | Food.com features + ratings | Decision tree ensemble rules: "if protein > X and has 'chicken' → GOOD"         | LightGBM classifier              |
| `lgb.train()` (nutrition)      | OFF features + Nutri-Scores | Regression rules: "if sugar high + sat_fat high → Nutri-Score ~25 (unhealthy)"  | LightGBM regressor               |
| `lgb.train()` (sustainability) | OFF features + Eco-Scores   | Regression rules: "if has 'organic' + low energy → Eco-Score ~60 (sustainable)" | LightGBM regressor               |

---

## 12. Project Structure

```
model2_lightGBM_v2/
├── pipeline_classification.py    # Main pipeline (classification for taste)
├── pipeline.py                    # Baseline pipeline (regression for taste)
├── models/
│   └── preprocess_foodcom.py     # Preprocessing utilities
├── data/
│   ├── recipes_with_predictions_classification.csv
│   └── top_recommendations_classification.csv
├── artifacts/
│   └── models/                    # Saved LightGBM models
├── plots_classification/          # Evaluation plots
└── README.md                      # This file
```

---

## 13. References

- [Food.com Recipes Dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
- [Open Food Facts](https://world.openfoodfacts.org/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Nutri-Score Documentation](https://www.santepubliquefrance.fr/en/nutri-score)
- [Eco-Score Documentation](https://docs.score-environnemental.com/)

---
