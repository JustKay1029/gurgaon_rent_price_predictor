# 🏡 Gurgaon Residential Property Valuation Engine

An end-to-end Machine Learning data product that cleans volatile, web-scraped real estate listings and exposes a trained predictive regression model via an interactive cloud web application.

🚀 **Live Production Link:** https://gurgaonrentpricepredictor-bjktghwqgrt2o5n54nisiq.streamlit.app/

---

## 💻 Application Preview
Select structural components, adjust dimensions, and calculate real estate market evaluations instantly.



---

## 🛠️ Architecture & Pipeline Engineering

The raw data (`real_estate_main.csv`) contained substantial text noise, missing structural metrics, and extreme variance. The development lifecycle was executed across four distinct architectural phases:

### 1. Preprocessing & Data Isolation
* **Geographic Triage:** Filtered features to isolate flat listings strictly located within Gurgaon.
* **Redundancy Pruning:** Eliminated leaky variables (`Rate_per_sqft`, `Rate`, `carpet.area`) to prevent artificial training bias.
* **Low-Frequency Aggregation:** Calculated sector frequency thresholds; sectors appearing fewer than 3 times were compressed into an `"Other"` token to prevent model overfitting.

### 2. Statistical Outlier Mitigation (IQR Framework)
Instead of utilizing arbitrary domain guessing, extreme data variance (e.g., listings exceeding 33,000 sqft) was systematically truncated using the statistical Interquartile Range method:

$$IQR = Q_3 - Q_1$$
$$Upper\ Boundary = Q_3 + (1.5 \times IQR)$$

Any property item scaling past the calculated $Upper\ Boundary$ was stripped to preserve standard consumer pricing accuracy.

### 3. Feature Mapping Pipeline
Data streams are routed via a synchronized Scikit-Learn `ColumnTransformer`:
* **Spatial Target Mapping:** High-cardinality location descriptions (`clean_sector`) are transformed into clean numeric indexes using historical pricing means.
* **Categorical One-Hot Encoding:** Low-cardinality flags (`status`, `transaction`) are parsed into sparse binary matrices.
* **Median Imputation:** Missing numerical entries are caught via `SimpleImputer(strategy="median")` to prevent runtime pipeline breaks.

---

## 🚀 Local Installation & Setup

To replicate this environment locally, execute the following commands inside your terminal:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JustKay1029/gurgaon_rent_price_predictor
   cd gurugram_rent_project
   ```

## Modern Web UI

This repo includes a deployable static frontend:

* `index.html`
* `styles.css`
* `script.js`
* `api/predict.py` for Vercel's Python serverless runtime

### Deploy on Vercel

1. Import the GitHub repository in Vercel.
2. Keep the framework preset as **Other**.
3. Deploy from the repository root.

The frontend calls `/api/predict`, which loads `property_pipeline.pkl` and scores the same scikit-learn pipeline used by the Streamlit prototype.

### Deploy on Netlify

1. Import the GitHub repository in Netlify.
2. Use the included `netlify.toml`.
3. Publish from the repository root.

Netlify serves the polished static experience. Since Netlify does not run the included Python API by default, the UI automatically uses its browser-side preview estimate there.
