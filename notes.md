# Case Study Notes: Gurgaon Real Estate Prediction Pipeline

## Part 1: Data Cleaning & Row Filtering
1. **Target Isolation:** Filtered the global dataset to isolate listings exclusively in "Gurgaon".
2. **Column Pruning:** Dropped redundant or leaky features: `['Unnamed: 0', 'Rate', 'carpet.area', 'Rate_per_sqft']`.
3. **Dropping Empty Values:** Removed rows missing `total_area`. Because area is the most critical numeric driver of real estate value, imputing missing values here would introduce artificial bias into our baseline model.
4. **Text Parsing (Locality Extraction):** Processed messy raw headline strings to extract clean, standardized sector and phase names into a new column: `clean_sector`.
5. **Handling Rare Categories:** Calculated frequency counts for all sectors. Any sector appearing fewer than 3 times was grouped into a catch-all category called `"Other"`. This prevents the model from overfitting to unique, single-listing locations.

## Part 2: Statistical Outlier Mitigation
* Used **Matplotlib Boxplots** to visualize the distribution of property sizes. Found extreme listings reaching over 33,000 sqft.
* Applied the mathematical **Interquartile Range (IQR) Framework** to calculate a clean boundary threshold:
  * $IQR = Q3 - Q1$
  * $Upper Fence = Q3 + (1.5 * IQR)$
* Filtered out any listing where `total_area > Upper Fence` to eliminate extreme luxury mansions that would skew our normal residential pricing curve.



## Part 3: Feature Engineering & Preprocessing Pipeline
Instead of feeding raw text data directly into the model, data columns were routed through a synchronized `ColumnTransformer` based on their specific Data Archetype:

1. **High-Cardinality Mapping (clean_sector):** Instead of exploding the location column into 78 separate columns, we mapped each sector text name directly to its historical average pricing mean using a Pandas `.map()` dictionary. This compressed spatial wealth premiums into a single, high-value numeric column.
2. **Low-Cardinality Encoding (status, transaction):** Used `OneHotEncoder(handle_unknown="ignore")` to convert categories like "Resale" and "Ready to Move" into simple binary flags (0s and 1s).
3. **Numeric Imputation (bathroom, balcony, bedroom, total_area):** Used `SimpleImputer(strategy="median")` to automatically catch any empty fields left behind by agents and plug them using the median value of that specific feature.

------------------------------

# 🏡 Project Case Study: Gurgaon Real Estate Price Prediction Pipeline

## 🛠️ Part 1: Data Preprocessing & Structural Engineering

### 1. Target Market Isolation & Initial Pruning
* **Geographic Filtering:** Sliced the master dataset to isolate residential flat listings strictly located in **Gurgaon**.
* **Feature Pruning:** Dropped redundant or low-value columns (`['Unnamed: 0', 'Rate', 'carpet.area']`) to streamline training data.
* **Handling Critical Missing Values:** Identified 13 rows with missing values in the `total_area` column and explicitly dropped them. Because property dimensions are the primary driver of pricing, imputing synthetic metrics here would introduce major human bias into our data distribution.

### 2. High-Cardinality String Parsing (Locality Extraction)
The original dataset contained raw, erratic string text describing property locations. 
* **The Solution:** Parsed and tokenized these strings to isolate and extract clean, standardized sector designations (e.g., *"Sector 53"*, *"DLF Phase 5"*) into a dedicated column named `clean_sector`.
* **Rare-Category Aggregation:** Calculated unique frequency distributions across all extracted sectors. Any sector appearing **fewer than 3 times** was dynamically grouped into a catch-all category named `"Other"`. This explicitly prevents our machine learning model from overfitting to isolated, non-representative property listings.

### 3. The Mapping Blueprint: Resolving the `.groupby()` Mystery
Before encoding our categorical location column, we ran the following grouping statement:
```python
sector_means = ggn.groupby("clean_sector")["Price"].mean().sort_values(ascending=False)
This calculation served two critical engineering purposes:The Analytical Sanity Check: It allowed us to verify that our web-scraped data mirrored reality. By sorting the averages in descending order, we confirmed that premium neighborhoods like Sector 53 sat at the top (averaging ₹5–6 Crores), proving our data distribution was accurate.The Target-Encoding Blueprint: This generated the statistical mean-dictionary used to convert high-cardinality neighborhood names into robust numeric pricing indexes.4. Text StandardizationCleaned the structural timeline values by tracking variations in agent inputs. Standardized text components using string parsing tools:Pythonggn.loc[ggn["status"].str.startswith("Poss.", na=False), "status"] = "Under Construction"
📊 Part 2: Statistical Outlier Mitigation (The IQR Method)Instead of using arbitrary guesswork to drop large listings, we applied a formal mathematical framework using Box Plot Anatomy and the Interquartile Range (IQR) to clean our distributions.1. Reading the Distribution ArchitectureThe Main Box: Encapsulates the middle 50% of the dataset (the normal residential apartments in Gurgaon, typically sitting between 1,500 and 3,500 sqft).The Center Line: Represents the statistical median (the absolute middle property size).The Upper Whisker (Upper Fence): Represents the boundary for a normal maximum size based on the distribution variance. Any listing floating past this line is a mathematical outlier (with some extreme scraped rows exceeding 33,000 sqft).2. The IQR Mathematical FormulaTo programmatically truncate extreme variance without manual bias, we applied the IQR formula to our feature columns:$$IQR = Q_3 - Q_1$$$$Upper\ Fence = Q_3 + (1.5 \times IQR)$$Any property containing a total_area larger than the computed $Upper\ Fence$ was stripped from our final training data to prevent luxury mansions from skewing our standard consumer pricing predictions.🧠 Part 3: Advanced Core Learnings & Architectural PillarsLearning 1: Target Encoding vs. Dimensional BloatExpanding a high-cardinality text column like clean_sector via standard techniques would spread our dataset sideways into 78 distinct columns, creating a sparse, inefficient matrix.The Strategy: We compressed our spatial location features into a single, high-value numeric index. By mapping each neighborhood to its historical pricing average, our model learns the exact luxury premium of a sector without multiplying the underlying columns.Learning 2: Boolean Masking with .locFor precise, dual-axis data filtering in Pandas, the standard production tool is .loc[]. It ensures explicit execution by assigning criteria simultaneously across rows and columns:Pythondf.loc[row_condition, column_selection]
Learning 3: Data Leakage & Multi-Collinearity (The Senior Check)A critical rule of pipeline engineering is identifying deterministic relationships between features. We explicitly removed Rate_per_sqft from our final training dataframe due to a mathematical dependency:$$Price = total\_area \times Rate\_per\_sqft$$If Rate_per_sqft is left inside the feature matrix, the model achieves a perfect training score by simply multiplying the two columns together. However, this causes catastrophic Data Leakage. In production, an end-user will only know their property's size and bedroom count; they are using your application to discover the valuation. Relying on an upstream variable that is missing at runtime causes applications to crash in production.⚙️ Part 4: The Synchronized Preprocessing PipelineTo prepare our raw matrices for algorithmic computation, features were routed through automated scikit-learn preprocessing chains inside a centralized ColumnTransformer:Plaintext                           ┌──► OneHotEncoder ───► [status, transaction]
                           │
[Raw Input Dataframe] ────┼──► Target Mapping ──► [clean_sector]
                           │
                           └──► SimpleImputer ──► [bedroom, bathroom, balcony, total_area]
Categorical Transformer (OneHotEncoder): Explodes low-cardinality categorical properties (status, transaction) into independent binary flags (0 or 1), enabling linear coefficients to scale them accurately.Numeric Transformer (SimpleImputer): Automatically sweeps numeric values (bedroom, bathroom, balcony, total_area) to catch empty entries left by real estate agents, plugging holes using the statistical column median to avoid introducing scaling errors.