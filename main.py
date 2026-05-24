import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

df = pd.read_csv("real_estate_main.csv")

ggn = df[df["city"]== "Gurgaon"]

ggn = ggn.drop(columns = ["Unnamed: 0", "Rate", "carpet.area","Rate_per_sqft"])
ggn = ggn.dropna(subset = ["total_area"])
ggn["locality"] = ggn["Name"].str.split(" for Sale in ").str[1].str.split(", Gurgaon").str[0]
p = (ggn["locality"].value_counts())
ggn["clean_sector"] = ggn["locality"].str.split(",").str[-1].str.replace(" Gurgaon", "").str.strip()

sector_counts = ggn["clean_sector"].value_counts() 
rare_sectors = sector_counts[sector_counts < 3].index

ggn.loc[ggn["clean_sector"].isin(rare_sectors), "clean_sector"] = "Other"
#print(ggn["clean_sector"].value_counts())
 
q1 = ggn["total_area"].quantile(0.25)
q3 = ggn["total_area"].quantile(0.75)
iqr = q3 - q1
upper_fence = q3 + (1.5 * iqr)
ggn = ggn[ggn["total_area"] <= upper_fence]

sector_means = ggn.groupby("clean_sector")["Price"].mean().sort_values(ascending=False)
#print(sector_means.head(10))
#print(ggn.columns)
#print(ggn.shape)
#plt.boxplot(ggn["total_area"])
#plt.show()
ggn.loc[ggn["status"].str.startswith("Poss.", na=False), "status"] = "Under Construction"

sector_means = ggn.groupby("clean_sector")["Price"].mean()
ggn["clean_sector"] = ggn["clean_sector"].map(sector_means)

X = ggn.drop(columns=["Price", "Name", "locality", "floor", "facing", "overlooking", "ownership", "parking", "city", "location", "carpet_area_sqft", "property"])
y = ggn["Price"]
#print(X.columns.tolist())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

transformer = ColumnTransformer(transformers=[
    ("name_of_step2", OneHotEncoder(handle_unknown="ignore"), ["status", "transaction"]),
    ("numeric_impute", SimpleImputer(strategy="median"), ["bathroom", "balcony", "bedroom", "total_area", "clean_sector"])
])

X_train_transformed = transformer.fit_transform(X_train, y_train)
X_test_transformed = transformer.transform(X_test)

model = LinearRegression()
model.fit(X_train_transformed, y_train)
predictions = model.predict(X_test_transformed)
score = r2_score(y_test, predictions)
print(f"Your Baseline Model R2 Score is: {score}")
#print(X_train_transformed.shape)

with open("property_pipeline.pkl", "wb") as f:
    pickle.dump((transformer, model), f)
print("Pipeline successfully saved to property_pipeline.pkl!")