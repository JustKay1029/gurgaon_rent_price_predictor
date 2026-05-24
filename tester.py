import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

df = pd.read_csv("real_estate_main.csv")

ggn = df[df["city"]== "Gurgaon"]

print(ggn["property"].value_counts().head(10))
print(ggn["status"].value_counts().head(10))
print(ggn["transaction"].value_counts().head(10))