import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

data_source = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/adult/adult.data"
)
col_list = [
    "age", "workclass", "fnlwgt", "education",
    "education_num", "marital_status", "occupation",
    "relationship", "race", "sex", "capital_gain",
    "capital_loss", "hours_per_week", "native_country",
    "income"
]
income_df = pd.read_csv(
    data_source,
    names=col_list,
    skipinitialspace=True,
    na_values="?"
)

print("Initial 5 rows:")
print(income_df.head())
print("\nInitial shape:", income_df.shape)

print("\nNull count before cleaning:")
print(income_df.isnull().sum())

numeric_fields = income_df.select_dtypes(include=["int64", "float64"]).columns
for field in numeric_fields:
    income_df[field] = income_df[field].fillna(income_df[field].median())

categorical_fields = income_df.select_dtypes(include=["object", "category"]).columns
for field in categorical_fields:
    income_df[field] = income_df[field].fillna(income_df[field].mode()[0])

print("\nNull count after cleaning:")
print(income_df.isnull().sum())

encoded_df = pd.get_dummies(
    income_df,
    columns=categorical_fields.drop("income").tolist(),
    dtype=int
)
target_encoder = LabelEncoder()
encoded_df["income"] = target_encoder.fit_transform(encoded_df["income"])

print("\nData preview after encoding:")
print(encoded_df.head())

features_to_scale = [
    "age", "fnlwgt", "education_num",
    "capital_gain", "capital_loss",
    "hours_per_week"
]
normalizer = MinMaxScaler()
encoded_df[features_to_scale] = normalizer.fit_transform(encoded_df[features_to_scale])

print("\nData preview after normalization:")
print(encoded_df.head())

encoded_df.to_csv("adult_data_preprocessed.csv", index=False)
print("\nSaved preprocessed dataset!")
print("Output shape:", encoded_df.shape)