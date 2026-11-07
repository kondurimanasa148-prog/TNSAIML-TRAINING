import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
# 1. Load Dataset

df = pd.read_csv(
    "dataset/retail_sales_forecast_dataset.csv"
)
print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)
print("\nDataset Information:")
df.info()
print("\nMissing Values:")
print(df.isnull().sum())
# 2. Select Features
features = [
    "Year",
    "Month",
    "Product_Category",
    "Region",
    "Weather_Condition",
    "Season",
    "Inventory_Level",
    "Unit_Price",
    "Discount_Percent",
    "Promotion",
    "Competitor_Price"
]

# 3. Select Target

target = "Units_Sold"
df[target] = pd.to_numeric(
    df[target],
    errors="coerce"
)
df = df.dropna(
    subset=[target]
)

# 4. Input and Output

X = df[features]
y = df[target]

# 5. Identify Columns

categorical = X.select_dtypes(
    include=["object", "category"]
).columns
numerical = X.select_dtypes(
    include=["number"]
).columns

# 6. Categorical Preprocessing

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),

    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])

# 7. Numerical Preprocessing

numerical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    )
])


# 8. Combine Preprocessing

preprocessor = ColumnTransformer([
    (
        "cat",
        categorical_pipeline,
        categorical
    ),

    (
        "num",
        numerical_pipeline,
        numerical
    )

])

# 9. Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# 10. Random Forest Model

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    )
])

# 11. Train Model

model.fit(
    X_train,
    y_train
)
print("\nModel Training Completed!")

# 12. Prediction

predictions = model.predict(
    X_test
)

mae = mean_absolute_error(
    y_test,
    predictions
)
rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)
r2 = r2_score(
    y_test,
    predictions
)
print("\nModel Evaluation")
print("----------------------------")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 2))

# 14. Save Model

joblib.dump(
    model,
    "model.joblib"
)
print("\nModel saved successfully!")
print("File Name: model.joblib")