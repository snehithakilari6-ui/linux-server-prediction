import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv(r"D:messy_linux_server_prediction_.csv")
print("Dataset Shape:", df.shape)
print(df.head(10))

print("Column Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicate rows after cleaning:", df.duplicated().sum())

df["CPU_Usage"] = (
    df["CPU_Usage"]
    .astype(str)
    .str.replace("%", "", regex=False)
)

df["Disk_Usage"] = (
    df["Disk_Usage"]
    .astype(str)
    .str.replace("%", "", regex=False)
)

numeric_columns = [
    "CPU_Usage",
    "RAM_Usage",
    "Disk_Usage",
    "Network_Traffic",
    "Active_Users",
    "Running_Processes",
    "Server_Uptime",
    "Response_Time",
    "Error_Count"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )

print("Missing values after cleaning:")
print(df.isnull().sum())

df["Server_Status"] = (
    df["Server_Status"]
    .astype(str)
    .str.strip()
    .str.capitalize()
)

print(df["Server_Status"].unique())

print("Cleaned Dataset:")
print(df.head(10))

df.to_csv("cleaned_linux_server_dataset.csv", index=False)

print("Cleaned dataset saved successfully!")


features = [
    "CPU_Usage",
    "RAM_Usage",
    "Disk_Usage",
    "Network_Traffic",
    "Active_Users",
    "Running_Processes",
    "Server_Uptime",
    "Error_Count"
]

X = df[features]
y = df["Response_Time"]

print("Input Features:")
print(X.head())

print("\nTarget:")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=6,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)

print("Random Forest Regressor trained successfully!")

y_pred = model.predict(X_test)

print("Actual Response Time:")
print(y_test.values)

print("\nPredicted Response Time:")
print(np.round(y_pred, 2))

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("========== MODEL RESULTS ==========")
print("MAE       :", round(mae, 2))
print("MSE       :", round(mse, 2))
print("RMSE      :", round(rmse, 2))
print("R² Score  :", round(r2, 4))
print("R² %      :", round(r2 * 100, 2), "%")

results = pd.DataFrame({
    "Metric": [
        "MAE",
        "MSE",
        "RMSE",
        "R² Score"
    ],
    "Value": [
        round(mae, 2),
        round(mse, 2),
        round(rmse, 2),
        round(r2, 4)
    ]
})

print(results)

plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Response Time")
plt.ylabel("Predicted Response Time")
plt.title("Actual vs Predicted Response Time")

plt.show()