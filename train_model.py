import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("credit_train.csv")

# Fill missing values (simple forward fill)
data = data.ffill()

# Map categorical fields (manual encoding based on dataset)
term_map = {"Short Term": 0, "Long Term": 1}
home_ownership_map = {"Rent": 0, "Home Mortgage": 1, "Own Home": 2}
purpose_map = {"Debt Consolidation": 0, "Home Improvements": 1}
years_job_map = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10
}

data["Term"] = data["Term"].map(term_map)
data["Home Ownership"] = data["Home Ownership"].map(home_ownership_map).fillna(0).astype(int)
data["Purpose"] = data["Purpose"].map(purpose_map).fillna(2).astype(int)
data["Years in current job"] = data["Years in current job"].map(years_job_map).fillna(0).astype(int)

# Drop rows where Loan Status is missing
data = data[data["Loan Status"].notna()]

# Encode target
label_encoder = LabelEncoder()
data["Loan Status"] = label_encoder.fit_transform(data["Loan Status"])

# Select features and target
features = [
    'Current Loan Amount', 'Term', 'Credit Score', 'Annual Income',
    'Years in current job', 'Home Ownership', 'Purpose', 'Monthly Debt',
    'Years of Credit History', 'Months since last delinquent',
    'Number of Open Accounts', 'Number of Credit Problems',
    'Current Credit Balance', 'Maximum Open Credit',
    'Bankruptcies', 'Tax Liens'
]
X = data[features]
y = data["Loan Status"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
with open("models/loan_approval_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved at: models/loan_approval_model.pkl")
