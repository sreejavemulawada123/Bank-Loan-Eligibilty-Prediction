import numpy as np
import pandas as pd
import pickle

# Load the trained model
try:
    with open("models/loan_approval_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    print("❌ Model file not found. Please run the training script first.")
    exit()

# Function to safely get numeric inputs
def safe_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("❌ Enter a valid number.")

def safe_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("❌ Enter a valid integer.")

# Get user input
def get_user_input():
    print("\n📋 Enter your loan details:")

    current_loan_amount = safe_float("Current Loan Amount: ")
    term = input("Loan Term (Short Term/Long Term): ").strip().lower()
    credit_score = safe_float("Credit Score: ")
    annual_income = safe_float("Annual Income: ")
    years_in_job = input("Years in Current Job (< 1 year to 10+ years): ").strip()
    home_ownership = input("Home Ownership (Rent/Home Mortgage/Own Home): ").strip().lower()
    purpose = input("Purpose (Debt Consolidation/Home Improvements/Other): ").strip().lower()
    monthly_debt = safe_float("Monthly Debt: ")
    credit_history = safe_float("Years of Credit History: ")
    months_delinquent = safe_float("Months since last delinquent (-1 if none): ")
    open_accounts = safe_int("Number of Open Accounts: ")
    credit_problems = safe_int("Number of Credit Problems: ")
    current_credit_balance = safe_float("Current Credit Balance: ")
    max_open_credit = safe_float("Maximum Open Credit: ")
    bankruptcies = safe_float("Bankruptcies: ")
    tax_liens = safe_float("Tax Liens: ")

    # Encoding mappings
    term_val = 1 if term == "long term" else 0
    home_map = {"rent": 0, "home mortgage": 1, "own home": 2}
    purpose_map = {"debt consolidation": 0, "home improvements": 1}
    job_map = {
        "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
        "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9, "10+ years": 10
    }

    home_ownership = home_map.get(home_ownership, 0)
    purpose = purpose_map.get(purpose, 2)
    years_in_job = job_map.get(years_in_job, 0)

    # Feature list with correct column names
    feature_names = [
        'Current Loan Amount', 'Term', 'Credit Score', 'Annual Income',
        'Years in current job', 'Home Ownership', 'Purpose', 'Monthly Debt',
        'Years of Credit History', 'Months since last delinquent',
        'Number of Open Accounts', 'Number of Credit Problems',
        'Current Credit Balance', 'Maximum Open Credit', 'Bankruptcies',
        'Tax Liens'
    ]

    input_data = pd.DataFrame([[
        current_loan_amount, term_val, credit_score, annual_income,
        years_in_job, home_ownership, purpose, monthly_debt,
        credit_history, months_delinquent, open_accounts,
        credit_problems, current_credit_balance, max_open_credit,
        bankruptcies, tax_liens
    ]], columns=feature_names)

    return input_data, annual_income, current_loan_amount

# Prediction function
def check_loan_eligibility():
    user_data, annual_income, loan_amount = get_user_input()

    # Optional warning based on income
    if loan_amount > 0.5 * annual_income:
        print("\n⚠️ Warning: Loan amount is more than 50% of your annual income.")

    prediction = model.predict(user_data)

    # Result
    if prediction[0] == 1:
        print("\n✅ Loan Approved!")
    else:
        print("\n❌ Loan Rejected!")

# Run
if __name__ == "__main__":
    check_loan_eligibility()
