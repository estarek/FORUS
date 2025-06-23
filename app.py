
# --- IMPORTANT: Install Dependencies First ---
# Before running this Streamlit application, ensure you have all required libraries installed.
# You can do this by creating a 'requirements.txt' file in the same directory as this script
# with the following content:
#
# streamlit
# pandas
# Pillow
# joblib
# scikit-learn
#
# Then, open your terminal or command prompt, navigate to this directory, and run:
# pip install -r requirements.txt
# ---------------------------------------------

import streamlit as st
import pandas as pd
from PIL import Image
import joblib
import os
import random # Used in the original data generation logic, kept for completeness
import numpy as np # Used in the original data generation logic, kept for completeness
from datetime import datetime, timedelta # Used in the original data generation logic, kept for completeness

# --- Configuration and Data Loading ---
# IMPORTANT: For this Streamlit app to run, you need to place the following files
# in the same directory as your Streamlit Python script, or adjust the DATA_PATH.
# 1. borrowers_data.csv
# 2. loans_data.csv
# 3. payments_data.csv
# 4. random_forest_model.pkl
DATA_PATH = "./" # Current directory. Change to e.g., "./data/" if you put files in a 'data' folder.

@st.cache_data # Cache data loading to avoid reloading on every rerun
def load_all_data():
    try:
        borrowers_df = pd.read_csv(os.path.join(DATA_PATH, "borrowers_data.csv"))
        loans_df = pd.read_csv(os.path.join(DATA_PATH, "loans_data.csv"))
        payments_df = pd.read_csv(os.path.join(DATA_PATH, "payments_data.csv"))
        return borrowers_df, loans_df, payments_df
    except FileNotFoundError as e:
        st.error(f"Required data file not found: {e}. Please ensure 'borrowers_data.csv', 'loans_data.csv', and 'payments_data.csv' are in the '{DATA_PATH}' directory.")
        st.stop() # Stop the app if data is not found

@st.cache_resource # Cache model loading to avoid reloading on every rerun
def load_ml_model():
    try:
        model = joblib.load(os.path.join(DATA_PATH, "random_forest_model.pkl"))
        return model
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}. Please ensure 'random_forest_model.pkl' is in the '{DATA_PATH}' directory.")
        st.stop() # Stop the app if model is not found

# Load data and model at the start of the script
borrowers, loans, payments = load_all_data()
loaded_model = load_ml_model()

# --- Global Mappings for Categorical Features (from training data) ---
# These mappings are crucial and MUST be consistent with how the model was trained.
# They are derived from the loaded 'borrowers' DataFrame.
try:
    # Ensure these columns exist in your borrowers_data.csv
    # The original Gradio code had 'df' which was undefined for these mappings.
    # We assume 'borrowers' DataFrame contains the necessary columns for creating these maps.
    sector_map = {k: v for v, k in enumerate(borrowers['sector'].astype('category').cat.categories)}
    region_map = {k: v for v, k in enumerate(borrowers['region'].astype('category').cat.categories)}
    income_bracket_map = {k: v for v, k in enumerate(borrowers['income_bracket'].astype('category').cat.categories)}
    gender_map = {k: v for v, k in enumerate(borrowers['gender'].astype('category').cat.categories)}
except KeyError as e:
    st.error(f"Error creating categorical mappings: Column '{e}' not found in 'borrowers_data.csv'. "
             "Please ensure the CSV contains 'sector', 'region', 'income_bracket', and 'gender' columns, "
             "or adjust the mapping source if your model was trained with different data.")
    st.stop()

# --- Comprehensive Lists and Risk Logic (extracted from your data generation code) ---

# Job Titles and Risk Mapping
ALL_JOB_TITLES = [
    "Engineer", "Accountant", "Retail Manager", "Doctor", "Trader", "Laborer",
    "Nurse", "Technician", "Electrician", "Plumber", "Construction Worker",
    "Driver", "Teacher", "Professor", "IT Specialist", "Software Developer",
    "Civil Engineer", "Mechanical Engineer", "Electrical Engineer",
    "Architect", "Project Manager", "Site Supervisor", "HR Manager",
    "Sales Representative", "Customer Service Agent", "Warehouse Worker",
    "Security Guard", "Cleaner", "Chef", "Waiter", "Barista",
    "Pharmacist", "Radiologist", "Dentist", "Surgeon", "Financial Analyst",
    "Legal Advisor", "Bank Teller", "Operations Manager", "Procurement Officer",
    "Administrative Assistant", "Receptionist", "Marketing Specialist",
    "Content Creator", "Logistics Coordinator", "Supply Chain Manager"
]

HIGH_RISK_JOBS = [
    "Laborer", "Construction Worker", "Cleaner", "Waiter", "Barista",
    "Driver", "Security Guard", "Warehouse Worker", "Retail Manager",
    "Sales Representative", "Customer Service Agent", "Chef", "Trader",
    "Technician", "Administrative Assistant", "Receptionist", "Content Creator"
]

def is_high_risk_job_func(job_title):
    return job_title in HIGH_RISK_JOBS

JOB_SECTOR_MAP = {
    "Engineer": "Industrial", "Accountant": "Services", "Retail Manager": "Retail",
    "Doctor": "Services", "Trader": "Retail", "Laborer": "Industrial",
    "Nurse": "Services", "Technician": "Industrial", "Electrician": "Industrial",
    "Plumber": "Industrial", "Construction Worker": "Industrial", "Driver": "Services",
    "Teacher": "Services", "Professor": "Services", "IT Specialist": "Tech",
    "Software Developer": "Tech", "Civil Engineer": "Industrial", "Mechanical Engineer": "Industrial",
    "Electrical Engineer": "Industrial", "Architect": "Industrial", "Project Manager": "Services",
    "Site Supervisor": "Industrial", "HR Manager": "Services", "Sales Representative": "Retail",
    "Customer Service Agent": "Services", "Warehouse Worker": "Industrial", "Security Guard": "Services",
    "Cleaner": "Services", "Chef": "Services", "Waiter": "Services", "Barista": "Retail",
    "Pharmacist": "Services", "Radiologist": "Services", "Dentist": "Services", "Surgeon": "Services",
    "Financial Analyst": "Services", "Legal Advisor": "Services", "Bank Teller": "Services",
    "Operations Manager": "Services", "Procurement Officer": "Industrial", "Administrative Assistant": "Services",
    "Receptionist": "Services", "Marketing Specialist": "Services", "Content Creator": "Tech",
    "Logistics Coordinator": "Industrial", "Supply Chain Manager": "Industrial"
}

# Nationality and Risk Mapping
ALL_NATIONALITIES = [
    "Bangladesh", "India", "Pakistan", "Nepal", "Sri Lanka", "Indonesia",
    "Philippines", "Yemen", "Egypt", "Sudan", "Syria", "Jordan", "Lebanon",
    "Morocco", "Palestine", "Tunisia", "Libya", "Mauritania", "Oman",
    "Qatar", "Bahrain", "Kuwait", "Ethiopia", "Eritrea", "Kenya", "Nigeria",
    "Ghana", "Cameroon", "Uganda", "Senegal", "China", "Malaysia", "Thailand",
    "Myanmar", "Cambodia", "Japan", "South Korea", "USA", "UK", "Canada",
    "Australia", "Germany", "France", "Italy", "Spain", "Brazil", "Argentina",
    "Iran", "South Sudan", "Mozambique", "Tanzania" # Ensure high-risk ones are also selectable
]

HIGH_RISK_NATIONALITIES = [
    "Iran", "Myanmar", "Yemen", "Syria", "Lebanon", "Sudan", "South Sudan",
    "Nigeria", "Cameroon", "Senegal", "Mozambique", "Tanzania", "Philippines", "Ethiopia"
]

def is_high_risk_nationality_func(nationality):
    return nationality in HIGH_RISK_NATIONALITIES

# Country of Residence and Risk Mapping
ALL_COUNTRIES_OF_RESIDENCE = [
    "KSA", "UAE", "Egypt",
    "Iran", "Myanmar", "Yemen", "Syria", "Lebanon", "Sudan", "South Sudan",
    "Nigeria", "Cameroon", "Senegal", "Mozambique", "Tanzania", "Philippines", "Ethiopia" # Ensure high-risk ones are also selectable
]

HIGH_RISK_COUNTRIES_OF_RESIDENCE = [
    "Iran", "Myanmar", "Yemen", "Syria", "Lebanon", "Sudan", "South Sudan",
    "Nigeria", "Cameroon", "Senegal", "Mozambique", "Tanzania", "Philippines", "Ethiopia"
]

def is_high_risk_country_func(country_residence):
    return country_residence in HIGH_RISK_COUNTRIES_OF_RESIDENCE

# Other Options
# Combine sectors from JOB_SECTOR_MAP values and the generate_sector function's choices
# This ensures all possible sectors are available, even if not all are in the loaded 'borrowers' data.
ALL_SECTORS = sorted(list(set(JOB_SECTOR_MAP.values()) | set(["Retail", "Services", "Tech", "Industrial"])))
ALL_REGIONS = ["Central", "Eastern", "Western", "Southern"]
ALL_INCOME_BRACKETS = ["Low", "Medium", "High"]
ALL_GENDERS = ["Male", "Female"]


# === Helper Functions (adapted for Streamlit) ===
def get_all_borrower_ids():
    return borrowers['borrower_id'].tolist()

def get_loans_for_borrower(borrower_id):
    borrower_loans = loans[loans['borrower_id'] == borrower_id]
    if borrower_loans.empty:
        return pd.DataFrame() # Return empty DataFrame for no loans
    return borrower_loans

def get_borrower_details(borrower_id):
    borrower_info = borrowers[borrowers['borrower_id'] == borrower_id]
    if borrower_info.empty:
        return pd.DataFrame() # Return empty DataFrame for not found
    return borrower_info

def get_payments_for_loan(loan_id):
    loan_payments = payments[payments['loan_id'] == loan_id]
    if loan_payments.empty:
        return pd.DataFrame() # Return empty DataFrame for no payments
    return loan_payments

def get_payment_delay_possibility(loan_id):
    loan_payments = payments[payments['loan_id'] == loan_id].copy()

    if loan_payments.empty:
        return "No payment data to assess delay possibility."

    loan_payments = loan_payments.merge(loans, on="loan_id", how="left").merge(borrowers, on="borrower_id", how="left")

    loan_payments['payment_date'] = pd.to_datetime(loan_payments['payment_date'], errors='coerce')
    loan_payments['due_date'] = pd.to_datetime(loan_payments['due_date'], errors='coerce')
    loan_payments['delay_days'] = (loan_payments['payment_date'] - loan_payments['due_date']).dt.days
    loan_payments['delay_days'] = loan_payments['delay_days'].fillna(0).astype(int)
    loan_payments['payment_delayed'] = (loan_payments['delay_days'] > 2).astype(int)

    # Apply categorical mappings
    loan_payments['sector'] = loan_payments['sector'].map(sector_map).fillna(-1)
    loan_payments['region'] = loan_payments['region'].map(region_map).fillna(-1)
    loan_payments['income_bracket'] = loan_payments['income_bracket'].map(income_bracket_map).fillna(-1)
    loan_payments['gender'] = loan_payments['gender'].map(gender_map).fillna(-1)

    features = ['age', 'is_high_risk_nationality', 'is_high_risk_country',
                'is_high_risk_job', 'principal', 'apr', 'sector',
                'region', 'income_bracket', 'gender']

    if not all(col in loan_payments.columns for col in features):
        missing_cols = [col for col in features if col not in loan_payments.columns]
        return f"Missing required features for prediction: {', '.join(missing_cols)}. " \
               "Ensure your data files contain these columns."

    input_data = loan_payments[features]

    if loaded_model is None:
        return "Model not loaded. Cannot predict delay possibility."

    predictions = loaded_model.predict(input_data)
    probabilities = loaded_model.predict_proba(input_data)

    results = []
    for i, row in input_data.iterrows():
        payment_status = "Delayed" if predictions[i] == 1 else "On-time"
        probability_delayed = probabilities[i][1]
        due_date_display = loan_payments.loc[i, 'due_date'].date() if pd.notna(loan_payments.loc[i, 'due_date']) else "N/A"
        results.append(f"Payment due {due_date_display}: Predicted {payment_status} (Probability of Delay: {probability_delayed:.2f})")

    return "\n".join(results)

def predict_delay(age, nationality, residence, job_title, amount, apr, sector_val, region_val, income_bracket_val, gender_val):
    # Determine risk flags based on selected strings using the defined functions
    is_high_risk_nationality = 1 if is_high_risk_nationality_func(nationality) else 0
    is_high_risk_country = 1 if is_high_risk_country_func(residence) else 0
    is_high_risk_job = 1 if is_high_risk_job_func(job_title) else 0

    input_dict = {
        'age': age,
        'is_high_risk_nationality': is_high_risk_nationality,
        'is_high_risk_country': is_high_risk_country,
        'is_high_risk_job': is_high_risk_job,
        'principal': amount,
        'apr': apr,
        'sector': sector_val,
        'region': region_val,
        'income_bracket': income_bracket_val,
        'gender': gender_val
    }

    input_df = pd.DataFrame([input_dict])

    # Apply categorical mappings to the input DataFrame
    input_df['sector'] = input_df['sector'].map(sector_map).fillna(-1)
    input_df['region'] = input_df['region'].map(region_map).fillna(-1)
    input_df['income_bracket'] = input_df['income_bracket'].map(income_bracket_map).fillna(-1)
    input_df['gender'] = input_df['gender'].map(gender_map).fillna(-1)

    features = ['age', 'is_high_risk_nationality', 'is_high_risk_country',
                'is_high_risk_job', 'principal', 'apr', 'sector',
                'region', 'income_bracket', 'gender']

    if not all(col in input_df.columns for col in features):
        missing_cols = [col for col in features if col not in input_df.columns]
        return f"Missing required features for prediction: {', '.join(missing_cols)}. " \
               "This indicates an issue with input data or mapping."

    if loaded_model is None:
        return "Model not loaded. Cannot predict delay possibility."

    prediction = loaded_model.predict(input_df[features])
    probability = loaded_model.predict_proba(input_df[features])

    status = "Delayed" if prediction[0] == 1 else "On-time"
    prob_delayed = probability[0][1]

    return f"Prediction: {status} (Probability of Delay: {prob_delayed:.2f})"

# === Streamlit UI Layout ===
st.set_page_config(page_title="Loan Management Dashboard", layout="wide")

# Header Section with Logo and Title
col1, col2 = st.columns([1, 3])
with col1:
    # Placeholder for company logo. You can replace this with st.image("your_logo.png")
    st.image("https://www.forusinvest.com/wp-content/uploads/2019/03/logo-desktop.png", width=100)
with col2:
    st.markdown("## Loan Management & Delay Prediction System")

# Main Tabs for different functionalities
tab1, tab2 = st.tabs(["Data Explorer", "Single Loan Prediction"])

with tab1:
    st.header("Data Explorer")
    subtab1_1, subtab1_2 = st.tabs(["Borrower/Loan Overview", "Payment Details & Delay Possibility"])

    with subtab1_1:
        st.subheader("Borrower/Loan Overview")
        borrower_ids = get_all_borrower_ids()
        selected_borrower_id = st.selectbox("Select Borrower ID", options=borrower_ids, key="borrower_select_tab1")

        if selected_borrower_id:
            st.markdown("#### Borrower Details")
            borrower_details_df = get_borrower_details(selected_borrower_id)
            if not borrower_details_df.empty:
                st.dataframe(borrower_details_df, use_container_width=True)
            else:
                st.write("Borrower not found.")

            st.markdown("#### Loans for this Borrower")
            borrower_loans_df = get_loans_for_borrower(selected_borrower_id)
            if not borrower_loans_df.empty:
                st.dataframe(borrower_loans_df, use_container_width=True)
            else:
                st.write("No loans found for this borrower.")

    with subtab1_2:
        st.subheader("Payment Details & Delay Possibility")
        loan_ids = loans['loan_id'].tolist()
        selected_loan_id = st.selectbox("Select Loan ID", options=loan_ids, key="loan_select_tab1")

        if selected_loan_id:
            st.markdown("#### Payments for this Loan")
            loan_payments_df = get_payments_for_loan(selected_loan_id)
            if not loan_payments_df.empty:
                st.dataframe(loan_payments_df, use_container_width=True)
            else:
                st.write("No payments found for this loan.")

            st.markdown("#### Payment Delay Possibility")
            delay_possibility_text = get_payment_delay_possibility(selected_loan_id)
            st.text_area("Prediction for each payment:", value=delay_possibility_text, height=150)


with tab2:
    st.header("Single Loan Prediction")
    st.markdown("### Enter details to predict likelihood of payment delay")

    # Use comprehensive lists for dropdowns
    sector_options = ALL_SECTORS
    region_options = ALL_REGIONS
    income_bracket_options = ALL_INCOME_BRACKETS
    gender_options = ALL_GENDERS
    nationality_options = ALL_NATIONALITIES
    residence_options = ALL_COUNTRIES_OF_RESIDENCE
    job_title_options = ALL_JOB_TITLES

    col_pred1, col_pred2 = st.columns(2)

    with col_pred1:
        age = st.slider("Age", min_value=20, max_value=60, value=35)
        nationality_input = st.selectbox("Nationality", options=nationality_options, key="nationality_pred")
        residence_input = st.selectbox("Country of Residence", options=residence_options, key="residence_pred")
        job_input = st.selectbox("Job Title", options=job_title_options, key="job_pred")

    with col_pred2:
        amount = st.number_input("Principal Loan Amount (SAR)", min_value=0.0, value=1000.0, step=100.0)
        apr = st.number_input("APR (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
        sector_input = st.selectbox("Sector", options=sector_options, key="sector_pred")
        region_input = st.selectbox("Region", options=region_options, key="region_pred")
        income_bracket_input = st.selectbox("Income Bracket", options=income_bracket_options, key="income_bracket_pred")
        gender_input = st.selectbox("Gender", options=gender_options, key="gender_pred")

    if st.button("Predict Delay"):
        prediction_result = predict_delay(
            age, nationality_input, residence_input, job_input, amount, apr,
            sector_input, region_input, income_bracket_input, gender_input
        )
        st.write(prediction_result)

# Instructions for running the Streamlit app locally:
# 1. Save the code above as a Python file (e.g., `app.py`).
# 2. Save the `requirements.txt` file (as described at the top of this file) in the same directory.
# 3. Place your `borrowers_data.csv`, `loans_data.csv`, `payments_data.csv`,
#    and `random_forest_model.pkl` files in the same directory as `app.py`.
# 4. Open your terminal or command prompt, navigate to this directory.
# 5. Install dependencies: `pip install -r requirements.txt`
# 6. Run the app: `streamlit run app.py`
