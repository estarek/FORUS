import streamlit as st
import pandas as pd
from PIL import Image # PIL is used for image handling, though not directly for the logo upload in the Streamlit version
import joblib
import os

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

# --- Global Mappings for Categorical Features ---
# These mappings are crucial and MUST be consistent with how the model was trained.
# We are assuming 'borrowers' DataFrame contains the necessary columns for creating these maps.
# If your model was trained using a different DataFrame (e.g., a combined one),
# you should load that specific DataFrame here to derive the mappings.
try:
    # Ensure these columns exist in your borrowers_data.csv
    sector_map = {k: v for v, k in enumerate(borrowers['sector'].astype('category').cat.categories)}
    region_map = {k: v for v, k in enumerate(borrowers['region'].astype('category').cat.categories)}
    income_bracket_map = {k: v for v, k in enumerate(borrowers['income_bracket'].astype('category').cat.categories)}
    gender_map = {k: v for v, k in enumerate(borrowers['gender'].astype('category').cat.categories)}
except KeyError as e:
    st.error(f"Error creating categorical mappings: Column '{e}' not found in 'borrowers_data.csv'. "
             "Please ensure the CSV contains 'sector', 'region', 'income_bracket', and 'gender' columns, "
             "or adjust the mapping source if your model was trained with different data.")
    st.stop()

# === Helper Functions (adapted for Streamlit) ===
# These functions interact with the loaded DataFrames
def get_all_borrower_ids():
    return borrowers['borrower_id'].tolist()

def get_loans_for_borrower(borrower_id):
    borrower_loans = loans[loans['borrower_id'] == borrower_id]
    if borrower_loans.empty:
        return "No loans found for this borrower."
    return borrower_loans.to_string(index=False)

def get_borrower_details(borrower_id):
    borrower_info = borrowers[borrowers['borrower_id'] == borrower_id]
    if borrower_info.empty:
        return "Borrower not found."
    return borrower_info.to_string(index=False)

def get_payments_for_loan(loan_id):
    loan_payments = payments[payments['loan_id'] == loan_id]
    if loan_payments.empty:
        return "No payments found for this loan."
    return loan_payments.to_string(index=False)

def get_payment_delay_possibility(loan_id):
    # Make a copy to avoid SettingWithCopyWarning
    loan_payments = payments[payments['loan_id'] == loan_id].copy()

    if loan_payments.empty:
        return "No payment data to assess delay possibility."

    # Merge with loan and borrower info to get all necessary features
    loan_payments = loan_payments.merge(loans, on="loan_id", how="left").merge(borrowers, on="borrower_id", how="left")

    # Preprocessing steps, consistent with model training
    loan_payments['payment_date'] = pd.to_datetime(loan_payments['payment_date'], errors='coerce')
    loan_payments['due_date'] = pd.to_datetime(loan_payments['due_date'], errors='coerce')
    loan_payments['delay_days'] = (loan_payments['payment_date'] - loan_payments['due_date']).dt.days
    loan_payments['delay_days'] = loan_payments['delay_days'].fillna(0).astype(int)
    # 'payment_delayed' is the target variable, not used for prediction input
    loan_payments['payment_delayed'] = (loan_payments['delay_days'] > 2).astype(int)

    # Apply categorical mappings
    loan_payments['sector'] = loan_payments['sector'].map(sector_map).fillna(-1)
    loan_payments['region'] = loan_payments['region'].map(region_map).fillna(-1)
    loan_payments['income_bracket'] = loan_payments['income_bracket'].map(income_bracket_map).fillna(-1)
    loan_payments['gender'] = loan_payments['gender'].map(gender_map).fillna(-1)

    # Define features used by the model
    features = ['age', 'is_high_risk_nationality', 'is_high_risk_country',
                'is_high_risk_job', 'principal', 'apr', 'sector',
                'region', 'income_bracket', 'gender']

    # Check if all required features are present in the processed DataFrame
    if not all(col in loan_payments.columns for col in features):
        missing_cols = [col for col in features if col not in loan_payments.columns]
        return f"Missing required features for prediction: {', '.join(missing_cols)}. " \
               "Ensure your data files contain these columns."

    input_data = loan_payments[features]

    if loaded_model is None:
        return "Model not loaded. Cannot predict delay possibility."

    # Get predictions and probabilities for each payment
    predictions = loaded_model.predict(input_data)
    probabilities = loaded_model.predict_proba(input_data)

    results = []
    for i, row in input_data.iterrows():
        payment_status = "Delayed" if predictions[i] == 1 else "On-time"
        probability_delayed = probabilities[i][1]
        # Safely get due_date for display
        due_date_display = loan_payments.loc[i, 'due_date'].date() if pd.notna(loan_payments.loc[i, 'due_date']) else "N/A"
        results.append(f"Payment due {due_date_display}: Predicted {payment_status} (Probability of Delay: {probability_delayed:.2f})")

    return "\n".join(results)

def predict_delay(age, nationality_risk, residence_risk, job_risk, amount, apr, sector_val, region_val, income_bracket_val, gender_val):
    # Convert risk strings to 0/1 for model input
    is_high_risk_nationality = 1 if nationality_risk == "High Risk" else 0
    is_high_risk_country = 1 if residence_risk == "High Risk" else 0
    is_high_risk_job = 1 if job_risk == "High Risk" else 0

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

    # Check if all required features are present after mapping
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
    # Streamlit doesn't have a direct interactive image upload for a logo like Gradio.
    # You can use st.file_uploader to allow users to upload, then st.image to display.
    # For simplicity, using a placeholder image or a static image path.
    # If you have a company logo file (e.g., 'logo.png'), you can use:
    # st.image("logo.png", width=100)
    st.image("https://via.placeholder.com/100x100.png?text=Company+Logo", width=100 ) # Placeholder
with col2:
    st.markdown("## Loan Management & Delay Prediction System")

# Main Tabs for different functionalities
tab1, tab2 = st.tabs(["Data Explorer", "Single Loan Prediction"])

with tab1:
    st.header("Data Explorer")
    # Sub-tabs within Data Explorer
    subtab1_1, subtab1_2 = st.tabs(["Borrower/Loan Overview", "Payment Details & Delay Possibility"])

    with subtab1_1:
        st.subheader("Borrower/Loan Overview")
        borrower_ids = get_all_borrower_ids()
        # Use a unique key for each selectbox to prevent issues with multiple widgets
        selected_borrower_id = st.selectbox("Select Borrower ID", options=borrower_ids, key="borrower_select_tab1")

        if selected_borrower_id:
            st.text_area("Borrower Details", value=get_borrower_details(selected_borrower_id), height=150)
            st.text_area("Loans for this Borrower", value=get_loans_for_borrower(selected_borrower_id), height=150)

    with subtab1_2:
        st.subheader("Payment Details & Delay Possibility")
        loan_ids = loans['loan_id'].tolist()
        selected_loan_id = st.selectbox("Select Loan ID", options=loan_ids, key="loan_select_tab1")

        if selected_loan_id:
            st.text_area("Payments for this Loan", value=get_payments_for_loan(selected_loan_id), height=150)
            st.text_area("Payment Delay Possibility", value=get_payment_delay_possibility(selected_loan_id), height=150)

with tab2:
    st.header("Single Loan Prediction")
    st.markdown("### Enter details to predict likelihood of payment delay")

    # Options for risk-related dropdowns (simplified for demonstration)
    risk_options = ["Not High Risk", "High Risk"]

    # Get actual categories from the loaded data for other dropdowns
    sector_options = list(sector_map.keys())
    region_options = list(region_map.keys())
    income_bracket_options = list(income_bracket_map.keys())
    gender_options = list(gender_map.keys())

    # Use columns for a two-column layout for prediction inputs
    col_pred1, col_pred2 = st.columns(2)

    with col_pred1:
        age = st.slider("Age", min_value=20, max_value=60, value=35)
        # Using simplified risk options for these inputs
        nationality_input = st.selectbox("Nationality Risk", options=risk_options, key="nationality_risk_pred")
        residence_input = st.selectbox("Country of Residence Risk", options=risk_options, key="residence_risk_pred")
        job_input = st.selectbox("Job Title Risk", options=risk_options, key="job_risk_pred")

    with col_pred2:
        amount = st.number_input("Principal Loan Amount (SAR)", min_value=0.0, value=1000.0, step=100.0)
        apr = st.number_input("APR (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
        sector_input = st.selectbox("Sector", options=sector_options, key="sector_pred")
        region_input = st.selectbox("Region", options=region_options, key="region_pred")
        income_bracket_input = st.selectbox("Income Bracket", options=income_bracket_options, key="income_bracket_pred")
        gender_input = st.selectbox("Gender", options=gender_options, key="gender_pred")

    # Button to trigger prediction
    if st.button("Predict Delay"):
        prediction_result = predict_delay(
            age, nationality_input, residence_input, job_input, amount, apr,
            sector_input, region_input, income_bracket_input, gender_input
        )
        st.write(prediction_result)

# Instructions for running the Streamlit app:
# 1. Save the code above as a Python file (e.g., `app.py`).
# 2. Make sure you have `streamlit`, `pandas`, `scikit-learn`, and `Pillow` installed:
#    `pip install streamlit pandas scikit-learn Pillow`
# 3. Place your `borrowers_data.csv`, `loans_data.csv`, `payments_data.csv`,
#    and `random_forest_model.pkl` files in the same directory as `app.py`.
# 4. Run the app from your terminal: `streamlit run app.py`
