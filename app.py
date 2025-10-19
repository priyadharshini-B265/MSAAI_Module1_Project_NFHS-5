import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Load your trained pipeline
# -----------------------------
with open('malnutrition_model.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# -----------------------------
# Expected Feature Columns (same as training)
# -----------------------------
numeric_features = [
    'Children under age 6 months exclusively breastfed16 (%)',
    'Children under age 3 years breastfed within one hour of birth15 (%)',
    'Total children age 6-23 months receiving an adequate diet16, 17  (%)',
    'Prevalence of diarrhoea in the 2 weeks preceding the survey (Children under age 5 years) (%) ',
    'Children with fever or symptoms of ARI in the 2 weeks preceding the survey taken to a health facility or health provider (Children under age 5 years) (%)  ',
    'Women (age 15-49) who are literate4 (%)',
    'Women (age 15-49)  with 10 or more years of schooling (%)',
    'Women (age 15-49 years) whose Body Mass Index (BMI) is below normal (BMI <18.5 kg/m2)21 (%)',
    'Mothers who had at least 4 antenatal care visits  (for last birth in the 5 years before the survey) (%)',
    'Mothers who consumed iron folic acid for 100 days or more when they were pregnant (for last birth in the 5 years before the survey) (%)',
    'Women age 20-24 years married before age 18 years (%)',
    'Women (age 15-49)  who have ever used the internet (%)',
    'Population living in households with electricity (%)',
    'Population living in households with an improved drinking-water source1 (%)',
    'Population living in households that use an improved sanitation facility2 (%)',
    'Households using clean fuel for cooking3 (%)',
    'Households using iodized salt (%)',
    'Households with any usual member covered under a health insurance/financing scheme (%)',
    'Total Fertility Rate (number of children per woman)',
    'Neonatal mortality rate (per 1000 live births)',
    'Infant mortality rate (per 1000 live births)',
    'Under-five mortality rate (per 1000 live births)',
    'Current Use of Family Planning Methods (Currently Married Women Age 15-49  years) - Any method6 (%)'
]

# -----------------------------
# Target Columns
# -----------------------------
target_cols = [
    'Children under 5 years who are stunted (height-for-age)18 (%)',
    'Children under 5 years who are wasted (weight-for-height)18 (%)',
    'Children under 5 years who are underweight (weight-for-age)18 (%)'
]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Predict Child Malnutrition Indicators (NFHS-5 Data)")

st.write("Upload your Excel file containing socio-economic and health indicators.")

# Upload Excel
uploaded_file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # Select row
    row_index = st.number_input("Select Row Index for Prediction", min_value=0, max_value=len(df)-1, step=1)

    # Extract selected input
    input_df = df[numeric_features].iloc[[row_index]]

    # Predict button
    if st.button("Predict"):
        try:
            predictions = pipeline.predict(input_df)
            prediction_df = pd.DataFrame(predictions, columns=target_cols)

            st.subheader("✅ Predicted Child Malnutrition Indicators")
            st.table(prediction_df)

            st.subheader("📌 Input Data Used")
            st.dataframe(input_df.T.rename(columns={0: "Value"}))


        except Exception as e:
            st.error(f"Error in prediction: {e}")
else:
    st.info("Please upload an Excel file to proceed.")
