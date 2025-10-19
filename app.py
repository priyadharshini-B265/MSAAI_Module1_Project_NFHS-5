import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Load your trained pipeline
# -----------------------------
with open('malnutrition_model.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# -----------------------------
# Numeric features expected
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
# Target columns
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

st.write("""
Enter socio-economic and child/maternal health indicators below to predict:
- Stunting (Height-for-Age)
- Wasting (Weight-for-Height)
- Underweight (Weight-for-Age)
""")

# Collect user input
user_input = {}
for feature in numeric_features:
    user_input[feature] = st.number_input(feature, value=0.0, step=1.0)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Predict button
if st.button("Predict"):
    try:
        predictions = pipeline.predict(input_df)
        prediction_df = pd.DataFrame(predictions, columns=target_cols)
        
        st.subheader("Predicted Child Malnutrition Indicators")
        st.table(prediction_df)
        
    except Exception as e:
        st.error(f"Error in prediction: {e}")
