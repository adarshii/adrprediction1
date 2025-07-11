# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load model and preprocessing components
@st.cache_resource
def load_components():
    model = joblib.load('semaglutide_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    top_reactions = joblib.load('top_reactions.pkl')
    return model, preprocessor, top_reactions

model, preprocessor, TOP_REACTIONS = load_components()

# App title and description
st.title("Semaglutide ADR Risk Assessment")
st.markdown("""
**Clinical Decision Support Tool**  
Predict serious adverse drug reaction risk for patients prescribed Semaglutide (Ozempic, Wegovy)
""")

# Patient information form
with st.form("patient_form"):
    st.header("Patient Information")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=120, value=58)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=300, value=85)
        sex = st.selectbox("Sex", ["Male", "Female", "Unknown"])
    
    with col2:
        indication = st.selectbox("Reason for Semaglutide", 
                                 ["Diabetes", "Weight Loss", "Other"])
        country = st.selectbox("Country", 
                              ["US", "UK", "CA", "AU", "DE", "FR", "JP", "Other"])
        reactions = st.text_input("Observed Reactions (comma separated)", 
                                 "Nausea, Vomiting")
    
    submitted = st.form_submit_button("Assess ADR Risk")

# Prediction and explanation
if submitted:
    # Process inputs
    reaction_list = [r.strip().lower() for r in reactions.split(",")]
    
    # Create feature dictionary
    features = {
        'age': age,
        'wt': weight,
        'sex': sex,
        'country': country,
        'indication': indication,
    }
    
    # Add reaction flags
    for r in TOP_REACTIONS:
        features[f'react_{r}'] = 1 if r in reaction_list else 0
    
    # Convert to DataFrame for processing
    features_df = pd.DataFrame([features])
    
    # Preprocess and predict
    processed = preprocessor.transform(features_df)
    probability = model.predict_proba(processed)[0][1]
    
    # Risk category
    if probability < 0.3:
        risk_category = "Low Risk"
        color = "green"
    elif probability < 0.7:
        risk_category = "Medium Risk"
        color = "orange"
    else:
        risk_category = "High Risk"
        color = "red"
    
    # Display results
    st.subheader("Risk Assessment")
    st.markdown(f"### <span style='color:{color}; font-size: 24px;'>{risk_category}</span>", 
                unsafe_allow_html=True)
    st.progress(probability)
    st.markdown(f"**Probability of Serious ADR:** {probability:.1%}")
    
    # SHAP explanation
    st.subheader("Risk Factors Breakdown")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(processed)
    
    # Top risk factors
    feature_names = preprocessor.get_feature_names_out()
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': shap_values.values[0]
    }).sort_values('SHAP Value', ascending=False)
    
    # Filter top 10 features that contribute to risk
    top_risk_factors = shap_df[shap_df['SHAP Value'] > 0].head(10)
    
    if not top_risk_factors.empty:
        st.write("Top contributing factors to serious ADR risk:")
        for _, row in top_risk_factors.iterrows():
            feature = row['Feature'].replace('cat__', '').replace('react_', '')
            st.markdown(f"- **{feature}**: +{row['SHAP Value']:.2f} risk points")
    else:
        st.info("No significant risk factors identified")
    
    # Clinical recommendations
    st.subheader("Clinical Recommendations")
    if probability > 0.7:
        st.warning("""
        - **Monitor closely** for serious adverse reactions
        - Consider weekly follow-ups for first month
        - Educate patient on warning signs (pancreatitis symptoms, renal impairment)
        - Baseline pancreatic enzymes and renal function tests
        """)
    elif probability > 0.3:
        st.info("""
        - Standard monitoring protocol
        - Educate patient on common side effects
        - Schedule 2-week follow-up
        - Consider dose titration schedule
        """)
    else:
        st.success("""
        - Routine monitoring appropriate
        - Provide standard patient education
        - Schedule monthly follow-ups
        """)
    
    # SHAP visualization
    st.subheader("Risk Factor Analysis")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)

# Additional resources section
st.sidebar.header("Clinical Resources")
st.sidebar.markdown("""
- [Semaglutide Prescribing Information](https://www.novo-pi.com/ozempic.pdf)
- [FDA Adverse Event Reporting System](https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers)
- [ADR Management Guidelines](https://www.ncbi.nlm.nih.gov/books/NBK574518/)
""")

st.sidebar.header("About This Tool")
st.sidebar.markdown("""
This predictive model was developed using:
- **42826** Semaglutide cases from FAERS
- XGBoost machine learning algorithm
- Clinical validation against known ADR patterns

**Intended Use:** Clinical decision support only
""")
