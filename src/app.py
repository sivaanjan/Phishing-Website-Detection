import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# --- CONFIGURATION: Must be first command ---
st.set_page_config(
    page_title="🛡️ Enterprise Phishing Intelligence", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS FOR UI POLISH ---
st.markdown("""
    <style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .metric-card { border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; text-align: center; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# Import pipeline
try:
    from model_train import URLFeatureExtractor, ALL_FEATURE_NAMES
except ImportError:
    st.error("Critical Dependency Error: 'model_train.py' not found. Please ensure it is in the same directory.")
    st.stop()

# --- GLOBAL CONFIGURATION ---
MODEL_FILENAME = 'C:\\Users\\Siva\\OneDrive\\Desktop\\phishing\\phishing\\phishing_model.pkl'
METRICS_FILENAME = 'C:\\Users\\Siva\\OneDrive\\Desktop\\phishing\\phishing\\project_metrics.pkl'

# --- UTILITY FUNCTIONS ---
@st.cache_resource
def load_resources():
    try:
        with open(MODEL_FILENAME, 'rb') as file:
            model = pickle.load(file)
        with open(METRICS_FILENAME, 'rb') as f:
            metrics = pickle.load(f)
        
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': ALL_FEATURE_NAMES,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        return model, feature_importance_df, metrics

    except FileNotFoundError:
        st.error(f"System Error: Model file not found. Please execute 'model_train.py' first.")
        st.stop()

# --- MAIN APP LAYOUT ---
def run_app():
    model, feature_importance_df, metrics = load_resources()
    extractor = URLFeatureExtractor()

    # --- HEADER ---
    st.title("🛡️ Enterprise Phishing Intelligence")
    st.markdown("### Advanced Threat Detection System")
    st.markdown("---")

    # --- INPUT SECTION (Wrapped in Form to prevent errors) ---
    with st.form(key='analysis_form'):
        col_input, col_btn = st.columns([4, 1])
        
        with col_input:
            url_input = st.text_input(
                "Target URL for Inspection", 
                placeholder="https://example-suspicious-site.com/login",
                help="Enter the full URL including http/https"
            )
        
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.form_submit_button(label='🔍 Analyze Threat', type="primary", use_container_width=True)

    # --- LOGIC & VISUALIZATION ---
    if submit_button and url_input:
        
        with st.spinner(' performing heuristic analysis and model inference...'):
            try:
                # 1. Feature Extraction
                feature_vector = extractor.get_features(url_input)
                
                # Validation check
                if len(feature_vector) != len(ALL_FEATURE_NAMES):
                    st.error(f"Vector Mismatch: Expected {len(ALL_FEATURE_NAMES)} features, got {len(feature_vector)}.")
                    st.stop()
                
                # 2. Prediction
                input_data = np.array(feature_vector).reshape(1, -1)
                prediction_prob_model = model.predict_proba(input_data)[0][1] 
                
                # 3. Post-Processing (Lexical Penalty)
                critical_lexical_features = ['Sensitive_Word_Count', 'URLURL_Length', 'Punycode_Encoding', 'High_Entropy_URL', 'External_Form_Action']
                feature_to_index = {name: i for i, name in enumerate(ALL_FEATURE_NAMES)}
                critical_indices = [feature_to_index[f] for f in critical_lexical_features]
                num_critical_flags = np.sum(np.array(feature_vector)[critical_indices] == -1)
                
                suspicion_penalty = min(0.75, num_critical_flags * 0.15)
                
                if prediction_prob_model > 0.5:
                    prediction_prob = max(0.01, prediction_prob_model - suspicion_penalty)
                else:
                    prediction_prob = prediction_prob_model 

                prediction = 1 if prediction_prob >= 0.5 else -1
                
                # 4. Score Calculation
                num_suspicious_features = np.sum(np.array(feature_vector) == -1)
                danger_score_raw = int((num_suspicious_features / len(ALL_FEATURE_NAMES)) * 100)
                
                if prediction == 1:
                    danger_score = min(40, int(danger_score_raw * (1 - prediction_prob)))
                else:
                    danger_score = max(60, int(danger_score_raw * (1 + (1 - prediction_prob))))
                    danger_score = min(100, danger_score)
                
                trust_score = 100 - danger_score

                # --- UI: RESULTS SECTION ---
                
                # Top Banner Result
                if prediction == 1:
                    st.success(f"## ✅ LEGITIMATE SITE DETECTED")
                else:
                    st.error(f"## 🚨 PHISHING THREAT DETECTED")

                # Metrics Row
                m1, m2, m3 = st.columns(3)
                m1.metric("Trust Score", f"{trust_score}/100", delta=f"{trust_score-50}", delta_color="normal")
                m2.metric("Threat Probability", f"{(1-prediction_prob):.1%}", delta=f"{danger_score}% Risk", delta_color="inverse")
                m3.metric("Suspicious Indicators", f"{num_suspicious_features} / {len(ALL_FEATURE_NAMES)}")

                # --- TABS FOR DETAILED VIEW ---
                tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🏢 Enterprise Checks", "🔬 Technical Breakdown"])

                # TAB 1: Visualizations
                with tab1:
                    col_chart, col_desc = st.columns([2, 1])
                    
                    with col_chart:
                        # Confidence Scatter Plot
                        current_data = pd.DataFrame({
                            'Confidence': [prediction_prob if prediction == 1 else (1 - prediction_prob)],
                            'Risk Level': [num_suspicious_features],
                            'Type': ['Current Analysis']
                        })
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.scatter(current_data['Risk Level'], current_data['Confidence'], 
                                   c='red' if prediction == -1 else 'green', s=300, edgecolors='black')
                        ax.set_title("Threat Confidence vs. Indicator Count")
                        ax.set_xlabel("Number of Suspicious Indicators")
                        ax.set_ylabel("Model Confidence")
                        ax.set_xlim(0, 15)
                        ax.set_ylim(0, 1.1)
                        ax.grid(True, linestyle='--', alpha=0.5)
                        st.pyplot(fig)
                    
                    with col_desc:
                        st.markdown("#### Model Performance")
                        st.info(f"**Recall:** {metrics['phishing_recall']:.2f}")
                        st.info(f"**Precision:** {metrics['phishing_precision']:.2f}")
                        st.info(f"**Accuracy:** {metrics['overall_accuracy']:.2f}")
                        if suspicion_penalty > 0:
                            st.warning(f"**Heuristic Penalty Applied:** {suspicion_penalty:.2f} (Due to high lexical risk)")

                # TAB 2: Enterprise Security
                with tab2:
                    st.markdown("#### Security Compliance & Heuristics")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    
                    # 1. HSTS Check
                    with c1:
                        if feature_vector[ALL_FEATURE_NAMES.index('SSLfinal_State')] == -1:
                            st.error("🚫 SSL/HSTS Failed")
                            st.caption("Site lacks valid HTTPS/HSTS configuration.")
                        else:
                            st.success("✅ SSL/HSTS Valid")
                            st.caption("Connection is secure.")

                    # 2. Form Integrity
                    with c2:
                        if feature_vector[ALL_FEATURE_NAMES.index('External_Form_Action')] == -1:
                            st.error("🚫 Data Exfiltration Risk")
                            st.caption("Forms submit data to external/insecure domains.")
                        else:
                            st.success("✅ Form Integrity")
                            st.caption("Form actions appear local/safe.")

                    # 3. Entropy
                    with c3:
                        if feature_vector[ALL_FEATURE_NAMES.index('High_Entropy_URL')] == -1:
                            st.error("🚫 High Randomness")
                            st.caption("URL structure implies generated/obfuscated text.")
                        else:
                            st.success("✅ Standard Syntax")
                            st.caption("URL follows standard naming conventions.")

                    # 4. Keywords
                    with c4:
                        if feature_vector[ALL_FEATURE_NAMES.index('Sensitive_Word_Count')] == -1:
                            st.warning("⚠️ Sensitive Keywords")
                            st.caption("Contains 'Login', 'Bank', or 'Verify'.")
                        else:
                            st.success("✅ Clean URL")
                            st.caption("No social engineering keywords found.")

                # TAB 3: Full Data
                with tab3:
                    st.markdown("#### Feature Importance & Status")
                    
                    # Prepare Data
                    feature_df = pd.DataFrame({
                        'Feature': ALL_FEATURE_NAMES,
                        'Value': feature_vector,
                        'Meaning': [
                            "IP Address used as Domain", "URL Length > 75 chars", "Shortening Service Used", 
                            "@ Symbol in URL", "Double Slash Redirect", "Dash in Domain", 
                            "Sub-domain Count", "SSL State", 
                            "Domain Registration Length", "Favicon Source", "Standard Port", 
                            "HTTPS Token in Domain", "Request URL %", 
                            "Anchor URL %", "Links in Tags %", 
                            "Server Form Handler", "Mailto in code", 
                            "Abnormal URL", "Redirect Count", 
                            "OnMouseOver Status", "Right Click Disabled", "Pop-up Window", 
                            "IFrame Usage", "Domain Age", "DNS Record", 
                            "Web Traffic Rank", "Page Rank", "Google Index", 
                            "Links Pointing to Page", "Statistical Report",
                            "Punycode (Homograph)", "External Form Action", "High Entropy", "Sensitive Keywords"
                        ]
                    })
                    feature_df['Status'] = feature_df['Value'].apply(
                        lambda x: 'Phishing (-1)' if x == -1 else ('Suspicious (0)' if x == 0 else 'Legitimate (1)')
                    )
                    
                    # Merge with importance and display
                    feature_df_merged = feature_df.merge(feature_importance_df, on='Feature', how='left')
                    
                    st.dataframe(
                        feature_df_merged[['Feature', 'Meaning', 'Status', 'Importance']].sort_values(by='Importance', ascending=False),
                        use_container_width=True,
                        hide_index=True
                    )

            except Exception as e:
                st.exception(f"Analysis Error: {e}")

    elif submit_button and not url_input:
        st.warning("⚠️ Please enter a URL to proceed.")

if __name__ == '__main__':
    run_app()