import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and metrics
def load_model():
    model = joblib.load('email_engagement_model.joblib')
    metrics = joblib.load('email_engagement_metrics.joblib')
    return model, metrics

model, metrics = load_model()

# Page title and description
st.set_page_config(page_title="Email Engagement Prediction", layout="centered")
st.title('📧 Email Engagement Prediction')
st.markdown("""
    Welcome to the **Email Engagement Prediction App**!  
    Enter the email and user details below to predict the engagement status.
""")

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.8rem;
        border-radius: 10px;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# Input fields with sections
st.header("User and Email Details")
st.markdown("### User Information")
col1, col2 = st.columns(2)
with col1:
    hour = st.number_input('🕒 Hour of Day (0-23)', min_value=0, max_value=23, value=12)
    weekday = st.selectbox('📅 Weekday', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
with col2:
    user_country = st.selectbox('🌍 User Country', ['US', 'UK', 'FR', 'ES'])

st.markdown("### Email Information")
col3, col4 = st.columns(2)
with col3:
    email_text = st.selectbox('✉️ Email Text Type', ['short_email', 'long_email'])
with col4:
    email_version = st.selectbox('📄 Email Version', ['personalized', 'generic'])

# Enhanced prediction display
if st.button('🔮 Predict Engagement'):
    st.markdown("### 📊 Prediction Analysis")
    with st.spinner('Analyzing engagement patterns...'):
        # Create DataFrame for prediction
        X = pd.DataFrame([[hour, weekday, user_country, email_text, email_version]],
                         columns=['hour', 'weekday', 'user_country', 'email_text', 'email_version'])
        
        try:
            # Make prediction
            prediction = model.predict(X)[0]
            
            # Map numeric predictions to engagement status
            status_mapping = {
                0: "Not Opened",
                1: "Opened but Not Clicked",
                2: "Clicked and Opened"
            }
            
            # Convert numeric prediction to status string
            engagement_status = status_mapping[prediction]
            
            # Use Streamlit's native components for better styling
            st.markdown("### 🎯 Predicted Engagement Status")
            
            if engagement_status == "Clicked and Opened":
                st.success(f"### {engagement_status}")
            elif engagement_status == "Opened but Not Clicked":
                st.warning(f"### {engagement_status}")
            else:
                st.error(f"### {engagement_status}")
                
            # Add engagement probability indicator
            st.progress(0.7)  # You can adjust this value based on model confidence
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

        # Display engagement insights
        st.markdown("### 📈 Engagement Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("📊 Key Factors")
            st.markdown(f"""
                - Time: {hour:02d}:00 hrs (Best hours: {', '.join(f"{h:02d}:00" for h in metrics['best_hours'])})
                - Day: {weekday} (Best days: {', '.join(metrics['best_days'])})
                - Region: {user_country} (Top regions: {', '.join(metrics['top_countries'])})
            """)
        
        with col2:
            st.success("💡 Optimization Tips")
            if hour not in metrics['best_hours']:
                st.markdown("- Consider sending during peak hours")
            if weekday not in metrics['best_days']:
                st.markdown("- Try sending on best performing days")
            if user_country not in metrics['top_countries']:
                st.markdown("- Engagement is higher in other regions")

        # Add actual historical performance comparison
        st.markdown("### 📋 Campaign Performance Metrics")
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric(label="Avg. Open Rate", 
                     value=f"{metrics['open_rate']:.1f}%", 
                     delta=f"{metrics['open_rate']-30:.1f}%" if metrics['open_rate'] > 30 else f"{30-metrics['open_rate']:.1f}%")
        with metrics_col2:
            st.metric(label="Click Rate", 
                     value=f"{metrics['click_rate']:.1f}%", 
                     delta=f"{metrics['click_rate']-10:.1f}%" if metrics['click_rate'] > 10 else f"{10-metrics['click_rate']:.1f}%")
        with metrics_col3:
            st.metric(label="Engagement Score", 
                     value=f"{metrics['engagement_score']:.1f}/10", 
                     delta=f"{metrics['engagement_score']-7:.1f}" if metrics['engagement_score'] > 7 else f"{7-metrics['engagement_score']:.1f}")
