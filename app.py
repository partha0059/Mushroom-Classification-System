"""
Mushroom Classification Web Application
A professional Streamlit app for predicting whether mushrooms are edible or poisonous
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Mushroom Classification System",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional dark glassmorphism UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #2d1b4e 100%);
    }
    
    /* Header styling */
    .main-header {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.7);
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    /* Input fields */
    .stNumberInput label, .stSelectbox label {
        color: #a8b2d1 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    .stNumberInput input, .stSelectbox select {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        padding: 0.75rem !important;
    }
    
    .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25) !important;
    }
    
    /* Buttons */
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Success/Error boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(10, 14, 39, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #667eea !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Animated gradient text */
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .animated-gradient {
        background: linear-gradient(270deg, #667eea, #764ba2, #f093fb, #667eea);
        background-size: 800% 800%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 5s ease infinite;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model from pickle file"""
    try:
        with open('mushroom_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please run train_model.py first to generate the model.")
        return None

@st.cache_data
def load_data():
    """Load the dataset for analysis"""
    try:
        df = pd.read_csv('mushroom_classification.csv')
        return df
    except FileNotFoundError:
        return None

def predict_mushroom(model_data, input_data):
    """Make prediction using the trained model with proper scaling"""
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    
    # Create DataFrame with feature names
    input_df = pd.DataFrame([input_data], columns=features)
    
    # CRITICAL: Scale the input data using the same scaler used during training
    input_scaled = scaler.transform(input_df)
    
    # Make prediction on scaled data
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    return prediction, probability

def create_probability_gauge(probability, prediction):
    """Create a gauge chart for prediction probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability[prediction] * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence", 'font': {'size': 24, 'color': '#a8b2d1'}},
        number={'suffix': "%", 'font': {'size': 40, 'color': '#667eea'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#667eea"},
            'bar': {'color': "#667eea" if prediction == 0 else "#ff6b9d"},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.2)",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255, 107, 157, 0.3)'},
                {'range': [50, 75], 'color': 'rgba(255, 193, 7, 0.3)'},
                {'range': [75, 100], 'color': 'rgba(102, 126, 234, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#a8b2d1", 'family': "Inter"},
        height=300
    )
    
    return fig

def create_feature_importance_chart(model_data):
    """Create feature importance bar chart"""
    importances = model_data['feature_importances']
    features = list(importances.keys())
    values = list(importances.values())
    
    fig = go.Figure([go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker=dict(
            color=values,
            colorscale='Purples',
            line=dict(color='rgba(255,255,255,0.2)', width=1)
        ),
        text=[f'{v:.3f}' for v in values],
        textposition='auto',
    )])
    
    fig.update_layout(
        title={'text': 'Feature Importance', 'font': {'size': 20, 'color': '#667eea'}},
        xaxis={'title': 'Importance Score', 'gridcolor': 'rgba(255,255,255,0.1)'},
        yaxis={'title': '', 'gridcolor': 'rgba(255,255,255,0.1)'},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.03)',
        font={'color': "#a8b2d1", 'family': "Inter"},
        height=400
    )
    
    return fig

def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🍄 Mushroom Classification System</h1>
            <p>Advanced AI-Powered Edibility Prediction</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model_data = load_model()
    
    if model_data is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Navigation")
        page = st.radio("Select a page:", ["🔮 Prediction", "📊 Data Analysis", "ℹ️ About"], label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("### 📊 Model Architecture")
        st.markdown("""
        **Algorithm:**  
        Gradient Boosting Classifier
        
        **Accuracy:**  
        98.10%
        
        **Features:**  
        8 Input Features
        
        **Hyperparameters:**
        - Estimators: 200
        - Max Depth: 5
        - Learning Rate: 0.1
        
        **Status:** ✅ Ready
        """)
        
        st.markdown("---")
        st.markdown("### 👨‍💻 Developer Information")
        st.markdown('<p style="font-size: 14px; margin: 0;"><b>Developer:</b> Ayyapparaja</p>', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 14px; margin: 0;"><b>Project:</b> Mushroom Classification System</p>', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 14px; margin: 0;"><b>Email:</b> ayyapparaja227@gmail.com</p>', unsafe_allow_html=True)
    
    if page == "🔮 Prediction":
        st.markdown("## Make a Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🍄 Cap Characteristics")
            cap_diameter = st.number_input("Cap Diameter", min_value=0, max_value=3000, value=1000, step=10,
                                          help="Diameter of the mushroom cap (range: 0-3000)")
            cap_shape = st.number_input("Cap Shape (Encoded)", min_value=0, max_value=10, value=2, step=1,
                                        help="Common values: 2, 4, 6")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🌿 Gill Characteristics")
            gill_attachment = st.number_input("Gill Attachment (Encoded)", min_value=0, max_value=5, value=2, step=1,
                                             help="Common values: 0, 2")
            gill_color = st.number_input("Gill Color (Encoded)", min_value=0, max_value=20, value=10, step=1,
                                        help="Encoded color value (range: 0-20)")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🌱 Stem Characteristics")
            stem_height = st.number_input("Stem Height", min_value=0.0, max_value=5.0, value=2.0, step=0.1,
                                         help="Height of the stem (range: 0-5)")
            stem_width = st.number_input("Stem Width", min_value=0, max_value=3000, value=1500, step=10,
                                        help="Width of the stem (range: 0-3000)")
            stem_color = st.number_input("Stem Color (Encoded)", min_value=0, max_value=20, value=11, step=1,
                                        help="Encoded color value (range: 0-20)")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🌍 Environmental")
            season = st.number_input("Season (Encoded)", min_value=0.0, max_value=2.0, value=0.943, step=0.001, format="%.3f",
                                    help="Common values: ~0.888, ~0.943, ~1.804")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🔮 Predict Mushroom Type", use_container_width=True):
            input_data = [cap_diameter, cap_shape, gill_attachment, gill_color, 
                         stem_height, stem_width, stem_color, season]
            
            prediction, probability = predict_mushroom(model_data, input_data)
            
            # Results
            st.markdown("---")
            st.markdown("## 📊 Prediction Results")
            
            result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
            
            with result_col2:
                if prediction == 0:
                    st.success("### ✅ EDIBLE MUSHROOM")
                    st.markdown(f"**Confidence:** {probability[0]*100:.2f}%")
                    st.markdown("This mushroom is predicted to be **safe to eat**.")
                else:
                    st.error("### ⚠️ POISONOUS MUSHROOM")
                    st.markdown(f"**Confidence:** {probability[1]*100:.2f}%")
                    st.markdown("This mushroom is predicted to be **dangerous**. Do NOT consume!")
                
                # Probability gauge
                fig = create_probability_gauge(probability, prediction)
                st.plotly_chart(fig, use_container_width=True)
            
            # Probability breakdown
            st.markdown("### Probability Breakdown")
            prob_col1, prob_col2 = st.columns(2)
            with prob_col1:
                st.metric("Edible Probability", f"{probability[0]*100:.2f}%")
            with prob_col2:
                st.metric("Poisonous Probability", f"{probability[1]*100:.2f}%")
    
    elif page == "📊 Data Analysis":
        st.markdown("## Data Analysis Dashboard")
        
        df = load_data()
        if df is not None:
            tab1, tab2, tab3 = st.tabs(["📈 Overview", "🎯 Features", "📊 Distribution"])
            
            with tab1:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Samples", f"{len(df):,}")
                with col2:
                    st.metric("Features", len(model_data['features']))
                with col3:
                    edible_count = (df['class'] == 0).sum()
                    st.metric("Edible", f"{edible_count:,}")
                with col4:
                    poisonous_count = (df['class'] == 1).sum()
                    st.metric("Poisonous", f"{poisonous_count:,}")
                
                st.markdown("### Dataset Preview")
                st.dataframe(df.head(10), use_container_width=True)
            
            with tab2:
                st.markdown("### Feature Importance Analysis")
                fig = create_feature_importance_chart(model_data)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.markdown("### Class Distribution")
                class_counts = df['class'].value_counts()
                fig = go.Figure(data=[go.Pie(
                    labels=['Poisonous', 'Edible'],
                    values=[class_counts[1], class_counts[0]],
                    hole=0.4,
                    marker=dict(colors=['#ff6b9d', '#667eea']),
                    textfont=dict(size=16, color='white')
                )])
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#a8b2d1", 'family': "Inter"},
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
    
    else:  # About page
        st.markdown("## About This Project")
        
        st.markdown("""
        ### 🍄 Mushroom Classification System
        
        This application uses advanced machine learning to predict whether mushrooms
        are edible or poisonous based on their physical characteristics.
        
        #### 🎯 Key Features
        - **Gradient Boosting Algorithm**: State-of-the-art ensemble learning method
        - **High Accuracy**: 98.10% test accuracy with proper feature scaling
        - **Real-time Predictions**: Instant classification results
        - **Interactive Visualizations**: Comprehensive data analysis tools
        
        #### 📊 Model Performance
        The model has been optimized with the following parameters:
        - **Accuracy**: 98.10% on test set (13,509 samples)
        - **Learning Rate**: 0.1
        - **Number of Estimators**: 200
        - **Max Depth**: 5
        - **Subsample**: 0.8
        - **Feature Scaling**: StandardScaler (critical for accuracy)
        
        #### 🔬 Training Details
        - **Total Samples**: 54,035 mushrooms
        - **Training Set**: 40,526 samples (75%)
        - **Test Set**: 13,509 samples (25%)
        - **Class Balance**: 45% Edible | 55% Poisonous
        - **False Negatives**: Only 0.98% (132 out of 13,509) - minimized for safety!
        
        #### ⚠️ Important Safety Notice
        This is a predictive model for **educational and demonstration purposes only**. 
        
        **NEVER consume wild mushrooms based solely on this prediction!** Always consult with:
        - Certified mycology experts
        - Field guides with multiple identification methods
        - Professional foragers with extensive experience
        
        Some poisonous mushrooms can be **fatal**. The 1.9% error rate means this model gets 
        256 predictions wrong out of 13,509 test cases. Your life is worth more than a meal!
        
        #### 👨‍💻 Developer
        **Ayyapparaja**  
        Data Scientist & ML Engineer
        
        ---
        
        *Built with Streamlit, scikit-learn, and Plotly | College Project 2026*
        """)
    
    # Footer at the bottom of all pages
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 14px; color: #888; margin-top: 40px; padding: 20px;">Created by Ayyapparaja</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
