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
    page_title="Mushroom Intelligence System",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional dark glassmorphism UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(10, 14, 39) 0%, rgb(20, 25, 50) 90%);
    }
    
    /* Header styling with Glassmorphism */
    .main-header {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    .main-header h1 {
        background: linear-gradient(135deg, #00f260 0%, #0575e6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 0 40px rgba(5, 117, 230, 0.3);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.75);
        font-size: 1.25rem;
        margin-top: 0.75rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    /* Premium Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.8rem;
        margin: 1.2rem 0;
        box-shadow: 0 4px 24px -1px rgba(0, 0, 0, 0.2);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .glass-card:hover {
        transform: translateY(-4px) scale(1.01);
        box-shadow: 0 12px 40px -8px rgba(5, 117, 230, 0.25);
        border-color: rgba(5, 117, 230, 0.3);
        background: rgba(255, 255, 255, 0.06);
    }
    
    .glass-card h4 {
        color: #e0e6ed;
        margin-bottom: 1.2rem;
        font-weight: 600;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 0.5rem;
    }
    
    /* Input fields refinement */
    .stNumberInput label, .stSelectbox label {
        color: #a8b2d1 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.2px;
    }
    
    .stNumberInput input, .stSelectbox select {
        background: rgba(10, 14, 30, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        padding: 0.75rem !important;
        transition: all 0.3s ease;
    }
    
    .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: #0575e6 !important;
        box-shadow: 0 0 0 3px rgba(5, 117, 230, 0.15) !important;
        background: rgba(10, 14, 30, 0.8) !important;
    }
    
    /* Buttons */
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #00f260 0%, #0575e6 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.85rem 2rem;
        border: none;
        border-radius: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(5, 117, 230, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(5, 117, 230, 0.5);
    }
    
    /* Success/Error boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00f260 0%, #0575e6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a8b2d1;
        font-size: 1rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(10, 14, 39, 0.85);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #0575e6 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.1);
    }
    
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def retrieve_model_assets():
    """Load the trained model from pickle file"""
    try:
        with open('mushroom_model.pkl', 'rb') as f:
            inference_artifacts = pickle.load(f)
        return inference_artifacts
    except FileNotFoundError:
        st.error("⚠️ Model artifacts missing! Please run train_model.py first to generate the model.")
        return None

@st.cache_data
def fetch_dataset():
    """Load the dataset for analysis"""
    try:
        df = pd.read_csv('mushroom_classification.csv')
        return df
    except FileNotFoundError:
        return None

def calculate_risk_score(inference_artifacts, sample_data):
    """Make prediction using the trained model with proper scaling"""
    model = inference_artifacts['model']
    scaler = inference_artifacts['scaler']
    features = inference_artifacts['features']
    
    # Create DataFrame with feature names
    sample_df = pd.DataFrame([sample_data], columns=features)
    
    # CRITICAL: Scale the input data using the same scaler used during training
    sample_scaled = scaler.transform(sample_df)
    
    # Make prediction on scaled data
    risk_prediction = model.predict(sample_scaled)[0]
    risk_probability = model.predict_proba(sample_scaled)[0]
    
    return risk_prediction, risk_probability

def render_confidence_dial(probability, predicted_class):
    """Create a gauge chart for prediction probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability[predicted_class] * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Analysis Confidence", 'font': {'size': 24, 'color': '#a8b2d1'}},
        number={'suffix': "%", 'font': {'size': 40, 'color': '#0575e6'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#0575e6", 'tickfont': {'color': '#a8b2d1'}},
            'bar': {'color': "#00f260" if predicted_class == 0 else "#ff5252"},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.1)",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255, 82, 82, 0.2)'},
                {'range': [50, 75], 'color': 'rgba(255, 193, 7, 0.2)'},
                {'range': [75, 100], 'color': 'rgba(0, 242, 96, 0.2)'}
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
        font={'color': "#a8b2d1", 'family': "Outfit"},
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def render_feature_impact(inference_artifacts):
    """Create feature importance bar chart"""
    importances = inference_artifacts['feature_importances']
    feature_list = list(importances.keys())
    values = list(importances.values())
    
    fig = go.Figure([go.Bar(
        x=values,
        y=feature_list,
        orientation='h',
        marker=dict(
            color=values,
            colorscale='Viridis',
            line=dict(color='rgba(255,255,255,0.1)', width=1)
        ),
        text=[f'{v:.3f}' for v in values],
        textposition='auto',
    )])
    
    fig.update_layout(
        title={'text': 'Feature Impact Analysis', 'font': {'size': 20, 'color': '#0575e6'}},
        xaxis={'title': 'Impact Score', 'gridcolor': 'rgba(255,255,255,0.05)'},
        yaxis={'title': '', 'gridcolor': 'rgba(255,255,255,0.05)'},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.02)',
        font={'color': "#a8b2d1", 'family': "Outfit"},
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🍄 Advanced Mycology Intelligence</h1>
            <p>Next-Generation Edibility Prediction System</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    inference_artifacts = retrieve_model_assets()
    
    if inference_artifacts is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 System Navigation")
        app_mode = st.radio("Select Interface:", ["🔮 Risk Assessment", "📊 Dataset Analytics", "ℹ️ System Info"], label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("### 📊 Architecture Spec")
        st.markdown("""
        **Core Engine:**  
        Gradient Boosting v2.0
        
        **Accuracy Rate:**  
        98.10% (Validated)
        
        **Input Vector:**  
        8 Biological Markers
        
        **Hyperparameters:**
        - Estimators: 200
        - Deep Learning Depth: 5
        - Learning Rate: 0.1
        
        **Status:** ✅ Online
        """)
        
        st.markdown("---")
        st.markdown("### 👨‍💻 Engineering Team")
        st.markdown('<p style="font-size: 14px; margin: 0; color: #a8b2d1;"><b>Lead Architect:</b> Partha sarathi R</p>', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 14px; margin: 0; color: #a8b2d1;"><b>System:</b> Mushroom Classification</p>', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 14px; margin: 0; color: #a8b2d1;"><b>Contact:</b> sarathio324@gmail.com</p>', unsafe_allow_html=True)
    
    if app_mode == "🔮 Risk Assessment":
        st.markdown("## Real-time Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🍄 Cap Morphology")
            cap_diameter = st.number_input("Cap Diameter (mm)", min_value=0, max_value=3000, value=1000, step=10,
                                          help="Diameter of the mushroom cap in millimeters")
            cap_shape = st.number_input("Cap Shape Index", min_value=0, max_value=10, value=2, step=1,
                                        help="Morphological classification index (0-10)")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🌿 Gill Structure")
            gill_attachment = st.number_input("Gill Attachment Type", min_value=0, max_value=5, value=2, step=1,
                                             help="Attachment classification (0-5)")
            gill_color = st.number_input("Gill Color Spectrum", min_value=0, max_value=20, value=10, step=1,
                                        help="Color spectrum index (0-20)")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🌱 Stem Metrics")
            stem_height = st.number_input("Stem Height (cm)", min_value=0.0, max_value=5.0, value=2.0, step=0.1,
                                         help="Vertical height of the stem")
            stem_width = st.number_input("Stem Width (mm)", min_value=0, max_value=3000, value=1500, step=10,
                                        help="Cross-sectional width")
            stem_color = st.number_input("Stem Color Index", min_value=0, max_value=20, value=11, step=1,
                                        help="Color index value")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🌍 Ecosystem Data")
            season = st.number_input("Seasonal Index", min_value=0.0, max_value=2.0, value=0.943, step=0.001, format="%.3f",
                                    help="Seasonal climatic coefficient")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🚀 INITIATE ANALYSIS", use_container_width=True):
            sample_vector = [cap_diameter, cap_shape, gill_attachment, gill_color, 
                         stem_height, stem_width, stem_color, season]
            
            risk_class, confidence_score = calculate_risk_score(inference_artifacts, sample_vector)
            
            # Results
            st.markdown("---")
            st.markdown("## 📊 Analysis Report")
            
            result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
            
            with result_col2:
                if risk_class == 0:
                    st.success("### ✅ CONSUMPTION APPROVED")
                    st.markdown(f"**Safety Confidence:** {confidence_score[0]*100:.2f}%")
                    st.markdown("Bio-markers indicate this specimen is **edible**.")
                else:
                    st.error("### ☣️ TOXIC HAZARD DETECTED")
                    st.markdown(f"**Toxicity Confidence:** {confidence_score[1]*100:.2f}%")
                    st.markdown("Bio-markers indicate this specimen is **POISONOUS**. Do not consume.")
                
                # Probability gauge
                fig = render_confidence_dial(confidence_score, risk_class)
                st.plotly_chart(fig, use_container_width=True)
            
            # Probability breakdown
            st.markdown("### Risk Probability Distribution")
            prob_col1, prob_col2 = st.columns(2)
            with prob_col1:
                st.metric("Edible Probability", f"{confidence_score[0]*100:.2f}%")
            with prob_col2:
                st.metric("Toxicity Probability", f"{confidence_score[1]*100:.2f}%")
    
    elif app_mode == "📊 Dataset Analytics":
        st.markdown("## Dataset Analytics Dashboard")
        
        df = fetch_dataset()
        if df is not None:
            tab1, tab2, tab3 = st.tabs(["📈 Data Overview", "🎯 Feature Impact", "📊 Class Logic"])
            
            with tab1:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Samples", f"{len(df):,}")
                with col2:
                    st.metric("Dimensions", len(inference_artifacts['features']))
                with col3:
                    edible_count = (df['class'] == 0).sum()
                    st.metric("Edible Class", f"{edible_count:,}")
                with col4:
                    poisonous_count = (df['class'] == 1).sum()
                    st.metric("Toxic Class", f"{poisonous_count:,}")
                
                st.markdown("### Raw Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
            
            with tab2:
                st.markdown("### Feature Importance Analysis")
                fig = render_feature_impact(inference_artifacts)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.markdown("### Classification Distribution")
                class_counts = df['class'].value_counts()
                fig = go.Figure(data=[go.Pie(
                    labels=['Toxic', 'Edible'],
                    values=[class_counts[1], class_counts[0]],
                    hole=0.4,
                    marker=dict(colors=['#ff5252', '#00f260']),
                    textfont=dict(size=16, color='white')
                )])
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#a8b2d1", 'family': "Outfit"},
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
    
    else:  # About page
        st.markdown("## System Information")
        
        st.markdown("""
        ### 🍄 Advanced Mycology Intelligence System
        
        This application utilizes state-of-the-art machine learning algorithms to perform
        real-time risk assessment of mycological specimens.
        
        #### 🎯 Core Capabilities
        - **Gradient Boosting Engine**: High-performance ensemble learning
        - **Precision Scaling**: 98.10% validation accuracy
        - **Instant Inference**: Real-time classification latency <50ms
        - **Visual Analytics**: Interactive data exploration
        
        #### 📊 Performance Metrics
        - **Validation Accuracy**: 98.10% (N=13,509)
        - **False Negative Rate**: <1.0% (Optimized for Safety)
        - **Training Corpus**: 54,035 vetted samples
        
        #### ⚠️ Safety Protocol
        This AI system is for **educational augmentation only**. 
        
        **CRITICAL WARNING:**
        Never rely solely on AI prediction for consumption decisions.
        - Consult certified professionals
        - Use multiple identification vectors
        - When in doubt, discard
        
        #### 👨‍💻 System Architect
        **Partha sarathi R**  
        AI Systems Engineer
        
        ---
        
        *Powered by Streamlit & Scikit-learn*
        """)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; color: rgba(255,255,255,0.4); padding: 20px; font-size: 14px;">
            <p>Designed & Engineered by Partha sarathi R</p>
            <p style="font-size: 12px;">© 2026 Advanced Mycology Systems</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
