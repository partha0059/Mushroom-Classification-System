# Mushroom Classification System
### Advanced Mycology Intelligence | Partha sarathi R

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview
The **Mushroom Classification System** is an advanced AI-powered web application designed to predict the edibility of mushrooms based on their physical characteristics. Utilizing state-of-the-art machine learning algorithms (Gradient Boosting Classifier), the system analyzes biological markers such as cap shape, gill color, and stem dimensions to assess consumption risk with high precision.

This project uses a modern, glassmorphic user interface to provide a premium user experience while delivering critical safety information.

## 🚀 Key Features

### 🧠 Advanced AI Core
- **Gradient Boosting Engine**: Implements an optimized ensemble learning model for maximum accuracy.
- **Precision Metrics**: Achieves **98.10% validation accuracy** with a false negative rate optimized to <1.0% for safety.
- **Robust Preprocessing**: Includes automated feature scaling and encoding pipelines.

### 🎨 Modern UI/UX
- **Glassmorphism Design**: A sleek, translucent interface with dynamic gradients and blur effects.
- **Interactive Visualizations**: Real-time confidence gauges and feature importance charts using Plotly.
- **Responsive Layout**: Optimized for various screen sizes.

### 📊 Comprehensive Analytics
- **Real-time Inference**: Instant risk assessment (<50ms latency).
- **Data Exploration**: Integrated dashboard for analyzing the underlying dataset.
- **Probability Distribution**: Detailed breakdown of confidence scores for every prediction.

## 🛠️ Technology Stack
- **Frontend**: Streamlit, HTML5, CSS3 (Custom Glassmorphism)
- **Backend/ML**: Python, Scikit-learn, Pandas, NumPy
- **Visualization**: Plotly Express, Plotly Graph Objects
- **Data Persistence**: Pickle (Serialization)

## 📦 Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/partha0059/Mushroom-Classification-System.git
   cd Mushroom-Classification-System
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model** (Required for first run)
   ```bash
   python train_model.py
   ```
   *This will generate the `mushroom_model.pkl` artifact required for inference.*

4. **Launch the Application**
   ```bash
   streamlit run app.py
   ```

## 📖 Usage Guide
1. Launch the app and navigate to the **Risk Assessment** page.
2. Input the mushroom's physical characteristics (Cap Diameter, Gill Color, Stem Height, etc.).
3. Click **INITIATE ANALYSIS** to receive the safety prediction.
4. Review the confidence score and probability distribution before making any decisions.

## 👨‍💻 Developer Information
**Partha sarathi R**  
*AI Systems Engineer & Lead Architect*

- **Contact**: sarathio324@gmail.com
- **GitHub**: [partha0059](https://github.com/partha0059)

---

> **⚠️ SAFETY DISCLAIMER**: This application is for educational and demonstrative purposes only. Never determine the edibility of wild mushrooms solely based on this model. Always consult with certified mycologists or professional foragers.
