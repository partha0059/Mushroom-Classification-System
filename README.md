# 🍄 Mushroom Classification System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

A professional machine learning web application that predicts whether mushrooms are edible or poisonous based on their physical characteristics.

## 🎯 Features

- **High Accuracy**: 98.10% prediction accuracy using Gradient Boosting
- **Professional UI**: Dark glassmorphism design with interactive visualizations
- **Real-time Predictions**: Instant classification results
- **Data Analysis Dashboard**: Comprehensive insights and visualizations
- **Educational**: Detailed model information and safety warnings

## 🚀 Live Demo

Visit the live application: [Mushroom Classification System](https://your-app-url.streamlit.app)

## 📊 Model Performance

- **Algorithm**: Gradient Boosting Classifier
- **Accuracy**: 98.10% on test set
- **Training Samples**: 40,526 mushrooms
- **Test Samples**: 13,509 mushrooms
- **Features**: 8 input characteristics
- **False Negatives**: Only 0.98% (minimized for safety)

## 🛠️ Tech Stack

- **Python 3.8+**
- **Streamlit** - Web framework
- **scikit-learn** - Machine learning
- **Plotly** - Interactive visualizations
- **Pandas & NumPy** - Data processing

## 📦 Installation

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ayyapparaja227-arch/Mushroom-Classification-System.git
   cd Mushroom-Classification-System
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the app**
   - Open your browser to `http://localhost:8501`

## ☁️ Streamlit Cloud Deployment

### Quick Deploy

1. **Fork this repository** to your GitHub account

2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**

3. **Deploy the app**:
   - Click "New app"
   - Select your GitHub repository: `ayyapparaja227-arch/Mushroom-Classification-System`
   - Set main file path: `app.py`
   - Click "Deploy"

4. **That's it!** Your app will be live in a few minutes.

### Configuration

The app will automatically:
- Install dependencies from `requirements.txt`
- Load the pre-trained model from `mushroom_model.pkl`
- Use the dataset from `mushroom_classification.csv`

**Note**: The `mushroom_model.pkl` file is included in the repository. If you want to retrain the model, run:
```bash
python train_model.py
```

## 📁 Project Structure

```
Mushroom-Classification-System/
├── app.py                          # Main Streamlit application
├── train_model.py                  # Model training script
├── mushroom_model.pkl              # Pre-trained model (377 KB)
├── mushroom_classification.csv     # Dataset (54,035 samples)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🎮 Usage

### Making Predictions

1. Navigate to the **Prediction** page
2. Enter mushroom characteristics:
   - Cap Diameter, Cap Shape
   - Gill Attachment, Gill Color
   - Stem Height, Stem Width, Stem Color
   - Season
3. Click **"🔮 Predict Mushroom Type"**
4. View results with confidence scores

### Example Test Values

**Poisonous Mushroom** (from dataset):
- Cap Diameter: 1372
- Cap Shape: 2
- Gill Attachment: 2
- Gill Color: 10
- Stem Height: 3.807
- Stem Width: 1545
- Stem Color: 11
- Season: 1.804

**Expected Result**: ⚠️ POISONOUS MUSHROOM

## ⚠️ Important Safety Notice

**THIS IS FOR EDUCATIONAL PURPOSES ONLY!**

- **NEVER** consume wild mushrooms based on this prediction
- The model has a 1.9% error rate
- Some poisonous mushrooms are **FATAL**
- Always consult certified mycology experts
- Use multiple identification methods
- When in doubt, don't eat it!

## 📊 Dataset Information

- **Source**: Kaggle - Prisha Sawhney
- **Total Samples**: 54,035 mushrooms
- **Features**: 8 physical characteristics
- **Classes**: Edible (0) and Poisonous (1)
- **Class Balance**: 45% Edible, 55% Poisonous

## 🔧 Model Details

### Hyperparameters
- Learning Rate: 0.1
- Number of Estimators: 200
- Max Depth: 5
- Subsample: 0.8
- Feature Scaling: StandardScaler

### Feature Importance
1. **stem_width** (23.56%)
2. **gill_attachment** (21.49%)
3. **stem_color** (15.28%)
4. **gill_color** (12.01%)
5. **stem_height** (10.13%)

## 🐛 Troubleshooting

### Common Issues

**Model file not found**:
```bash
python train_model.py
```

**Port already in use**:
```bash
streamlit run app.py --server.port 8502
```

**Dependencies error**:
```bash
pip install --upgrade -r requirements.txt
```

## 👨‍💻 Developer

**Name**: Ayyapparaja  
**Email**: ayyapparaja227@gmail.com  
**Project**: Mushroom Classification System  
**Year**: 2026

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Dataset by Prisha Sawhney (Kaggle)
- Built with Streamlit, scikit-learn, and Plotly
- Inspired by the need for safe mushroom identification

## 📞 Support

For questions or issues:
- Email: ayyapparaja227@gmail.com
- GitHub Issues: [Report a bug](https://github.com/ayyapparaja227-arch/Mushroom-Classification-System/issues)

---

<div align="center">

**🍄 Stay Safe, Classify Smart! 🍄**

Made with ❤️ by Ayyapparaja

</div>
