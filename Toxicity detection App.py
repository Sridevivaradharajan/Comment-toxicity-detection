import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import io
import os
import gdown
import json
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Toxic Comment Classifier", layout="wide")

# Add CSS styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Navigation Styles */
    .nav-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 30px;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #9370DB 0%, #8A2BE2 100%);
        color: white;
        padding: 60px 40px;
        border-radius: 20px;
        text-align: center;
        margin: 30px 0;
        box-shadow: 0 15px 35px rgba(147, 112, 219, 0.3);
    }
    
    .hero-title {
        font-size: 48px;
        font-weight: 700;
        margin-bottom: 20px;
        text-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
    }
    
    .hero-subtitle {
        font-size: 20px;
        font-weight: 300;
        line-height: 1.6;
        max-width: 800px;
        margin: 0 auto;
        opacity: 0.95;
    }
    
    /* Feature Cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 30px;
        margin: 15px 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.06);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(147, 112, 219, 0.1);
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.12);
    }
    
    .feature-icon {
        font-size: 40px;
        margin-bottom: 15px;
        text-align: center;
    }
    
    .feature-title {
        color: #9370DB;
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 15px;
        text-align: center;
    }
    
    .feature-description {
        color: #666;
        font-size: 16px;
        line-height: 1.6;
        text-align: center;
    }
    
    /* Stats Container */
    .stats-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 40px;
        border-radius: 20px;
        margin: 40px 0;
        text-align: center;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 30px;
        margin-top: 30px;
    }
    
    .stat-item {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 15px;
        padding: 25px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stat-number {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    
    .stat-label {
        font-size: 14px;
        font-weight: 500;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Custom Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #9370DB 0%, #8A2BE2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 500;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(147, 112, 219, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(147, 112, 219, 0.4);
    }
    
    /* Alert Styles */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric Styles */
    .metric-container {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(147, 112, 219, 0.1);
    }
    
    /* Progress Bar */
    .stProgress .st-bo {
        background: linear-gradient(135deg, #9370DB 0%, #8A2BE2 100%);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #f7f7f7 0%, #e8e8e8 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #333;
        font-weight: 600;
    }
    
    h1 {
        border-bottom: 3px solid #9370DB;
        padding-bottom: 10px;
    }
    
    /* Dataframe Styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    }
    
    /* File Uploader */
    .stFileUploader {
        background: white;
        border-radius: 10px;
        border: 2px dashed #9370DB;
        padding: 20px;
    }
    
    /* Text Areas and Inputs */
    .stTextArea > div > div > textarea,
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        font-size: 16px;
        padding: 12px;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus,
    .stTextInput > div > div > input:focus {
        border-color: #9370DB;
        box-shadow: 0 0 0 3px rgba(147, 112, 219, 0.1);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #9370DB 0%, #8A2BE2 100%);
        color: white;
        border-radius: 10px;
        font-weight: 500;
    }
    
    /* Checkbox */
    .stCheckbox {
        font-size: 16px;
    }
    
    /* Custom spacing */
    .section-spacing {
        margin: 40px 0;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 32px;
        }
        
        .hero-subtitle {
            font-size: 18px;
        }
        
        .stats-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .feature-card {
            padding: 20px;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_bilstm_model():
    model_path = "bilstm_model.h5"

    if not os.path.exists(model_path):
        with st.spinner("⬇️ Downloading BiLSTM model... Please wait."):
            # ✅ Use direct download link (replace with your file_id)
            file_id = "1kTfFlFAfCAiUdZO5MgCF0N0uxVNd0ZZn"
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)

    # ✅ return loaded model
    return load_model(model_path)

# Load Tokenizer
@st.cache_resource
def load_tokenizer():
    tokenizer_path = "tokenizer.pkl"

    if not os.path.exists(tokenizer_path):
        with st.spinner("⬇️ Downloading tokenizer... Please wait."):
            file_id = "1psCM-sISb3ToTc6IYhhw3nSLWqaTVAJm"
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, tokenizer_path, quiet=False)

    # Debugging check
    with open(tokenizer_path, "rb") as handle:
        tokenizer = pickle.load(handle)

    return tokenizer

# Load model and tokenizer
model = load_bilstm_model()
tokenizer = load_tokenizer()
MAX_LEN = 122  # Based on preprocessing
THRESHOLD = 0.5  # Threshold for binary classification

if model is None or tokenizer is None:
    st.error("Failed to load model or tokenizer. Please check if the files exist.")
    st.stop()

def preprocess_text(text):
    """Preprocess text for model prediction"""
    if not isinstance(text, str):
        text = str(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    return padded

def predict_toxicity(text, return_probabilities=False):
    """Predict toxicity for a single text"""
    processed = preprocess_text(text)
    prediction = model.predict(processed, verbose=0)[0]
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
   
    if return_probabilities:
        # Return probabilities for internal use - convert to regular Python float
        return dict(zip(labels, [float(p) for p in prediction]))
    else:
        # Return binary predictions (1/0) for user display
        binary_preds = [1 if p > THRESHOLD else 0 for p in prediction]
        return dict(zip(labels, binary_preds))

def create_bar_chart_with_proper_margins(categories, values, title, ylabel, colors, figsize=(10,6)):
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bars
    bars = ax.bar(categories, values, color=colors)
    
    # Set title and labels
    ax.set_title(title, fontsize=16, fontweight='bold', pad=30)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Improve label rotation and positioning
    plt.xticks(rotation=15, ha='center', fontsize=10, wrap=True)
    plt.yticks(fontsize=10)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
               f'{int(value)}' if isinstance(value, (int, float)) and value == int(value) else f'{value:.3f}',
               ha='center', va='bottom', fontsize=9)
    
    # Adjust layout to prevent cutoff
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # Increase bottom margin
    
    return fig

def create_histogram_subplots_with_proper_margins(data_arrays, labels, title, threshold=0.5, figsize=(16, 12)):
    """Create histogram subplots with proper margins"""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']

    for i, (label, data) in enumerate(zip(labels, data_arrays)):
        axes[i].hist(data, bins=10, alpha=0.7, color=colors[i])
        axes[i].set_title(f"{label.replace('_', ' ').title()}", fontsize=12, fontweight='bold', pad=15)
        axes[i].set_xlabel("Probability Score", fontsize=10)
        axes[i].set_ylabel("Frequency", fontsize=10)
        axes[i].axvline(x=threshold, color='red', linestyle='--', alpha=0.7, 
                   label=f'Threshold ({threshold})')
        axes[i].legend(fontsize=9)
        axes[i].tick_params(axis='x', labelsize=9)
        axes[i].tick_params(axis='y', labelsize=9)

    # Proper spacing between subplots
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.25, bottom=0.08, top=0.92, left=0.08, right=0.95)

    return fig

def get_model_metrics():
    try:
        # Helper to format shapes properly
        def format_shape(shape):
            if shape is None:
                return None
            if isinstance(shape, list):
                return [tuple("Batch" if dim is None else dim for dim in s) for s in shape]
            return tuple("Batch" if dim is None else dim for dim in shape)

        # Model info
        model_info = {
            "Model Type": "BiLSTM",
            "Total Parameters": model.count_params(),
            "Trainable Parameters": int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)),
            "Non-trainable Parameters": int(sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)),
            "Input Shape": format_shape(model.input_shape) if hasattr(model, 'input_shape') else None,
            "Output Shape": format_shape(model.output_shape) if hasattr(model, 'output_shape') else None,
            "Number of Layers": len(model.layers),
            "Optimizer": model.optimizer.get_config()['name'] if hasattr(model, 'optimizer') else "Unknown",
            "Loss Function": model.loss if isinstance(model.loss, str) else getattr(model.loss, '__name__', str(model.loss))
        }

        # Layer info
        layer_info = []
        for i, layer in enumerate(model.layers):
            layer_info.append({
                "Layer": i + 1,
                "Name": layer.name,
                "Type": layer.__class__.__name__,
                "Parameters": layer.count_params()
            })

        return model_info, layer_info

    except Exception as e:
        st.error(f"Error extracting model metrics: {e}")
        return {}, []

def evaluate_model_performance():
    """Evaluate model performance on test data and return actual metrics"""
    # Create a more comprehensive test dataset
    test_data = [
        # Clearly toxic comments
        {"text": "You are so stupid and ugly!", "labels": [1, 0, 1, 0, 1, 0]},
        {"text": "Go kill yourself, you worthless piece of trash!", "labels": [1, 1, 0, 1, 1, 0]},
        {"text": "I will find you and hurt you badly!", "labels": [1, 0, 0, 1, 0, 0]},
        {"text": "You're such an idiot, nobody likes you!", "labels": [1, 0, 0, 0, 1, 0]},
        {"text": "I hate all people from that country!", "labels": [1, 0, 0, 0, 0, 1]},
        {"text": "You moron, learn to drive properly!", "labels": [1, 0, 0, 0, 1, 0]},
        {"text": "Shut up you disgusting freak!", "labels": [1, 0, 1, 0, 1, 0]},
        {"text": "I'm going to destroy your pathetic life!", "labels": [1, 1, 0, 1, 1, 0]},
        
        # Non-toxic comments
        {"text": "I love this! Such a great experience.", "labels": [0, 0, 0, 0, 0, 0]},
        {"text": "Wow, you are amazing, keep going!", "labels": [0, 0, 0, 0, 0, 0]},
       {"text": "That's an interesting perspective, thanks for sharing.", "labels": [0, 0, 0, 0, 0, 0]},
        {"text": "The weather is really nice today.", "labels": [0, 0, 0, 0, 0, 0]},
        {"text": "Thank you for your help, much appreciated.", "labels": [0, 0, 0, 0, 0, 0]},
        {"text": "Great job on the presentation!", "labels": [0, 0, 0, 0, 0, 0]},
        {"text": "Could you please help me with this problem?", "labels": [0, 0, 0, 0, 0, 0]},
        {"text": "This is a wonderful community.", "labels": [0, 0, 0, 0, 0, 0]},
        {"text": "What a beautiful sunset today.", "labels": [0, 0, 0, 0, 0, 0]},
        
        # Borderline cases
        {"text": "This is a bad product, waste of money!", "labels": [0, 0, 0, 0, 0, 0]},
        {"text": "This movie was terrible, worst acting ever!", "labels": [0, 0, 0, 0, 0, 0]},
        {"text": "I really don't like this at all.", "labels": [0, 0, 0, 0, 0, 0]}
    ]
    
    # Convert to arrays
    texts = [item["text"] for item in test_data]
    y_true = np.array([item["labels"] for item in test_data])
    
    # Get model predictions
    y_pred_proba = []
    for text in texts:
        processed = preprocess_text(text)
        prediction = model.predict(processed, verbose=0)[0]
        y_pred_proba.append(prediction)
    
    y_pred_proba = np.array(y_pred_proba)
    y_pred_binary = (y_pred_proba > THRESHOLD).astype(int)
    
    # Calculate metrics
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    
    # Overall metrics
    overall_accuracy = accuracy_score(y_true.flatten(), y_pred_binary.flatten())
    
    # Per-class metrics
    class_metrics = {}
    for i, label in enumerate(labels):
        y_true_class = y_true[:, i]
        y_pred_class = y_pred_binary[:, i]
        
        # Calculate metrics for this class
        accuracy = accuracy_score(y_true_class, y_pred_class)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_class, y_pred_class, average='binary', zero_division=0
        )
        
        class_metrics[label] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    # Create detailed classification report
    report = classification_report(
        y_true, y_pred_binary, 
        target_names=labels, 
        output_dict=True, 
        zero_division=0
    )
    
    return {
        'overall_accuracy': overall_accuracy,
        'class_metrics': class_metrics,
        'classification_report': report,
        'test_data': test_data,
        'predictions': y_pred_binary,
        'probabilities': y_pred_proba,
        'true_labels': y_true
    }

# Navigation System
def render_navigation():
    st.markdown('<div class="nav-title">Toxic Comment Detection System</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Home'
    
    # Navigation buttons
    pages = ['Home', 'Live Detection', 'Bulk Analysis', 'Model Insights', 'Test Cases']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    columns = [col1, col2, col3, col4, col5]
    
    for i, page in enumerate(pages):
        with columns[i]:
            if st.button(page, key=f"nav_{page}", use_container_width=True):
                st.session_state.current_page = page
    
    return st.session_state.current_page

# Navigation
current_page = render_navigation()

# ---------------------------
# HOME PAGE
# ---------------------------
if current_page == 'Home':
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">Advanced Toxicity Detection</div>
        <div class="hero-subtitle">
            Powered by cutting-edge BiLSTM neural networks, our AI system provides real-time toxicity detection 
            across multiple categories with high precision and reliability.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Section - Using columns instead of HTML grid
    st.markdown("## ✨ Key Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Real-time Detection</div>
            <div class="feature-description">
                Instant analysis of text content with millisecond response times. 
                Get immediate feedback on toxicity levels across six different categories.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Bulk Predictions</div>
            <div class="feature-description">
                Upload CSV files and process thousands of comments simultaneously. 
                Perfect for content moderation at scale with detailed analytics.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <div class="feature-title">Model Performance</div>
            <div class="feature-description">
                Comprehensive model insights with accuracy metrics, confusion matrices, 
                and detailed performance analysis across all toxicity categories.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧪</div>
            <div class="feature-title">Sample Test Cases</div>
            <div class="feature-description">
                Pre-loaded test comments to explore model behavior and understand 
                classification patterns across different content types.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Getting Started Section
    st.markdown("## 🚀 Get Started")
    
    st.markdown("""
    <div style="text-align: center; max-width: 800px; margin: 30px auto; padding: 30px; background: rgba(255, 255, 255, 0.95); border-radius: 20px; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.06);">
        <p style="color: #666; font-size: 18px; line-height: 1.6; margin-bottom: 30px;">
            Ready to start detecting toxic content? Choose from our powerful tools above to begin your analysis journey.
        </p>
        <div style="display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; margin-top: 25px;">
            <div style="background: linear-gradient(135deg, #9370DB 0%, #8A2BE2 100%); color: white; padding: 15px 25px; border-radius: 25px; font-weight: 500; box-shadow: 0 4px 15px rgba(147, 112, 219, 0.3);">
                ⚡ Live Detection - For single comments
            </div>
            <div style="background: linear-gradient(135deg, #9370DB 0%, #8A2BE2 100%); color: white; padding: 15px 25px; border-radius: 25px; font-weight: 500; box-shadow: 0 4px 15px rgba(147, 112, 219, 0.3);">
                📊 Bulk Analysis - For CSV files
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats Section
    st.markdown("""
    <div class="stats-container">
        <h2 style="margin-bottom: 20px; font-size: 32px;">🚀 System Performance</h2>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-number">122</div>
                <div class="stat-label">Max Sequence Length</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">6</div>
                <div class="stat-label">Toxicity Categories</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">0.5</div>
                <div class="stat-label">Classification Threshold</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">&lt; 100ms</div>
                <div class="stat-label">Response Time</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # How it Works Section - Using Streamlit columns
    st.markdown("## 🔍 How It Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 30px 20px; background: rgba(255, 255, 255, 0.95); border-radius: 15px; margin: 10px 0; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);">
            <div style="font-size: 48px; margin-bottom: 20px;">📝</div>
            <h4 style="color: #9370DB; margin-bottom: 15px;">1. Input Text</h4>
            <p style="color: #666; line-height: 1.6;">Enter your comment or upload a CSV file with multiple comments for analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 30px 20px; background: rgba(255, 255, 255, 0.95); border-radius: 15px; margin: 10px 0; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);">
            <div style="font-size: 48px; margin-bottom: 20px;">⚙️</div>
            <h4 style="color: #9370DB; margin-bottom: 15px;">2. AI Processing</h4>
            <p style="color: #666; line-height: 1.6;">Our BiLSTM model analyzes the text using advanced natural language processing.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 30px 20px; background: rgba(255, 255, 255, 0.95); border-radius: 15px; margin: 10px 0; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);">
            <div style="font-size: 48px; margin-bottom: 20px;">📊</div>
            <h4 style="color: #9370DB; margin-bottom: 15px;">3. Results</h4>
            <p style="color: #666; line-height: 1.6;">Get instant binary classifications (1/0) for six different toxicity categories.</p>
        </div>
        """, unsafe_allow_html=True)

#
