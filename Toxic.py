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
import json

# ---------------------------
# Load Model and Tokenizer
# ---------------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("bilstm_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_tokenizer():
    try:
        with open("tokenizer.pkl", "rb") as handle:
            tokenizer = pickle.load(handle)
        return tokenizer
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        return None

# Load model and tokenizer
model = load_model()
tokenizer = load_tokenizer()
MAX_LEN = 122  # Based on preprocessing
THRESHOLD = 0.5  # Threshold for binary classification

# Check if model and tokenizer loaded successfully
if model is None or tokenizer is None:
    st.error("Failed to load model or tokenizer. Please check if the files exist.")
    st.stop()

# ---------------------------
# Helper Functions
# ---------------------------
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

def get_model_metrics():
    """Extract model architecture information with professional formatting"""
    try:
        # Get model architecture info with proper formatting
        model_info = {
            "Model Architecture": "Bidirectional LSTM",
            "Total Parameters": f"{model.count_params():,}",
            "Trainable Parameters": f"{sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}",
            "Non-trainable Parameters": f"{sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]):,}",
            "Input Dimensions": f"Batch Size × {model.input_shape[1]} (Sequence Length)",
            "Output Dimensions": f"Batch Size × {model.output_shape[1]} (Categories)",
            "Sequence Length": f"{model.input_shape[1]} tokens",
            "Output Categories": f"{model.output_shape[1]} classes",
            "Optimizer": model.optimizer.__class__.__name__ if hasattr(model, 'optimizer') else "Adam (Default)",
            "Loss Function": "Binary Cross-entropy" if hasattr(model, 'loss') else "Binary Cross-entropy (Default)"
        }
        
        # Get layer information with proper formatting
        layer_info = []
        for i, layer in enumerate(model.layers):
            output_shape = str(layer.output_shape) if hasattr(layer, 'output_shape') else "Variable"
            # Clean up shape display
            if output_shape.startswith("(None,"):
                output_shape = output_shape.replace("(None, ", "Batch × ").replace(")", "")
            
            layer_info.append({
                "Layer #": i + 1,
                "Layer Name": layer.name.replace("_", " ").title(),
                "Layer Type": layer.__class__.__name__,
                "Output Shape": output_shape if output_shape != "Variable" else "Dynamic",
                "Parameters": f"{layer.count_params():,}" if layer.count_params() > 0 else "0",
                "Trainable": "Yes" if layer.trainable else "No"
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

# ---------------------------
# Page Configuration and Styling
# ---------------------------
st.set_page_config(
    page_title="ToxiGuard - AI Toxicity Detection", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛡️"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Background with blur effect */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Add background image with blur */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000"><defs><linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" style="stop-color:%23667eea;stop-opacity:0.1" /><stop offset="100%" style="stop-color:%23764ba2;stop-opacity:0.1" /></linearGradient></defs><rect width="1000" height="1000" fill="url(%23grad1)"/><circle cx="200" cy="200" r="100" fill="rgba(255,255,255,0.05)"/><circle cx="800" cy="300" r="150" fill="rgba(255,255,255,0.03)"/><circle cx="400" cy="700" r="120" fill="rgba(255,255,255,0.04)"/></svg>');
        background-size: cover;
        background-position: center;
        filter: blur(1px);
        z-index: -1;
    }
    
    /* Main container styling */
    .main .block-container {
        padding: 2rem 3rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        margin: 2rem auto;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Header styling */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem !important;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif;
    }
    
    /* Subheader styling */
    h2, h3 {
        color: #2c3e50;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
    }
    
    .css-17eq0hr {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Metric styling */
    .metric-container {
        background: linear-gradient(135deg, #f8f9ff 0%, #e8ecff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #e1e8ff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    /* Table styling */
    .dataframe {
        background: white;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e1e8ff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e1e8ff;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        border-radius: 10px;
        border: 2px dashed #667eea;
        background: linear-gradient(135deg, #f8f9ff 0%, #e8ecff 100%);
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Success message */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
    }
    
    /* Error message */
    .stError {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
    }
    
    /* Info message */
    .stInfo {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        color: #0c5460;
    }
    
    /* Warning message */
    .stWarning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1>🛡️ ToxiGuard AI</h1>
    <p style="font-size: 1.2rem; color: #6c757d;">Advanced Toxicity Detection System</p>
    <p style="color: #6c757d;">Powered by Bidirectional LSTM Neural Networks</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; margin-bottom: 2rem;">
    <h2 style="color: white; margin-bottom: 0.5rem;">🧭 Navigation</h2>
    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">Select a section to explore</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("", ["🔍 Real-time Prediction", "📂 Bulk Prediction", "📊 Model Insights", "🧪 Sample Test Cases"])

# Sidebar model info
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
    <h3 style="color: white; margin-bottom: 1rem;">⚙️ Model Configuration</h3>
    <div style="color: rgba(255,255,255,0.9);">
        <p><strong>Classification Threshold:</strong> {THRESHOLD}</p>
        <p><strong>Max Sequence Length:</strong> {MAX_LEN}</p>
        <p><strong>Model Architecture:</strong> BiLSTM</p>
        <p><strong>Output Categories:</strong> 6</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# 1. Real-time Prediction
# ---------------------------
if page == "🔍 Real-time Prediction":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>🔍 Real-time Comment Analysis</h2>
        <p style="color: #6c757d;">Enter any comment to analyze its toxicity in real-time</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional input section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8f9ff 0%, #e8ecff 100%); 
                padding: 2rem; border-radius: 15px; margin-bottom: 2rem; 
                border: 1px solid #e1e8ff;">
        <h4 style="color: #2c3e50; margin-bottom: 1rem;">💬 Enter Your Comment</h4>
    </div>
    """, unsafe_allow_html=True)
    
    user_input = st.text_area(
        "Type a comment below:", 
        height=120, 
        placeholder="Enter your comment here to analyze for toxicity...",
        help="Enter any text comment to get real-time toxicity analysis"
    )
    
    col1, col2, col3 = st.columns([2, 2, 4])
    with col1:
        predict_button = st.button("🔍 Analyze Comment", type="primary")
    with col2:
        show_probabilities = st.checkbox("📊 Show confidence scores")
    
    if predict_button:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a valid comment to analyze.")
        else:
            with st.spinner("🤖 Analyzing comment for toxicity..."):
                # Get binary predictions
                binary_result = predict_toxicity(user_input, return_probabilities=False)
                # Get probabilities if requested
                prob_result = predict_toxicity(user_input, return_probabilities=True)
            
            # Results section
            st.markdown("""
            <div style="margin: 2rem 0;">
                <h3 style="color: #2c3e50; text-align: center;">🎯 Analysis Results</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Per-Class Performance Table
        st.markdown("""
        <div style="margin: 3rem 0 2rem 0;">
            <h4 style="color: #2c3e50; text-align: center;">🏷️ Per-Category Performance Analysis</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a professional metrics table
        labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        metrics_data = []
        
        for label in labels:
            class_report = evaluation_results['classification_report'][label]
            metrics_data.append({
                'Category': label.replace('_', ' ').title(),
                'Precision': f"{class_report['precision']:.3f}",
                'Recall': f"{class_report['recall']:.3f}",
                'F1-Score': f"{class_report['f1-score']:.3f}",
                'Support': int(class_report['support'])
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Display table with professional styling
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 15px; 
                    border: 1px solid #e1e8ff; margin-bottom: 2rem;">
        """, unsafe_allow_html=True)
        
        st.dataframe(
            metrics_df, 
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Confusion Matrix Visualization
        st.subheader("🎯 Model Performance Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy by category
            categories = [item['Category'] for item in metrics_data]
            accuracies = [evaluation_results['class_metrics'][label]['accuracy'] for label in labels]
            
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            bars = ax1.bar(categories, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3'])
            ax1.set_title("Accuracy by Category")
            ax1.set_ylabel("Accuracy")
            ax1.set_ylim(0, 1)
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig1)
        
        with col2:
            # F1-Score comparison
            f1_scores = [evaluation_results['class_metrics'][label]['f1_score'] for label in labels]
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            bars = ax2.bar(categories, f1_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3'])
            ax2.set_title("F1-Score by Category")
            ax2.set_ylabel("F1-Score")
            ax2.set_ylim(0, 1)
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, f1 in zip(bars, f1_scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{f1:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig2)
        
        # Detailed Test Results
        st.subheader("📋 Detailed Test Results")
        with st.expander("View All Test Predictions vs Ground Truth"):
            test_results = []
            for i, item in enumerate(evaluation_results['test_data']):
                result = {
                    'Comment': item['text'][:50] + "..." if len(item['text']) > 50 else item['text'],
                    'Full_Comment': item['text']
                }
                
                # Add true labels
                for j, label in enumerate(labels):
                    result[f'True_{label}'] = item['labels'][j]
                
                # Add predictions
                for j, label in enumerate(labels):
                    result[f'Pred_{label}'] = evaluation_results['predictions'][i][j]
                
                # Add match indicators
                for j, label in enumerate(labels):
                    result[f'Match_{label}'] = "✅" if item['labels'][j] == evaluation_results['predictions'][i][j] else "❌"
                
                test_results.append(result)
            
            test_df = pd.DataFrame(test_results)
            st.dataframe(test_df.drop('Full_Comment', axis=1), use_container_width=True)
        
        # Model Confidence Analysis
        st.subheader("🎖️ Model Confidence Analysis")
        
        # Calculate confidence metrics
        high_confidence = 0
        medium_confidence = 0
        low_confidence = 0
        
        for i in range(len(evaluation_results['probabilities'])):
            max_prob = np.max(evaluation_results['probabilities'][i])
            if max_prob >= 0.8:
                high_confidence += 1
            elif max_prob >= 0.6:
                medium_confidence += 1
            else:
                low_confidence += 1
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("High Confidence (≥80%)", high_confidence)
        with col2:
            st.metric("Medium Confidence (60-80%)", medium_confidence)
        with col3:
            st.metric("Low Confidence (<60%)", low_confidence)
        
        # Probability Distribution
        st.subheader("📊 Probability Score Distribution")
        fig3, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, label in enumerate(labels):
            prob_scores = evaluation_results['probabilities'][:, i]
            axes[i].hist(prob_scores, bins=10, alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3'][i])
            axes[i].set_title(f"{label.replace('_', ' ').title()} Score Distribution")
            axes[i].set_xlabel("Probability Score")
            axes[i].set_ylabel("Frequency")
            axes[i].axvline(x=THRESHOLD, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({THRESHOLD})')
            axes[i].legend()
        
        plt.tight_layout()
        st.pyplot(fig3)

# ---------------------------
# 4. Sample Test Cases
# ---------------------------
elif page == "🧪 Sample Test Cases":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>🧪 Pre-loaded Test Cases</h2>
        <p style="color: #6c757d;">Test the model with curated examples of toxic and non-toxic comments</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional test section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8f9ff 0%, #e8ecff 100%); 
                padding: 2rem; border-radius: 15px; margin-bottom: 2rem; 
                border: 1px solid #e1e8ff;">
        <h4 style="color: #2c3e50; margin-bottom: 1rem;">📋 Testing Instructions</h4>
        <p style="color: #6c757d; margin: 0;">
            Click on any comment below to see its binary toxicity predictions (1 = Toxic, 0 = Non-toxic).
            You can also analyze all comments at once for a comprehensive overview.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    sample_comments = [
        "You are so stupid and ugly!",
        "I love this! Such a great experience.",
        "I will find you and hurt you.",
        "This is a bad product, waste of money!",
        "Wow, you are amazing, keep going!",
        "Go kill yourself, you worthless piece of trash!",
        "That's an interesting perspective, thanks for sharing.",
        "I hate all people from that religion!",
        "The weather is really nice today.",
        "You're such an idiot, nobody likes you!",
        "This movie was terrible, worst acting ever!",
        "Thank you for your help, much appreciated.",
        "I'm going to destroy your life!",
        "What a beautiful sunset today.",
        "You moron, learn to drive properly!"
    ]
    
    # Option to analyze all at once
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔍 Analyze All Comments", type="primary"):
            st.markdown("""
            <div style="margin: 2rem 0;">
                <h3 style="color: #2c3e50; text-align: center;">📊 Bulk Analysis Results</h3>
            </div>
            """, unsafe_allow_html=True)
            
            all_results = []
            for comment in sample_comments:
                binary_pred = predict_toxicity(comment, return_probabilities=False)
                result = {"Comment": comment[:50] + "..." if len(comment) > 50 else comment}
                result.update(binary_pred)
                result["Total_Toxic_Categories"] = sum(binary_pred.values())
                all_results.append(result)
            
            results_df = pd.DataFrame(all_results)
            
            # Display results with professional styling
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 15px; 
                        border: 1px solid #e1e8ff; margin-bottom: 2rem;">
                <h5 style="color: #2c3e50; margin-bottom: 1rem;">🏷️ Classification Results</h5>
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(results_df, use_container_width=True)
            
            # Summary statistics
            toxic_count = (results_df['toxic'] == 1).sum()
            clean_count = len(sample_comments) - toxic_count
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                            padding: 1.5rem; border-radius: 15px; text-align: center;
                            border: 1px solid #2196f3;">
                    <h2 style="color: #1976d2; margin: 0;">{len(sample_comments)}</h2>
                    <p style="color: #1976d2; margin: 0.5rem 0 0 0; font-weight: 500;">Total Comments</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); 
                            padding: 1.5rem; border-radius: 15px; text-align: center;
                            border: 1px solid #f44336;">
                    <h2 style="color: #d32f2f; margin: 0;">{toxic_count}</h2>
                    <p style="color: #d32f2f; margin: 0.5rem 0 0 0; font-weight: 500;">Toxic Comments</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); 
                            padding: 1.5rem; border-radius: 15px; text-align: center;
                            border: 1px solid #4caf50;">
                    <h2 style="color: #2e7d32; margin: 0;">{clean_count}</h2>
                    <p style="color: #2e7d32; margin: 0.5rem 0 0 0; font-weight: 500;">Clean Comments</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        show_probabilities = st.checkbox("📊 Show probability scores", key="sample_probs")
    
    st.markdown("---")
    st.markdown("""
    <div style="margin: 2rem 0;">
        <h3 style="color: #2c3e50; text-align: center;">🔍 Individual Comment Analysis</h3>
        <p style="color: #6c757d; text-align: center;">Click on any comment below to analyze it individually</p>
    </div>
    """, unsafe_allow_html=True)
    
    for i, comment in enumerate(sample_comments):
        with st.expander(f"💬 Comment {i+1}: {comment[:60]}{'...' if len(comment) > 60 else ''}"):
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f8f9ff 0%, #e8ecff 100%); 
                        padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem;
                        border: 1px solid #e1e8ff;">
                <h5 style="color: #2c3e50; margin-bottom: 1rem;">📝 Full Comment</h5>
                <p style="font-style: italic; color: #495057; margin: 0; font-size: 1.1rem;">"{comment}"</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if st.button(f"🔍 Analyze", key=f"analyze_{i}", type="primary"):
                    with st.spinner("🤖 Analyzing..."):
                        binary_preds = predict_toxicity(comment, return_probabilities=False)
                        prob_preds = predict_toxicity(comment, return_probabilities=True)
                    
                    # Store results in session state
                    st.session_state[f"binary_{i}"] = binary_preds
                    st.session_state[f"prob_{i}"] = prob_preds
            
            with col2:
                # Display results if they exist
                if f"binary_{i}" in st.session_state:
                    binary_preds = st.session_state[f"binary_{i}"]
                    prob_preds = st.session_state[f"prob_{i}"]
                    
                    # Results section
                    st.markdown("""
                    <div style="background: white; padding: 1.5rem; border-radius: 15px; 
                                border: 1px solid #e1e8ff;">
                        <h6 style="color: #2c3e50; margin-bottom: 1rem;">🏷️ Classification Results</h6>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    toxic_count = 0
                    toxic_categories = []
                    
                    for label, prediction in binary_preds.items():
                        if prediction == 1:
                            st.error(f"🚨 **{label.replace('_', ' ').title()}**: DETECTED")
                            toxic_count += 1
                            toxic_categories.append(label.replace('_', ' ').title())
                        else:
                            st.success(f"✅ **{label.replace('_', ' ').title()}**: CLEAN")
                    
                    if show_probabilities:
                        st.markdown("---")
                        st.markdown("**📊 Confidence Scores:**")
                        for label, score in prob_preds.items():
                            # Convert to Python float for display
                            score_float = float(score)
                            st.write(f"• **{label.replace('_', ' ').title()}**: {score_float:.1%}")
                    
                    # Overall assessment
                    st.markdown("---")
                    if toxic_count > 0:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); 
                                    padding: 1rem; border-radius: 10px; margin-top: 1rem;
                                    border: 1px solid #f44336; text-align: center;">
                            <h6 style="color: #d32f2f; margin: 0;">⚠️ TOXIC CONTENT DETECTED</h6>
                            <p style="color: #d32f2f; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                                {toxic_count} categories: {', '.join(toxic_categories)}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); 
                                    padding: 1rem; border-radius: 10px; margin-top: 1rem;
                                    border: 1px solid #4caf50; text-align: center;">
                            <h6 style="color: #2e7d32; margin: 0;">✅ CLEAN CONTENT</h6>
                            <p style="color: #2e7d32; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                                No toxicity detected
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

# Footer with professional styling
st.markdown("---")
st.markdown(f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 2rem; border-radius: 15px; margin-top: 3rem; text-align: center;">
    <div style="color: white;">
        <h3 style="color: white; margin-bottom: 1rem;">🛡️ ToxiGuard AI</h3>
        <p style="margin: 0.5rem 0; opacity: 0.9;">
            <strong>Advanced Toxicity Detection System</strong>
        </p>
        <p style="margin: 0.5rem 0; opacity: 0.8;">
            Powered by Bidirectional LSTM Neural Networks
        </p>
        <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.2);">
            <p style="margin: 0; opacity: 0.7; font-size: 0.9rem;">
                Built with Streamlit & TensorFlow | Binary Classification System<br>
                Threshold: {THRESHOLD} | Max Length: {MAX_LEN} tokens | Categories: 6
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
            
            # Create columns for better layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("""
                <div style="background: white; padding: 1.5rem; border-radius: 15px; 
                            border: 1px solid #e1e8ff; margin-bottom: 1rem;">
                    <h4 style="color: #2c3e50; margin-bottom: 1rem;">🏷️ Binary Classifications</h4>
                </div>
                """, unsafe_allow_html=True)
                
                toxic_categories = []
                for label, prediction in binary_result.items():
                    if prediction == 1:
                        st.error(f"🚨 **{label.replace('_', ' ').title()}**: DETECTED")
                        toxic_categories.append(label.replace('_', ' ').title())
                    else:
                        st.success(f"✅ **{label.replace('_', ' ').title()}**: CLEAN")
            
            if show_probabilities:
                with col2:
                    st.markdown("""
                    <div style="background: white; padding: 1.5rem; border-radius: 15px; 
                                border: 1px solid #e1e8ff; margin-bottom: 1rem;">
                        <h4 style="color: #2c3e50; margin-bottom: 1rem;">📊 Confidence Scores</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for label, score in prob_result.items():
                        # Convert to Python float and ensure it's between 0 and 1
                        normalized_score = max(0.0, min(1.0, float(score)))
                        st.write(f"**{label.replace('_', ' ').title()}:** {normalized_score:.1%}")
                        st.progress(normalized_score)
            
            # Overall toxicity indicator
            toxic_count = sum(binary_result.values())
            if toxic_count > 0:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); 
                            padding: 1.5rem; border-radius: 15px; margin: 2rem 0;
                            border: 2px solid #f44336; text-align: center;">
                    <h3 style="color: #d32f2f; margin-bottom: 0.5rem;">⚠️ TOXIC CONTENT DETECTED</h3>
                    <p style="color: #d32f2f; font-weight: 500; margin: 0;">
                        {toxic_count} toxic categories identified: {', '.join(toxic_categories)}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); 
                            padding: 1.5rem; border-radius: 15px; margin: 2rem 0;
                            border: 2px solid #4caf50; text-align: center;">
                    <h3 style="color: #2e7d32; margin-bottom: 0.5rem;">✅ CLEAN CONTENT</h3>
                    <p style="color: #2e7d32; font-weight: 500; margin: 0;">
                        No toxicity detected in this comment
                    </p>
                </div>
                """, unsafe_allow_html=True)

# ---------------------------
# 2. Bulk Prediction
# ---------------------------
elif page == "📂 Bulk Prediction":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>📂 Bulk Content Processing</h2>
        <p style="color: #6c757d;">Upload CSV files for batch toxicity analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional file upload section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8f9ff 0%, #e8ecff 100%); 
                padding: 2rem; border-radius: 15px; margin-bottom: 2rem; 
                border: 1px solid #e1e8ff;">
        <h4 style="color: #2c3e50; margin-bottom: 1rem;">📋 File Requirements</h4>
        <ul style="color: #6c757d; margin: 0;">
            <li>CSV format with 'comment_text' column</li>
            <li>UTF-8 encoding recommended</li>
            <li>Maximum file size: 200MB</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose your CSV file", 
        type=["csv"], 
        help="Upload a CSV file containing comments to analyze"
    )
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            
            if "comment_text" not in data.columns:
                st.error("❌ CSV must have a column named 'comment_text'")
                st.info(f"📊 Available columns: {', '.join(data.columns.tolist())}")
            else:
                # Success message with file info
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); 
                            padding: 1.5rem; border-radius: 15px; margin: 1rem 0;
                            border: 1px solid #4caf50;">
                    <h4 style="color: #2e7d32; margin-bottom: 0.5rem;">✅ File Uploaded Successfully</h4>
                    <p style="margin: 0; color: #2e7d32;">
                        📊 <strong>{len(data):,}</strong> records found | 
                        📁 <strong>{uploaded_file.name}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("👀 Preview Data (First 10 rows)"):
                    st.dataframe(data.head(10), use_container_width=True)

                col1, col2 = st.columns([1, 1])
                with col1:
                    include_probabilities = st.checkbox(
                        "📊 Include confidence scores", 
                        help="Add probability columns alongside binary predictions"
                    )
                
                if st.button("🚀 Start Bulk Analysis", type="primary"):
                    # Progress section
                    st.markdown("""
                    <div style="background: white; padding: 2rem; border-radius: 15px; 
                                margin: 1rem 0; border: 1px solid #e1e8ff;">
                        <h4 style="color: #2c3e50;">🔄 Processing Comments...</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    binary_predictions = []
                    prob_predictions = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, text in enumerate(data["comment_text"].fillna("")):
                        binary_preds = predict_toxicity(text, return_probabilities=False)
                        binary_predictions.append(binary_preds)
                        
                        if include_probabilities:
                            prob_preds = predict_toxicity(text, return_probabilities=True)
                            prob_predictions.append(prob_preds)
                        
                        progress_bar.progress((i + 1) / len(data))
                        status_text.markdown(f"""
                        <div style="text-align: center; color: #6c757d;">
                            Processing: <strong>{i + 1:,}/{len(data):,}</strong> comments 
                            (<strong>{((i + 1) / len(data) * 100):.1f}%</strong>)
                        </div>
                        """, unsafe_allow_html=True)

                    # Create results dataframe
                    binary_df = pd.DataFrame(binary_predictions)
                    result_df = pd.concat([data, binary_df], axis=1)
                    
                    if include_probabilities:
                        prob_df = pd.DataFrame(prob_predictions)
                        prob_df.columns = [f"{col}_confidence" for col in prob_df.columns]
                        result_df = pd.concat([result_df, prob_df], axis=1)

                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); 
                                padding: 1.5rem; border-radius: 15px; margin: 2rem 0;
                                border: 2px solid #4caf50; text-align: center;">
                        <h3 style="color: #2e7d32; margin-bottom: 0.5rem;">🎉 Analysis Complete!</h3>
                        <p style="color: #2e7d32; font-weight: 500; margin: 0;">
                            All comments have been processed and classified
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show results preview
                    with st.expander("👀 Results Preview (First 10 rows)"):
                        st.dataframe(result_df.head(10), use_container_width=True)

                    # Summary statistics with professional cards
                    st.markdown("""
                    <div style="margin: 2rem 0;">
                        <h3 style="color: #2c3e50; text-align: center; margin-bottom: 2rem;">📊 Analysis Summary</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
                    
                    # Main metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total_comments = len(data)
                    toxic_comments = (binary_df['toxic'] == 1).sum()
                    clean_comments = total_comments - toxic_comments
                    toxicity_rate = (toxic_comments / total_comments) * 100
                    
                    with col1:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                                    padding: 1.5rem; border-radius: 15px; text-align: center;
                                    border: 1px solid #2196f3; margin-bottom: 1rem;">
                            <h2 style="color: #1976d2; margin: 0;">{total_comments:,}</h2>
                            <p style="color: #1976d2; margin: 0.5rem 0 0 0; font-weight: 500;">Total Comments</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); 
                                    padding: 1.5rem; border-radius: 15px; text-align: center;
                                    border: 1px solid #f44336; margin-bottom: 1rem;">
                            <h2 style="color: #d32f2f; margin: 0;">{toxic_comments:,}</h2>
                            <p style="color: #d32f2f; margin: 0.5rem 0 0 0; font-weight: 500;">Toxic Comments</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); 
                                    padding: 1.5rem; border-radius: 15px; text-align: center;
                                    border: 1px solid #4caf50; margin-bottom: 1rem;">
                            <h2 style="color: #2e7d32; margin: 0;">{clean_comments:,}</h2>
                            <p style="color: #2e7d32; margin: 0.5rem 0 0 0; font-weight: 500;">Clean Comments</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        rate_color = "#d32f2f" if toxicity_rate > 20 else "#ff9800" if toxicity_rate > 10 else "#2e7d32"
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); 
                                    padding: 1.5rem; border-radius: 15px; text-align: center;
                                    border: 1px solid #ff9800; margin-bottom: 1rem;">
                            <h2 style="color: {rate_color}; margin: 0;">{toxicity_rate:.1f}%</h2>
                            <p style="color: {rate_color}; margin: 0.5rem 0 0 0; font-weight: 500;">Toxicity Rate</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Category breakdown chart
                    st.markdown("""
                    <div style="margin: 2rem 0;">
                        <h4 style="color: #2c3e50; text-align: center;">🏷️ Toxicity Categories Breakdown</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    category_counts = binary_df[labels].sum()
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
                    bars = ax.bar(category_counts.index, category_counts.values, color=colors)
                    ax.set_title("Toxic Comments by Category", fontsize=16, fontweight='bold', pad=20)
                    ax.set_ylabel("Number of Toxic Comments", fontsize=12)
                    ax.set_xlabel("Toxicity Categories", fontsize=12)
                    
                    # Improve x-axis labels
                    category_labels = [label.replace('_', '\n').title() for label in category_counts.index]
                    ax.set_xticks(range(len(category_labels)))
                    ax.set_xticklabels(category_labels, fontsize=10)
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + max(category_counts.values()) * 0.01,
                               f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
                    
                    # Style improvements
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.grid(axis='y', alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Download section
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #f8f9ff 0%, #e8ecff 100%); 
                                padding: 2rem; border-radius: 15px; margin: 2rem 0;
                                border: 1px solid #667eea; text-align: center;">
                        <h4 style="color: #2c3e50; margin-bottom: 1rem;">💾 Download Results</h4>
                        <p style="color: #6c757d; margin-bottom: 1.5rem;">
                            Get your complete analysis results with all predictions and confidence scores
                        </p>
                    """, unsafe_allow_html=True)
                    
                    csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Complete Results (CSV)", 
                        csv, 
                        f"toxicity_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                        "text/csv",
                        type="primary",
                        help="Download the complete results with binary predictions and confidence scores"
                    )
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            st.info("💡 Please ensure your CSV file is properly formatted and contains a 'comment_text' column.")

# ---------------------------
# 3. Model Insights & Metrics
# ---------------------------
elif page == "📊 Model Insights":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>📊 Model Architecture & Performance</h2>
        <p style="color: #6c757d;">Comprehensive model analysis and performance metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get model metrics
    model_info, layer_info = get_model_metrics()
    
    # Model Architecture Overview
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; margin-bottom: 2rem; color: white;">
        <h3 style="color: white; margin-bottom: 1rem; text-align: center;">🏗️ Model Architecture Overview</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics in cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                    padding: 2rem; border-radius: 15px; text-align: center;
                    border: 1px solid #2196f3;">
            <h2 style="color: #1976d2; margin: 0;">{model_info.get('Total Parameters', '0')}</h2>
            <p style="color: #1976d2; margin: 0.5rem 0 0 0; font-weight: 500;">Total Parameters</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); 
                    padding: 2rem; border-radius: 15px; text-align: center;
                    border: 1px solid #4caf50;">
            <h2 style="color: #2e7d32; margin: 0;">{model_info.get('Trainable Parameters', '0')}</h2>
            <p style="color: #2e7d32; margin: 0.5rem 0 0 0; font-weight: 500;">Trainable Parameters</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); 
                    padding: 2rem; border-radius: 15px; text-align: center;
                    border: 1px solid #ff9800;">
            <h2 style="color: #ef6c00; margin: 0;">{len(layer_info)}</h2>
            <p style="color: #ef6c00; margin: 0.5rem 0 0 0; font-weight: 500;">Network Layers</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model Configuration Details
    st.markdown("""
    <div style="margin: 2rem 0;">
        <h4 style="color: #2c3e50;">ℹ️ Model Configuration</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 15px; 
                    border: 1px solid #e1e8ff; margin-bottom: 1rem;">
            <h5 style="color: #2c3e50; margin-bottom: 1rem;">📋 Architecture Details</h5>
        """, unsafe_allow_html=True)
        
        for key, value in model_info.items():
            if key not in ['Total Parameters', 'Trainable Parameters', 'Non-trainable Parameters']:
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; 
                            border-bottom: 1px solid #f0f0f0;">
                    <span style="font-weight: 500; color: #2c3e50;">{key}:</span>
                    <span style="color: #6c757d;">{value}</span>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 15px; 
                    border: 1px solid #e1e8ff; margin-bottom: 1rem;">
            <h5 style="color: #2c3e50; margin-bottom: 1rem;">🔧 Layer Architecture</h5>
        """, unsafe_allow_html=True)
        
        if layer_info:
            layer_df = pd.DataFrame(layer_info)
            # Style the dataframe
            st.dataframe(
                layer_df, 
                use_container_width=True,
                hide_index=True
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Performance Analysis Section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8f9ff 0%, #e8ecff 100%); 
                padding: 2rem; border-radius: 15px; margin: 2rem 0;
                border: 1px solid #667eea;">
        <h4 style="color: #2c3e50; text-align: center; margin-bottom: 1rem;">🎯 Performance Evaluation</h4>
        <p style="color: #6c757d; text-align: center; margin: 0;">
            Run comprehensive performance analysis on curated test dataset
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🧪 Run Performance Analysis", type="primary"):
        with st.spinner("🤖 Evaluating model performance..."):
            evaluation_results = evaluate_model_performance()
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); 
                    padding: 1.5rem; border-radius: 15px; margin: 2rem 0;
                    border: 2px solid #4caf50; text-align: center;">
            <h3 style="color: #2e7d32; margin-bottom: 0.5rem;">✅ Evaluation Complete!</h3>
            <p style="color: #2e7d32; font-weight: 500; margin: 0;">
                Model performance metrics calculated successfully
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Overall Performance Metrics
        st.markdown("""
        <div style="margin: 2rem 0;">
            <h4 style="color: #2c3e50; text-align: center;">📈 Overall Performance Metrics</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        overall_accuracy = evaluation_results['overall_accuracy']
        macro_f1 = evaluation_results['classification_report']['macro avg']['f1-score']
        weighted_f1 = evaluation_results['classification_report']['weighted avg']['f1-score']
        test_samples = len(evaluation_results['test_data'])
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                        padding: 1.5rem; border-radius: 15px; text-align: center;
                        border: 1px solid #2196f3;">
                <h2 style="color: #1976d2; margin: 0;">{overall_accuracy:.1%}</h2>
                <p style="color: #1976d2; margin: 0.5rem 0 0 0; font-weight: 500;">Overall Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); 
                        padding: 1.5rem; border-radius: 15px; text-align: center;
                        border: 1px solid #4caf50;">
                <h2 style="color: #2e7d32; margin: 0;">{macro_f1:.3f}</h2>
                <p style="color: #2e7d32; margin: 0.5rem 0 0 0; font-weight: 500;">Macro F1-Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); 
                        padding: 1.5rem; border-radius: 15px; text-align: center;
                        border: 1px solid #ff9800;">
                <h2 style="color: #ef6c00; margin: 0;">{weighted_f1:.3f}</h2>
                <p style="color: #ef6c00; margin: 0.5rem 0 0 0; font-weight: 500;">Weighted F1-Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); 
                        padding: 1.5rem; border-radius: 15px; text-align: center;
                        border: 1px solid #9c27b0;">
                <h2 style="color: #7b1fa2; margin: 0;">{test_samples}</h2>
                <p style="color: #7b1fa2; margin: 0.5rem 0 0 0; font-weight: 500;">Test Samples</p>
            </div>
            """, unsafe_allow_html=True)