import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import io
import os
import json
import re
import string
import gdown
import zipfile
import shutil
from pathlib import Path

st.set_page_config(page_title="Toxic Comment Classifier", layout="wide")

# Add CSS styling with #DDD4FF background and pink gradient accents for containers and buttons
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
        background-color: #DDD4FF !important;
    }
    
    /* Streamlit main container background */
    .stApp {
        background-color: #DDD4FF !important;
    }
    
    /* Main content area - Pink Gradient Container */
    .main .block-container {
        background: linear-gradient(135deg, #F8BBD0 0%, #F48FB1 100%) !important;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem auto;
        box-shadow: 0 8px 25px rgba(244, 143, 177, 0.2);
        border: 1px solid rgba(244, 143, 177, 0.3);
    }
    
    /* Navigation Styles */
    .nav-title {
        background: rgba(255, 255, 255, 0.9);
        color: #4A4A4A;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Hero Section */
    .hero-section {
        background: rgba(255, 255, 255, 0.9);
        color: #4A4A4A;
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        margin: 30px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .hero-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 20px;
        color: #4A4A4A;
    }
    
    .hero-subtitle {
        font-size: 18px;
        font-weight: 300;
        line-height: 1.6;
        max-width: 800px;
        margin: 0 auto;
        color: #666;
    }
    
    /* Feature Cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.9);
        color: #4A4A4A;
        border-radius: 12px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(0, 0, 0, 0.1);
    }

    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    }

    .feature-icon {
        font-size: 32px;
        margin-bottom: 15px;
        text-align: center;
        color: #F48FB1;
    }

    .feature-title {
        color: #4A4A4A;
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 15px;
        text-align: center;
    }

    .feature-description {
        color: #666;
        font-size: 14px;
        line-height: 1.6;
        text-align: center;
    }
    
    /* Stats Container */
    .stats-container {
        background: rgba(255, 255, 255, 0.9);
        color: #4A4A4A;
        padding: 30px;
        border-radius: 15px;
        margin: 30px 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 20px;
        margin-top: 25px;
    }
    
    .stat-item {
        background: rgba(248, 187, 208, 0.2);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(244, 143, 177, 0.2);
    }
    
    .stat-number {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 8px;
        color: #F48FB1;
    }
    
    .stat-label {
        font-size: 14px;
        font-weight: 500;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Custom Button Styles - Pink Gradient */
    .stButton > button {
        background: linear-gradient(135deg, #F48FB1 0%, #EC407A 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(244, 143, 177, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(244, 143, 177, 0.4) !important;
        background: linear-gradient(135deg, #EC407A 0%, #D81B60 100%) !important;
    }
    
    /* Primary Button Variant */
    .stButton > button[data-baseweb="button"][kind="primary"] {
        background: linear-gradient(135deg, #EC407A 0%, #D81B60 100%) !important;
        box-shadow: 0 4px 15px rgba(236, 64, 122, 0.4) !important;
    }
    
    .stButton > button[data-baseweb="button"][kind="primary"]:hover {
        background: linear-gradient(135deg, #D81B60 0%, #C2185B 100%) !important;
        box-shadow: 0 8px 25px rgba(216, 27, 96, 0.5) !important;
    }
    
    /* Alert Styles */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        background-color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Metric Styles */
    .metric-container {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(0, 0, 0, 0.1);
    }
    
    /* Progress Bar */
    .stProgress .st-bo {
        background: linear-gradient(135deg, #F48FB1 0%, #EC407A 100%) !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #DDD4FF 0%, #CCC2FF 100%) !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #4A4A4A;
        font-weight: 600;
    }
    
    h1 {
        border-bottom: 3px solid #F48FB1;
        padding-bottom: 12px;
    }
    
    /* Dataframe Styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        background-color: rgba(255, 255, 255, 0.9);
    }
    
    /* File Uploader */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 2px dashed #F48FB1;
        padding: 20px;
    }
    
    /* Text Areas and Inputs */
    .stTextArea > div > div > textarea,
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        font-size: 16px;
        padding: 12px;
        transition: border-color 0.3s ease;
        background-color: rgba(255, 255, 255, 0.9) !important;
    }
    
    .stTextArea > div > div > textarea:focus,
    .stTextInput > div > div > input:focus {
        border-color: #F48FB1;
        box-shadow: 0 0 0 3px rgba(244, 143, 177, 0.1);
    }
    
    /* Expander header */
    .streamlit-expanderHeader {
        background: rgba(248, 187, 208, 0.3) !important;
        color: #4A4A4A !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: 1px solid rgba(244, 143, 177, 0.2);
    }
    
    .streamlit-expanderContent {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border-radius: 0 0 8px 8px;
    }
    
    /* Checkbox */
    .stCheckbox {
        font-size: 16px;
        background-color: rgba(255, 255, 255, 0.9);
        padding: 10px;
        border-radius: 8px;
        border: 1px solid rgba(0, 0, 0, 0.1);
    }
    
    /* Select boxes and other inputs */
    .stSelectbox > div > div > div {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border-radius: 8px;
    }
    
    /* Columns styling */
    .st-emotion-cache-ocqkz7 {
        background-color: transparent;
    }
    
    /* Custom spacing */
    .section-spacing {
        margin: 40px 0;
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background-color: rgba(212, 237, 218, 0.9) !important;
        border-radius: 10px;
    }
    
    .stError {
        background-color: rgba(248, 215, 218, 0.9) !important;
        border-radius: 10px;
    }
    
    .stInfo {
        background-color: rgba(209, 236, 241, 0.9) !important;
        border-radius: 10px;
    }
    
    .stWarning {
        background-color: rgba(255, 243, 205, 0.9) !important;
        border-radius: 10px;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 28px;
        }
        
        .hero-subtitle {
            font-size: 16px;
        }
        
        .stats-grid {
            grid-template-columns: 1fr;
        }
        
        .feature-card {
            padding: 20px;
        }
        
        .main .block-container {
            padding: 1.5rem;
            margin: 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'

# Google Drive file IDs - REPLACE THESE WITH YOUR ACTUAL FILE IDs (IDs OR FULL SHARE URLS ARE OK)
GOOGLE_DRIVE_CONFIG = {
    "tokenizer_file_id": "1tVNqozfIRQPIFI7AQ6pAVZDZNG9J6v2I",
    "model_file_id": "1PykKw6x1aRw8nkyhe0bZGV8allQId3NN",
}

def _extract_drive_id(value: str) -> str | None:
    """Return a Google Drive file id from either a raw id or a share/download URL."""
    if not value:
        return None
    value = value.strip()
    # Already an ID?
    if _is_probable_drive_id(value):
        return value
    # Try to pull from common URL formats
    import re
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]{10,})",      # .../file/d/<id>/view
        r"[?&]id=([a-zA-Z0-9_-]{10,})",       # ...?id=<id>
        r"/uc\?export=download&(?:.*&)?id=([a-zA-Z0-9_-]{10,})",
        r"/open\?(?:.*&)?id=([a-zA-Z0-9_-]{10,})",
    ]
    for pat in patterns:
        m = re.search(pat, value)
        if m:
            return m.group(1)
    return None

def _is_probable_drive_id(value: str) -> bool:
    """Heuristic: Drive IDs are URL-safe, usually 25+ chars of [A-Za-z0-9_-]."""
    import re
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{10,}", value or ""))

def validate_and_normalize_file_ids(config: dict) -> tuple[bool, dict]:
    """
    Validate presence & shape of Drive IDs. Accepts either raw IDs or full URLs.
    Returns (ok, normalized_config_with_ids_only).
    """
    normalized = dict(config)
    problems = []

    for key in ["tokenizer_file_id", "model_file_id"]:
        raw = normalized.get(key, "") or ""
        extracted = _extract_drive_id(raw)
        if not extracted:
            problems.append(f"- `{key}` is missing or not a valid Google Drive file ID/URL.")
        elif not _is_probable_drive_id(extracted):
            problems.append(f"- `{key}` doesn’t look like a valid Drive file ID.")
        else:
            normalized[key] = extracted  # store the clean ID

    if problems:
        st.error("Google Drive file ID validation failed:\n" + "\n".join(problems))
        st.info("Tip: paste the full share URL (…/file/d/<ID>/view) or just the file ID.")
        return False, config
    return True, normalized

def _gdrive_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?id={file_id}"

def download_from_gdrive(file_id, output_path, file_type="tokenizer"):
    """Download files from Google Drive with progress tracking (accepts raw ID or URL)."""
    try:
        # Normalize input (ID or URL)
        norm_id = _extract_drive_id(file_id)
        if not norm_id:
            st.error(f"Invalid Google Drive {file_type} reference: {file_id}")
            return False

        url = _gdrive_download_url(norm_id)

        # Create directory for the file if needed
        dirn = os.path.dirname(output_path)
        if dirn:
            os.makedirs(dirn, exist_ok=True)

        with st.spinner(f"Downloading {file_type} from Google Drive..."):
            # fuzzy=True lets gdown accept IDs/URLs; we pass a neat url anyway
            gdown.download(url, output_path, quiet=False, fuzzy=True)

        if os.path.exists(output_path):
            st.success(f"{file_type.title()} downloaded successfully!")
            return True
        else:
            st.error(f"Failed to download {file_type}")
            return False

    except Exception as e:
        st.error(f"Error downloading {file_type}: {str(e)}")
        return False

def extract_zip_file(zip_path, extract_to):
    """Extract zip file to specified directory"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(zip_path)
        return True
    except Exception as e:
        st.error(f"Error extracting zip file: {str(e)}")
        return False

def normalize_extracted_folder(path):
    """Fix one-level nesting (path/path/* -> path/*)."""
    try:
        if not os.path.isdir(path):
            return
        contents = os.listdir(path)
        if len(contents) == 1:
            inner = os.path.join(path, contents[0])
            if os.path.isdir(inner):
                for item in os.listdir(inner):
                    shutil.move(os.path.join(inner, item), path)
                shutil.rmtree(inner)
    except Exception as e:
        st.warning(f"Normalization skipped: {e}")

def _ensure_tokenizer_dir(expected_path: str) -> bool:
    """
    Ensure a directory named `expected_path` exists and contains tokenizer files (vocab.txt).
    If the zip extracted under a different folder name, find it and move/rename.
    """
    # already good?
    if os.path.isdir(expected_path) and os.path.exists(os.path.join(expected_path, "vocab.txt")):
        return True

    # search top-level dirs for a vocab.txt
    for d in os.listdir("."):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "vocab.txt")):
            if d != expected_path:
                shutil.rmtree(expected_path, ignore_errors=True)
                shutil.move(d, expected_path)
            return True
    return False

def _ensure_model_dir(expected_path: str) -> bool:
    """
    Ensure a directory named `expected_path` exists and contains TF model files.
    Expect: config.json AND (tf_model.h5 OR saved_model.pb).
    """
    def has_tf_weights(p):
        return os.path.exists(os.path.join(p, "config.json")) and (
            os.path.exists(os.path.join(p, "tf_model.h5")) or
            os.path.exists(os.path.join(p, "saved_model.pb"))
        )

    if os.path.isdir(expected_path) and has_tf_weights(expected_path):
        return True

    # search top-level dirs
    for d in os.listdir("."):
        if os.path.isdir(d) and has_tf_weights(d):
            if d != expected_path:
                shutil.rmtree(expected_path, ignore_errors=True)
                shutil.move(d, expected_path)
            return True
    return False

@st.cache_resource
def load_bert_model_and_tokenizer():
    """Load BERT model and tokenizer with cloud deployment support"""
    try:
        # Define paths for cloud deployment (no './')
        tokenizer_path = "bert_tokenizer"
        model_path = "bert_model"
        tokenizer_zip_path = "bert_tokenizer.zip"
        model_zip_path = "bert_model.zip"

        # Validate & normalize Drive IDs (accept URLs)
        ok, normalized_ids = validate_and_normalize_file_ids(GOOGLE_DRIVE_CONFIG)
        if not ok:
            return None, None

        # --- Step 1: If already present, try to load ---
        if os.path.exists(tokenizer_path) and os.path.exists(model_path):
            st.info("Loading existing BERT model files...")
            try:
                tokenizer = BertTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
                model = TFBertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
                st.success("BERT model and tokenizer loaded successfully!")
                return tokenizer, model
            except Exception as e:
                st.warning(f"Error loading existing files: {e}. Will download fresh copies.")
                shutil.rmtree(tokenizer_path, ignore_errors=True)
                shutil.rmtree(model_path, ignore_errors=True)

        st.info("BERT model files not found. Downloading from Google Drive...")

        # --- Step 2: Download tokenizer zip ---
        if not os.path.exists(tokenizer_path):
            if download_from_gdrive(normalized_ids["tokenizer_file_id"], tokenizer_zip_path, "tokenizer"):
                if not extract_zip_file(tokenizer_zip_path, "."):
                    return None, None
                normalize_extracted_folder(tokenizer_path)
                if not _ensure_tokenizer_dir(tokenizer_path):
                    st.error(
                        "Tokenizer folder not found or missing files. "
                        "Expected a folder containing at least `vocab.txt`."
                    )
                    return None, None
                st.success("Tokenizer extracted successfully!")
            else:
                return None, None

        # --- Step 3: Download model zip ---
        if not os.path.exists(model_path):
            if download_from_gdrive(normalized_ids["model_file_id"], model_zip_path, "model"):
                if not extract_zip_file(model_zip_path, "."):
                    return None, None
                normalize_extracted_folder(model_path)
                if not _ensure_model_dir(model_path):
                    st.error(
                        "Model folder not found or missing files. "
                        "Expected `config.json` and `tf_model.h5` (or `saved_model.pb`)."
                    )
                    return None, None
                st.success("Model extracted successfully!")
            else:
                return None, None

        # --- Step 4: Load final model + tokenizer ---
        st.info("Loading BERT model and tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        model = TFBertForSequenceClassification.from_pretrained(
            model_path, num_labels=6, local_files_only=True
        )

        st.success("BERT model and tokenizer loaded successfully!")
        return tokenizer, model

    except Exception as e:
        st.error(f"Error loading BERT model or tokenizer: {str(e)}")
        st.error("""
        **Troubleshooting tips:**
        1. Ensure your Google Drive files are publicly accessible
        2. Verify the file IDs are correct (raw ID or full share URL)
        3. Upload ZIP files, not folders
        4. ZIP must contain: 
           - tokenizer: vocab.txt (+ tokenizer_config.json if available)
           - model: config.json + tf_model.h5 (or saved_model.pb)
        """)
        return None, None

# Load BERT model and tokenizer
with st.spinner("Initializing BERT model..."):
    tokenizer, model = load_bert_model_and_tokenizer()

# BERT-specific constants
MAX_LEN = 512  # BERT's maximum sequence length
THRESHOLD = 0.5  # Threshold for binary classification

# Check if model and tokenizer loaded successfully
if model is None or tokenizer is None:
    st.error("Failed to load BERT model or tokenizer.")
    st.info("""
    **Setup Instructions for Cloud Deployment:**
    
    1. **Prepare your model files:**
       - Create a ZIP file of your bert_tokenizer folder
       - Create a ZIP file of your bert_model folder
    
    2. **Upload to Google Drive:**
       - Upload both ZIP files to Google Drive
       - Make them publicly accessible (Anyone with the link can view)
    
    3. **Get file IDs:**
       - Right-click each file → Share → Copy link
       - Paste the full link or just the file ID into GOOGLE_DRIVE_CONFIG
    """)
    st.stop()

def preprocess_text_for_bert(text):
    """Preprocess text for BERT (minimal preprocessing needed)"""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def predict_toxicity_bert(text, return_probabilities=False):
    """Predict toxicity for a single text using BERT"""
    if not isinstance(text, str):
        text = str(text)
    
    # Preprocess text
    text = preprocess_text_for_bert(text)
    
    # Tokenize for BERT
    inputs = tokenizer(
        text,
        return_tensors='tf',
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True
    )
    
    # Get model prediction
    outputs = model(inputs)
    logits = outputs.logits
    
    # Apply sigmoid to get probabilities (for multi-label classification)
    probabilities = tf.nn.sigmoid(logits).numpy()[0]
    
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
   
    if return_probabilities:
        # Return probabilities for internal use
        return dict(zip(labels, [float(p) for p in probabilities]))
    else:
        # Return binary predictions (1/0) for user display
        binary_preds = [1 if p > THRESHOLD else 0 for p in probabilities]
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

def get_bert_model_metrics():
    try:
        # Get model configuration
        config = model.config
        
        # Model info
        model_info = {
            "Model Type": "BERT for Sequence Classification",
            "Model Name": config.name_or_path if hasattr(config, 'name_or_path') else "bert-base-uncased",
            "Vocabulary Size": config.vocab_size,
            "Hidden Size": config.hidden_size,
            "Number of Attention Heads": config.num_attention_heads,
            "Number of Hidden Layers": config.num_hidden_layers,
            "Max Position Embeddings": config.max_position_embeddings,
            "Number of Labels": config.num_labels,
            "Hidden Dropout": config.hidden_dropout_prob,
            "Attention Dropout": config.attention_probs_dropout_prob,
        }

        # Layer info for BERT
        layer_info = [
            {"Component": "Embeddings", "Description": "Token, Position, and Segment Embeddings"},
            {"Component": "Encoder Layers", "Description": f"{config.num_hidden_layers} Transformer Blocks"},
            {"Component": "Attention Heads", "Description": f"{config.num_attention_heads} per layer"},
            {"Component": "Classification Head", "Description": f"Linear layer for {config.num_labels} labels"},
        ]

        return model_info, layer_info

    except Exception as e:
        st.error(f"Error extracting BERT model metrics: {e}")
        return {}, []

def evaluate_bert_performance():
    """Evaluate BERT model performance on test data"""
    # Create comprehensive test dataset
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
        prob_result = predict_toxicity_bert(text, return_probabilities=True)
        prediction = [prob_result[label] for label in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
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
    st.markdown('<div class="nav-title">BERT Toxic Comment Detection System</div>', unsafe_allow_html=True)
    
    # Navigation buttons
    pages = ['Home', 'Live Detection', 'Bulk Analysis', 'Model Insights', 'Test Cases']
    page_keys = ['Home', 'Live Detection', 'Bulk Analysis', 'Model Insights', 'Test Cases']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    columns = [col1, col2, col3, col4, col5]
    
    for i, (page, key) in enumerate(zip(pages, page_keys)):
        with columns[i]:
            if st.button(page, key=f"nav_{key}", use_container_width=True):
                st.session_state.current_page = key
                st.rerun()
    
    return st.session_state.current_page

# Navigation
current_page = render_navigation()

# Add some spacing
st.markdown("<br>", unsafe_allow_html=True)

# HOME PAGE
if current_page == 'Home':
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🚀 Advanced BERT Toxicity Detection</div>
        <div class="hero-subtitle">
            Powered by state-of-the-art BERT transformer architecture, our AI system provides highly accurate 
            toxicity detection across multiple categories with contextual understanding.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Section
    st.markdown("## ✨ Key Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">BERT-Powered Detection</div>
            <div class="feature-description">
                State-of-the-art transformer architecture with bidirectional context understanding 
                for superior toxicity detection accuracy.
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
                Comprehensive BERT model insights with attention mechanisms, transformer layers,
                and detailed performance analysis across all toxicity categories.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧪</div>
            <div class="feature-title">Sample Test Cases</div>
            <div class="feature-description">
                Pre-loaded test comments to explore BERT model behavior and understand 
                classification patterns with contextual awareness.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Getting Started Section
    st.markdown("## 🚀 Get Started")
    
    st.markdown("""
    <div style="text-align: center; max-width: 800px; margin: 30px auto; padding: 30px; background: rgba(255, 255, 255, 0.95); border-radius: 20px; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.06);">
        <p style="color: #666; font-size: 18px; line-height: 1.6; margin-bottom: 30px;">
            Ready to start detecting toxic content with BERT? Choose from our powerful tools above to begin your analysis journey.
        </p>
        <div style="display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; margin-top: 25px;">
            <div style="background: linear-gradient(135deg, #9370DB 0%, #8A2BE2 100%); color: white; padding: 15px 25px; border-radius: 25px; font-weight: 500; box-shadow: 0 4px 15px rgba(147, 112, 219, 0.3);">
                ⚡ Live Detection - For single comments
            </div>
            <div style="background: linear-gradient(135deg, #9370DB 0%, #8A2BE2 100%); color: white; padding: 15px 25px; border-radius: 25px; font-weight: 500; box-shadow: 0 4px 15px rgba(147, 112, 219, 0.3);">
                📊 Bulk Analysis For CSV files
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats Section
    st.markdown("""
    <div class="stats-container">
        <h2 style="margin-bottom: 20px; font-size: 32px;">BERT System Performance</h2>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-number">512</div>
                <div class="stat-label">Max Token Length</div>
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
                <div class="stat-number">12</div>
                <div class="stat-label">Transformer Layers</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # How it Works Section
    st.markdown("## 🔍 How BERT Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 30px 20px; background: rgba(255, 255, 255, 0.95); border-radius: 15px; margin: 10px 0; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);">
            <div style="font-size: 48px; margin-bottom: 20px;">📝</div>
            <h4 style="color: #9370DB; margin-bottom: 15px;">1. Input Text</h4>
            <p style="color: #666; line-height: 1.6;">Enter your comment or upload a CSV file. BERT tokenizes and processes the text with attention mechanisms.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 30px 20px; background: rgba(255, 255, 255, 0.95); border-radius: 15px; margin: 10px 0; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);">
            <div style="font-size: 48px; margin-bottom: 20px;">⚙️</div>
            <h4 style="color: #9370DB; margin-bottom: 15px;">2. BERT Processing</h4>
            <p style="color: #666; line-height: 1.6;">12 transformer layers with self-attention analyze bidirectional context for deep understanding.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 30px 20px; background: rgba(255, 255, 255, 0.95); border-radius: 15px; margin: 10px 0; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);">
            <div style="font-size: 48px; margin-bottom: 20px;">📊</div>
            <h4 style="color: #9370DB; margin-bottom: 15px;">3. Results</h4>
            <p style="color: #666; line-height: 1.6;">Get precise binary classifications (1/0) for six different toxicity categories with confidence scores.</p>
        </div>
        """, unsafe_allow_html=True)

# LIVE DETECTION PAGE
elif current_page == 'Live Detection':
    st.header("⚡ Real-time BERT Toxicity Detection")
    st.markdown("*Enter one or more comments (each line will be analyzed separately using BERT)*")

    # Multi-line input box
    user_input = st.text_area(
        "Type comments below:", 
        height=180, 
        placeholder="Enter one comment per line..."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        predict_button = st.button("🔍 Analyze with BERT", type="primary", use_container_width=True)
    with col2:
        show_probabilities = st.checkbox("Show Probabilities")

    if predict_button:
        # Split input into separate comments
        comments = [c.strip() for c in user_input.split("\n") if c.strip()]
        
        if not comments:
            st.warning("⚠️ Please enter at least one valid comment.")
        else:
            for idx, comment in enumerate(comments, start=1):
                st.subheader(f"💬 Comment {idx}:")
                
                with st.spinner("Analyzing with BERT..."):
                    # Get binary predictions
                    binary_result = predict_toxicity_bert(comment, return_probabilities=False)
                    # Get probabilities
                    prob_result = predict_toxicity_bert(comment, return_probabilities=True)

                # Layout columns
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown("**Binary Classifications:**")
                    for label, prediction in binary_result.items():
                        if prediction == 1:
                            st.error(f"{label.replace('_', ' ').title()}: **{prediction}** (TOXIC)")
                        else:
                            st.success(f"{label.replace('_', ' ').title()}: **{prediction}** (NON-TOXIC)")

                if show_probabilities:
                    with col2:
                        st.markdown("**Probability Scores:**")
                        for label, score in prob_result.items():
                            normalized_score = max(0.0, min(1.0, float(score)))
                            st.write(f"**{label.replace('_', ' ').title()}:** {score:.3f}")
                            st.progress(normalized_score)

                # Overall toxicity indicator
                toxic_count = sum(binary_result.values())
                if toxic_count > 0:
                    st.error(f"**TOXIC CONTENT DETECTED** - {toxic_count} toxic categories identified by BERT!")
                else:
                    st.success("**CLEAN CONTENT** - No toxicity detected by BERT!")

# BULK ANALYSIS PAGE
elif current_page == 'Bulk Analysis':
    st.header("📊 Bulk CSV Analysis with BERT")
    st.markdown("*Upload a CSV file with 'text' column to get BERT predictions for all comments.*")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], help="CSV must contain a column named 'text'")
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            
            if "text" not in data.columns:
                st.error("CSV must have a column named 'text'")
                st.info("Available columns: " + ", ".join(data.columns.tolist()))
            else:
                st.success(f"File uploaded successfully! Found **{len(data)}** rows.")
                
                with st.expander("Preview Data"):
                    st.dataframe(data.head(10))

                col1, col2 = st.columns([1, 1])
                with col1:
                    include_probabilities = st.checkbox("Include probability scores", help="Add probability columns alongside binary predictions")
                
                if st.button("Run BERT Bulk Predictions", type="primary"):
                    # Predictions with progress bar
                    binary_predictions = []
                    prob_predictions = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, text in enumerate(data["text"].fillna("")):
                        binary_preds = predict_toxicity_bert(text, return_probabilities=False)
                        binary_predictions.append(binary_preds)
                        
                        if include_probabilities:
                            prob_preds = predict_toxicity_bert(text, return_probabilities=True)
                            prob_predictions.append(prob_preds)
                        
                        progress_bar.progress((i + 1) / len(data))
                        status_text.text(f'Processing with BERT: {i + 1}/{len(data)} comments')

                    # Create results dataframe
                    binary_df = pd.DataFrame(binary_predictions)
                    result_df = pd.concat([data, binary_df], axis=1)
                    
                    if include_probabilities:
                        prob_df = pd.DataFrame(prob_predictions)
                        prob_df.columns = [f"{col}_prob" for col in prob_df.columns]
                        result_df = pd.concat([result_df, prob_df], axis=1)

                    st.success("BERT Predictions Completed!")
                    
                    # Show results preview
                    with st.expander("Preview Results"):
                        st.dataframe(result_df.head(10), use_container_width=True)

                    # Summary statistics
                    st.subheader("Summary Statistics")
                    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Comments", len(data))
                    with col2:
                        toxic_comments = (binary_df['toxic'] == 1).sum()
                        st.metric("Toxic Comments", toxic_comments)
                    with col3:
                        clean_comments = len(data) - toxic_comments
                        st.metric("Clean Comments", clean_comments)
                    with col4:
                        toxicity_rate = (toxic_comments / len(data)) * 100
                        st.metric("Toxicity Rate", f"{toxicity_rate:.1f}%")
                    
                    # Category breakdown
                    st.subheader("Category Breakdown")
                    category_counts = binary_df[labels].sum()
                    
                    # Use the fixed chart function
                    fig = create_bar_chart_with_proper_margins(
                        categories=[label.replace('_', ' ').title() for label in category_counts.index],
                        values=category_counts.values,
                        title="Toxic Comments by Category (BERT Analysis)",
                        ylabel="Number of Toxic Comments",
                        colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
                    )
                    
                    st.pyplot(fig)
                    plt.close()

                    # Download option
                    csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download BERT Predictions as CSV", 
                        csv, 
                        "bert_toxicity_predictions.csv", 
                        "text/csv",
                        type="primary",
                        help="Download the complete results with BERT binary predictions"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")

# MODEL INSIGHTS PAGE
elif current_page == 'Model Insights':
    st.header("📈 BERT Model Architecture & Performance")
    st.markdown("*Explore BERT transformer architecture, parameters, and performance metrics.*")
    
    # Get BERT model metrics
    model_info, layer_info = get_bert_model_metrics()
    
    # Model Architecture
    st.subheader("BERT Model Architecture")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Vocabulary Size", f"{model_info.get('Vocabulary Size', 'N/A'):,}")
    with col2:
        st.metric("Hidden Size", model_info.get('Hidden Size', 'N/A'))
    with col3:
        st.metric("Attention Heads", model_info.get('Number of Attention Heads', 'N/A'))
    
    # Model Details
    st.subheader("BERT Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Details:**")
        for key, value in model_info.items():
            if key not in ['Vocabulary Size', 'Hidden Size', 'Number of Attention Heads']:
                st.write(f"- **{key}:** {value}")
    
    with col2:
        st.markdown("**Architecture Components:**")
        if layer_info:
            layer_df = pd.DataFrame(layer_info)
            st.dataframe(layer_df, use_container_width=True)
    
    # Performance Analysis with Test Data
    st.subheader("BERT Performance Evaluation")
    
    if st.button("Run BERT Performance Analysis", type="primary"):
        with st.spinner("Evaluating BERT model performance..."):
            evaluation_results = evaluate_bert_performance()
        
        st.success("BERT evaluation completed!")
        
        # Overall Performance Metrics
        st.subheader("Overall BERT Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Accuracy", f"{evaluation_results['overall_accuracy']:.3f}")
        with col2:
            macro_f1 = evaluation_results['classification_report']['macro avg']['f1-score']
            st.metric("Macro F1-Score", f"{macro_f1:.3f}")
        with col3:
            weighted_f1 = evaluation_results['classification_report']['weighted avg']['f1-score']
            st.metric("Weighted F1-Score", f"{weighted_f1:.3f}")
        with col4:
            st.metric("Test Samples", len(evaluation_results['test_data']))
        
        # Per-Class Performance
        st.subheader("Per-Class Performance Metrics")
        
        # Create a detailed metrics table
        metrics_data = []
        labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        
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
        st.dataframe(metrics_df, use_container_width=True)
        
        # Visualization
        st.subheader("BERT Performance Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy by category
            categories = [item['Category'] for item in metrics_data]
            accuracies = [evaluation_results['class_metrics'][label]['accuracy'] for label in labels]
            
            fig1 = create_bar_chart_with_proper_margins(
                categories=categories,
                values=accuracies,
                title="BERT Accuracy by Category",
                ylabel="Accuracy",
                colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3'],
                figsize=(10, 8)
            )
            
            # Set y-axis limit for accuracy
            fig1.axes[0].set_ylim(0, 1.05)
            
            st.pyplot(fig1)
            plt.close()
        
        with col2:
            # F1-Score comparison
            f1_scores = [evaluation_results['class_metrics'][label]['f1_score'] for label in labels]

            fig2 = create_bar_chart_with_proper_margins(
                categories=categories,
                values=f1_scores,
                title="BERT F1-Score by Category",
                ylabel="F1-Score",
                colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3'],
                figsize=(10, 8)
            )

            # Set y-axis limit for F1-score
            fig2.axes[0].set_ylim(0, 1.05)

            st.pyplot(fig2)
            plt.close()

        # Detailed Test Results
        st.subheader("Detailed BERT Test Results")
        with st.expander("View All BERT Test Predictions vs Ground Truth"):
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
        st.subheader("BERT Model Confidence Analysis")
        
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
        st.subheader("BERT Probability Score Distribution")
        
        # Prepare data for histogram
        prob_data = []
        for i in range(len(labels)):
            prob_data.append(evaluation_results['probabilities'][:, i])
        
        fig3 = create_histogram_subplots_with_proper_margins(
            data_arrays=prob_data,
            labels=labels,
            title="BERT Probability Score Distribution by Category",
            threshold=THRESHOLD,
            figsize=(16, 12)
        )
        
        st.pyplot(fig3)
        plt.close()

# TEST CASES PAGE
elif current_page == 'Test Cases':
    st.header("🧪 Sample Test Cases with BERT")
    st.markdown("*Click on any comment below to see its BERT toxicity predictions (1 = Toxic, 0 = Non-toxic).*")
    
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
        if st.button("🔍 Analyze All Comments with BERT", type="primary"):
            st.subheader("BERT Bulk Analysis Results")
            
            all_results = []
            progress_bar = st.progress(0)
            
            for i, comment in enumerate(sample_comments):
                binary_pred = predict_toxicity_bert(comment, return_probabilities=False)
                result = {"Comment": comment[:50] + "..." if len(comment) > 50 else comment}
                result.update(binary_pred)
                result["Total_Toxic_Categories"] = sum(binary_pred.values())
                all_results.append(result)
                progress_bar.progress((i + 1) / len(sample_comments))
            
            results_df = pd.DataFrame(all_results)
            
            # Display results with color coding
            st.dataframe(results_df, use_container_width=True)
            
            # Summary
            toxic_count = (results_df['toxic'] == 1).sum()
            st.info(f"**BERT Summary:** {toxic_count}/{len(sample_comments)} comments detected as toxic")
    
    with col2:
        show_probabilities = st.checkbox("Show probability scores", key="sample_probs")
    
    st.markdown("---")
    st.subheader("🔍 Individual Comment Analysis with BERT")
    
    for i, comment in enumerate(sample_comments):
        with st.expander(f"Comment {i+1}: {comment[:60]}{'...' if len(comment) > 60 else ''}"):
            st.write(f"**Full Comment:** *{comment}*")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if st.button(f"🔍 Analyze with BERT", key=f"analyze_{i}"):
                    with st.spinner("Analyzing with BERT..."):
                        binary_preds = predict_toxicity_bert(comment, return_probabilities=False)
                        prob_preds = predict_toxicity_bert(comment, return_probabilities=True)
                    
                    # Store results in session state
                    st.session_state[f"binary_{i}"] = binary_preds
                    st.session_state[f"prob_{i}"] = prob_preds
            
            with col2:
                # Display results if they exist
                if f"binary_{i}" in st.session_state:
                    binary_preds = st.session_state[f"binary_{i}"]
                    prob_preds = st.session_state[f"prob_{i}"]
                    
                    st.write("**BERT Binary Classifications:**")
                    toxic_count = 0
                    for label, prediction in binary_preds.items():
                        if prediction == 1:
                            st.error(f"{label.replace('_', ' ').title()}: **{prediction}**")
                            toxic_count += 1
                        else:
                            st.success(f"{label.replace('_', ' ').title()}: **{prediction}**")
                    
                    if show_probabilities:
                        st.write("**BERT Probability Scores:**")
                        for label, score in prob_preds.items():
                            # Convert to Python float for display
                            score_float = float(score)
                            st.write(f"- {label.replace('_', ' ').title()}: {score_float:.3f}")
                    
                    # Overall assessment
                    if toxic_count > 0:
                        st.error(f"**TOXIC** - {toxic_count} categories detected by BERT!")
                    else:
                        st.success("**CLEAN** - No toxicity detected by BERT!")



