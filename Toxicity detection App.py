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
import gdown
import json
from tensorflow.keras.models import load_model


# ---------------------------
# Load Model and Tokenizer
# ---------------------------
@st.cache_resource
def load_bilstm_model():
    model_path = "bilstm_model.h5"

    # Download from Google Drive if not already downloaded
    if not os.path.exists(model_path):
        with st.spinner("Downloading BiLSTM model... Please wait."):
            url = "https://drive.google.com/file/d/1kTfFlFAfCAiUdZO5MgCF0N0uxVNd0ZZn/view?usp=sharing" 
            gdown.download(url, model_path, quiet=False)

    return load_model(model_path)

@st.cache_resource
def load_tokenizer():
    tokenizer_path = "tokenizer.pkl"
    
    # Add your tokenizer Google Drive URL here
    tokenizer_url = "https://drive.google.com/file/d/1psCM-sISb3ToTc6IYhhw3nSLWqaTVAJm/view?usp=sharing" 
    
    # Download from Google Drive if not already downloaded
    if not os.path.exists(tokenizer_path) and tokenizer_url != "https://drive.google.com/file/d/1psCM-sISb3ToTc6IYhhw3nSLWqaTVAJm/view?usp=sharing":
        with st.spinner("Downloading tokenizer... Please wait."):
            file_id = tokenizer_url.split('/d/')[1].split('/')[0]
            download_url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(download_url, tokenizer_path, quiet=False)
    
    try:
        with open(tokenizer_path, "rb") as handle:
            tokenizer = pickle.load(handle)
        return tokenizer
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        return None

# Load model and tokenizer
model = load_bilstm_model()
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
                "Parameters": layer.count_params(),
                "Input Shape": format_shape(getattr(layer, "input_shape", None)),
                "Output Shape": format_shape(getattr(layer, "output_shape", None)),
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
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Toxic Comment Classifier", layout="wide")
st.title("📝 Toxic Comment Detection App")
st.markdown("*Real-time toxicity detection with binary classification (1 = Toxic, 0 = Non-toxic)*")
st.markdown("---")

# Sidebar
st.sidebar.header("🧭 Navigation")
page = st.sidebar.radio("Go to", ["Real-time Prediction", "Bulk Prediction", "Model Insights & Metrics", "Sample Test Cases"])

# Sidebar model info
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Model Settings")
st.sidebar.write(f"**Threshold:** {THRESHOLD}")
st.sidebar.write(f"**Max Sequence Length:** {MAX_LEN}")

# ---------------------------
# 1. Real-time Prediction
# ---------------------------
if page == "Real-time Prediction":
    st.header("💬 Enter a Comment for Prediction")
    
    user_input = st.text_area("Type a comment below:", height=100, placeholder="Enter your comment here...")
    
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        predict_button = st.button("🔍 Predict", type="primary")
    with col2:
        show_probabilities = st.checkbox("Show Probabilities")
    
    if predict_button:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a valid comment.")
        else:
            with st.spinner("Analyzing comment..."):
                # Get binary predictions
                binary_result = predict_toxicity(user_input, return_probabilities=False)
                # Get probabilities if requested
                prob_result = predict_toxicity(user_input, return_probabilities=True)
            
            st.subheader("🎯 Prediction Results:")
            
            # Create columns for better layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Binary Classifications:**")
                for label, prediction in binary_result.items():
                    if prediction == 1:
                        st.error(f"🚨 {label.replace('_', ' ').title()}: **{prediction}** (TOXIC)")
                    else:
                        st.success(f"✅ {label.replace('_', ' ').title()}: **{prediction}** (NON-TOXIC)")
            
            if show_probabilities:
                with col2:
                    st.markdown("**Probability Scores:**")
                    for label, score in prob_result.items():
                        st.write(f"**{label.replace('_', ' ').title()}:** {score:.3f}")
                        # Convert to Python float and ensure it's between 0 and 1
                        normalized_score = max(0.0, min(1.0, float(score)))
                        st.progress(normalized_score)
            
            # Overall toxicity indicator
            toxic_count = sum(binary_result.values())
            if toxic_count > 0:
                st.error(f"⚠️ **TOXIC CONTENT DETECTED** - {toxic_count} toxic categories identified!")
            else:
                st.success("✅ **CLEAN CONTENT** - No toxicity detected!")

# ---------------------------
# 2. Bulk Prediction
# ---------------------------
elif page == "Bulk Prediction":
    st.header("📂 Upload CSV for Bulk Predictions")
    st.markdown("*Upload a CSV file with a 'comment_text' column to get binary toxicity predictions (1/0) for all comments.*")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], help="CSV must contain a column named 'comment_text'")
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            
            if "comment_text" not in data.columns:
                st.error("❌ CSV must have a column named 'comment_text'")
                st.info("Available columns: " + ", ".join(data.columns.tolist()))
            else:
                st.success(f"✅ File uploaded successfully! Found **{len(data)}** rows.")
                
                with st.expander("👀 Preview Data"):
                    st.dataframe(data.head(10))

                col1, col2 = st.columns([1, 1])
                with col1:
                    include_probabilities = st.checkbox("Include probability scores", help="Add probability columns alongside binary predictions")
                
                if st.button("🚀 Run Bulk Predictions", type="primary"):
                    # Predictions with progress bar
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
                        status_text.text(f'Processing: {i + 1}/{len(data)} comments')

                    # Create results dataframe
                    binary_df = pd.DataFrame(binary_predictions)
                    result_df = pd.concat([data, binary_df], axis=1)
                    
                    if include_probabilities:
                        prob_df = pd.DataFrame(prob_predictions)
                        prob_df.columns = [f"{col}_prob" for col in prob_df.columns]
                        result_df = pd.concat([result_df, prob_df], axis=1)

                    st.success("🎉 Predictions Completed!")
                    
                    # Show results preview
                    with st.expander("👀 Results Preview"):
                        st.dataframe(result_df.head(10))

                    # Summary statistics
                    st.subheader("📊 Summary Statistics")
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
                    st.subheader("🏷️ Category Breakdown")
                    category_counts = binary_df[labels].sum()
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(category_counts.index, category_counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3'])
                    ax.set_title("Toxic Comments by Category")
                    ax.set_ylabel("Number of Toxic Comments")
                    plt.xticks(rotation=45)
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                               f'{int(height)}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Download option
                    csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Predictions as CSV", 
                        csv, 
                        "toxicity_predictions.csv", 
                        "text/csv",
                        type="primary",
                        help="Download the complete results with binary predictions"
                    )
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# ---------------------------
# 3. Model Insights & Metrics
# ---------------------------
elif page == "Model Insights & Metrics":
    st.header("📊 Model Insights & Performance Metrics")
    st.markdown("*Explore model architecture, parameters, and performance metrics extracted directly from the trained model.*")
    
    # Get model metrics
    model_info, layer_info = get_model_metrics()
    
    # Model Architecture
    st.subheader("🏗️ Model Architecture")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Parameters", f"{model_info.get('Total Parameters', 'N/A'):,}")
    with col2:
        st.metric("Trainable Parameters", f"{model_info.get('Trainable Parameters', 'N/A'):,}")
    with col3:
        st.metric("Number of Layers", model_info.get('Number of Layers', 'N/A'))
    
    # Model Details
    st.subheader("ℹ️ Model Details")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Configuration:**")
        for key, value in model_info.items():
            if key not in ['Total Parameters', 'Trainable Parameters', 'Number of Layers']:
                st.write(f"- **{key}:** {value}")
    
    with col2:
        st.write("**Layer Architecture:**")
        if layer_info:
            layer_df = pd.DataFrame(layer_info)
            st.dataframe(layer_df, use_container_width=True)
    
    # Performance Analysis with Test Data
    st.subheader("🎯 Model Performance Analysis")
    
    if st.button("🧪 Run Performance Evaluation", type="primary"):
        with st.spinner("Evaluating model performance..."):
            evaluation_results = evaluate_model_performance()
        
        st.success("Evaluation completed!")
        
        # Overall Performance Metrics
        st.subheader("📊 Overall Performance")
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
        st.subheader("🏷️ Per-Class Performance Metrics")
        
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
elif page == "Sample Test Cases":
    st.header("🧪 Sample Test Cases")
    st.markdown("*Click on any comment below to see its binary toxicity predictions (1 = Toxic, 0 = Non-toxic).*")
    
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
            st.subheader("📊 Bulk Analysis Results")
            
            all_results = []
            for comment in sample_comments:
                binary_pred = predict_toxicity(comment, return_probabilities=False)
                result = {"Comment": comment[:50] + "..." if len(comment) > 50 else comment}
                result.update(binary_pred)
                result["Total_Toxic_Categories"] = sum(binary_pred.values())
                all_results.append(result)
            
            results_df = pd.DataFrame(all_results)
            
            # Color code the results
            def highlight_toxic(val):
                if isinstance(val, int) and val == 1:
                    return 'background-color: #ffcccc'
                return ''
            
            styled_df = results_df.style.applymap(highlight_toxic, subset=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Summary
            toxic_count = (results_df['toxic'] == 1).sum()
            st.info(f"📈 **Summary:** {toxic_count}/{len(sample_comments)} comments detected as toxic")
    
    with col2:
        show_probabilities = st.checkbox("Show probability scores", key="sample_probs")
    
    st.markdown("---")
    st.subheader("🔍 Individual Comment Analysis")
    
    for i, comment in enumerate(sample_comments):
        with st.expander(f"Comment {i+1}: {comment[:60]}{'...' if len(comment) > 60 else ''}"):
            st.write(f"**Full Comment:** *{comment}*")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if st.button(f"Analyze", key=f"analyze_{i}"):
                    with st.spinner("Analyzing..."):
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
                    
                    st.write("**Binary Classifications:**")
                    toxic_count = 0
                    for label, prediction in binary_preds.items():
                        if prediction == 1:
                            st.error(f"🚨 {label.replace('_', ' ').title()}: **{prediction}**")
                            toxic_count += 1
                        else:
                            st.success(f"✅ {label.replace('_', ' ').title()}: **{prediction}**")
                    
                    if show_probabilities:
                        st.write("**Probability Scores:**")
                        for label, score in prob_preds.items():
                            # Convert to Python float for display
                            score_float = float(score)
                            st.write(f"- {label.replace('_', ' ').title()}: {score_float:.3f}")
                    
                    # Overall assessment
                    if toxic_count > 0:
                        st.error(f"⚠️ **TOXIC** - {toxic_count} categories detected!")
                    else:
                        st.success("✅ **CLEAN** - No toxicity detected!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Toxic Comment Detection App</strong></p>
    <p>Built with Streamlit and TensorFlow | Binary Classification System</p>
    <p><em>Threshold: {threshold} | Max Length: {max_len}</em></p>
</div>
""".format(threshold=THRESHOLD, max_len=MAX_LEN), unsafe_allow_html=True)








