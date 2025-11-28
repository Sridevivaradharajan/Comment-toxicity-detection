# Comment Toxicity Classification

This project is a **Deep learning based web application** that classifies comments into different categories of toxicity. It helps detect and mitigate harmful online behavior such as hate speech, threats, and offensive language.

---

## Features
- Classifies comments into multiple toxicity categories:
  - Toxic
  - Severe Toxic
  - Obscene
  - Threat
  - Insult
  - Identity Hate
- Supports multiple models for comparison:
  - **LSTM**
  - **BiLSTM**
  - **BERT (Transformers)**
  - **RoBERTa (Transformers)**
- Interactive **Streamlit web app** for real-time predictions
- Performance visualization (Accuracy, F1 Score, Subset Accuracy, Micro-average, etc.)
- Deployment ready (Streamlit Cloud)

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Sridevivaradharajan/Comment-toxicity-detection.git
cd comment-toxicity
````

2. Create a virtual environment and install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```
---

## Model Ranking & Performance

Based on evaluation across multiple metrics, the models were ranked as follows:

| **Rank** | **Model**   | **Micro-Avg F1 (Overall)** | **Macro-Avg F1 (Balanced)** | **Performance on Critical Labels (threat & identity\_hate)** |
| -------- | ----------- | -------------------------- | --------------------------- | ------------------------------------------------------------ |
| 🥇 1st   | **BERT**    | **0.77**                   | **0.65**                    | Best performance on both labels.                             |
| 🥈 2nd   | **RoBERTa** | **0.77**                   | **0.60**                    | Strong performance on both labels.                           |
| 🥉 3rd   | **BiLSTM**  | **0.75**                   | **0.45**                    | Weak on *threat* (0.00) and struggles with *identity\_hate*. |
| 4th      | **LSTM**    | **0.77**                   | **0.44**                    | Complete failure on both labels (0.00).                      |

---

## Tech Stack

* **Python 3.10**
* **TensorFlow / Keras**
* **Hugging Face Transformers**
* **Scikit-learn**
* **Streamlit**

