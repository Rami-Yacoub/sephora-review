# Sephora Reviews Analysis — NLP & Deep Learning

An interactive Natural Language Processing (NLP) application for analyzing beauty product reviews, classifying customer sentiment, discovering discussion topics, and extracting meaningful insights from textual feedback.

The project combines traditional Machine Learning, Deep Learning, topic modeling, and explainable AI in a Streamlit web application.

<p align="center">
  <a href="https://lnkd.in/p/eqP_R76p">
    <strong>▶️ Watch the Project Demo on LinkedIn</strong>
  </a>
</p>

## 📌 Project Overview

Customer reviews provide valuable information about product satisfaction, recurring concerns, consumer preferences, and cosmetic ingredients. However, manually analyzing thousands of reviews can be time-consuming, especially when working with noisy and imbalanced textual data.

This project explores a complete NLP pipeline built around **more than 59,000 Sephora product reviews**. It covers data exploration, text preprocessing, semantic analysis, sentiment classification, topic modeling, and model interpretability.

Two main sentiment classification approaches are compared:

- **SVM + TF-IDF:** a traditional Machine Learning approach based on numerical text representations.
- **Bidirectional LSTM (BiLSTM):** a Deep Learning approach designed to learn patterns from sequences of words.

The classification task uses three sentiment categories: **negative, neutral, and positive**.

The final application brings these techniques together in an interactive Streamlit interface.

## 🎯 Project Objectives

- Explore and preprocess a large collection of beauty product reviews.
- Understand the distribution and characteristics of customer feedback.
- Classify reviews into negative, neutral, and positive sentiment categories.
- Discover recurring themes using topic modeling.
- Explore semantic relationships between words.
- Extract relevant entities, including brands and cosmetic ingredients.
- Compare traditional Machine Learning with Deep Learning.
- Interpret individual predictions using explainable AI techniques.
- Build an interactive application for exploring review insights.

## ✨ Key Features

### 1. Exploratory Data Analysis

Explore the review dataset to understand its structure, distribution, and main characteristics.

The analysis helps identify patterns in customer feedback and informs the preprocessing and modeling stages.

### 2. Text Preprocessing

Prepare raw reviews for NLP tasks through text cleaning and normalization.

The project uses NLP libraries such as NLTK and spaCy to support text processing and linguistic analysis.

### 3. Sentiment Analysis

Classify reviews into three sentiment categories:

- **Negative:** unfavorable feedback or dissatisfaction.
- **Neutral:** mixed, ambiguous, or relatively neutral feedback.
- **Positive:** favorable feedback or satisfaction.

Two approaches are explored:

| Model | Description |
|---|---|
| SVM + TF-IDF | Represents text using TF-IDF features and applies a Support Vector Machine classifier |
| Bidirectional LSTM | Learns sequential patterns from text using a recurrent neural network |

The comparison considers predictive performance, computational efficiency, and interpretability.

### 4. Topic Modeling with LDA

Latent Dirichlet Allocation (LDA) is used to explore recurring topics in customer reviews.

Topic modeling helps identify groups of words associated with common discussion themes without requiring predefined topic labels.

### 5. Semantic Analysis with Word2Vec

Word2Vec is used to learn distributed representations of words and explore semantic relationships within the review corpus.

A t-SNE visualization can be used to project word representations into a lower-dimensional space for visual exploration.

### 6. Named Entity Extraction

NLP techniques are used to identify relevant entities in reviews, including brand names and cosmetic ingredients.

This feature supports the exploration of products and ingredients frequently mentioned by customers.

### 7. Explainable AI with LIME

LIME (Local Interpretable Model-agnostic Explanations) helps explain individual sentiment predictions by identifying text features that contribute to a model's output.

This provides additional insight into why a review may be classified as positive, neutral, or negative.

### 8. Interactive Streamlit Application

The web application provides an interface for entering review text and exploring the available analysis results.

Depending on the selected functionality, users can inspect sentiment predictions, topics, keywords, and detected entities.

## 🔄 Project Workflow

The project follows a multi-stage NLP workflow:

1. **Data exploration** — Examine the dataset and understand the review distribution.
2. **Text preprocessing** — Clean and prepare the reviews for downstream analysis.
3. **Linguistic and semantic analysis** — Explore entities, word representations, and recurring topics.
4. **Model training and evaluation** — Train and compare SVM + TF-IDF and BiLSTM sentiment classifiers.
5. **Model interpretation** — Use LIME to investigate individual predictions.
6. **Application development** — Integrate the analysis pipeline into a Streamlit interface.

## 🧠 Model Comparison and Findings

The project compares two approaches to three-class sentiment classification.

| Approach | Text representation | Classifier |
|---|---|---|
| Traditional Machine Learning | TF-IDF | Support Vector Machine (SVM) |
| Deep Learning | Word sequences and learned representations | Bidirectional LSTM (BiLSTM) |

The original five-class sentiment formulation was reframed into three categories: negative, neutral, and positive.

In the reported experiments, the three-class formulation improved the macro F1-score. The SVM achieved performance close to the BiLSTM while running approximately ten times faster in the reported comparison.

These observations highlight the importance of task formulation and computational efficiency when developing sentiment analysis systems.

**Evaluation metrics**

The following metrics are relevant to the comparison:

- **Accuracy:** overall proportion of correct predictions.
- **Precision:** proportion of predicted class instances that are correct.
- **Recall:** proportion of actual class instances correctly identified.
- **Macro F1-score:** average F1-score across classes, giving equal weight to each class.
- **Confusion matrix:** detailed view of correct and incorrect classifications.

Exact numerical scores are not included here because the final evaluation values have not been reproduced in this README. They should be added from the verified experimental results.

## 🛠️ Tech Stack

**Programming language**
- Python

**Machine Learning and Deep Learning**
- Scikit-learn
- TensorFlow / Keras

**Natural Language Processing**
- NLTK
- spaCy
- Gensim

**Data processing**
- NumPy
- Pandas

**Visualization and explainability**
- Matplotlib
- Seaborn
- Plotly
- WordCloud
- LIME
- t-SNE

**Web application**
- Streamlit

## 📁 Repository Structure

```text
sephora-review/
├── .streamlit/       # Streamlit configuration
├── models/           # Trained models and supporting artifacts
├── app.py            # Main Streamlit application
├── requirements.txt  # Python dependencies
├── .gitignore
└── README.md
```

The `models/` directory contains the model artifacts and supporting files required by the corresponding application features.

The exact set of artifacts depends on the models and analysis components enabled in the application.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Rami-Yacoub/sephora-review.git
cd sephora-review
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Upgrade pip and install the dependencies listed in the project:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For reproducibility, use the dependency versions specified in `requirements.txt`.

## ▶️ Run the Application

From the project root, execute:

```bash
streamlit run app.py
```

Streamlit will start a local server and provide a URL to open the application in your browser.

Follow the interface instructions to enter a beauty product review and explore the available analysis results.

Make sure the required model artifacts are present in the `models/` directory.

## 📊 Potential Applications

This project demonstrates how NLP can support:

- Large-scale customer feedback analysis.
- Product sentiment monitoring.
- Identification of recurring customer concerns.
- Exploration of product and ingredient discussions.
- Comparison of text classification approaches.
- Interpretation of sentiment model predictions.

These applications illustrate the potential of NLP for extracting structured insights from unstructured customer feedback.

## ⚠️ Limitations

- Sentiment predictions depend on the training data, preprocessing choices, and model generalization.
- Sarcasm, ambiguous expressions, mixed opinions, and domain-specific language can make classification difficult.
- Topic modeling results depend on corpus characteristics and model configuration.
- Named entity extraction may miss unfamiliar product names or ingredient variations.
- LIME explanations provide local approximations of model behavior and should not be interpreted as definitive causal explanations.
- Customer sentiment does not establish the objective quality, safety, or effectiveness of a cosmetic product.

## 👤 Author

**Rami Yacoub**

- GitHub: [@Rami-Yacoub](https://github.com/Rami-Yacoub)
- Project repository: [sephora-review](https://github.com/Rami-Yacoub/sephora-review)
- Project demonstration: [Watch on LinkedIn](https://lnkd.in/p/eqP_R76p)

---

*An NLP and Deep Learning project exploring sentiment analysis, topic modeling, semantic representations, and explainable AI for beauty product reviews.*
