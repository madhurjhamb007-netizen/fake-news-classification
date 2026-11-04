# Fake News Classification Using Machine Learning

An NLP-based machine learning project that classifies news articles as **Real** or **Fake** using the WELFake dataset. The project explores multiple machine learning algorithms, investigates dataset-specific shortcuts, and evaluates a shortcut-reduced XGBoost model.

## Project Overview

The goal of this project is to build a text classification pipeline and investigate how dataset patterns can influence model predictions.

The project includes:
- Text preprocessing and feature extraction.
- Comparison of multiple machine learning models.
- Hyperparameter tuning.
- Investigation of potential dataset shortcuts.
- Evaluation using a held-out test set.
- An inference pipeline using saved model artifacts.

## Dataset

**WELFake Dataset**

- Original records: 72,134
- Cleaned records: 62,692
- Features used: `title`, `text`, `label`
- Target labels:
  - `0` – Real
  - `1` – Fake

Missing values and duplicate records were handled during data cleaning.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- NLTK
- Matplotlib
- Joblib

## Methodology

### 1. Text Preprocessing

- Converted text to lowercase.
- Removed punctuation.
- Removed English stopwords.
- Preserved negation words such as `no`, `not`, and `nor`.
- Removed selected source-related terms in the shortcut-reduced experiment.

### 2. Feature Extraction

Used `CountVectorizer` to convert text into numerical features.

Configuration:
- `min_df = 5`
- `max_df = 0.95`

The vectorizer was fitted on training data.

### 3. Model Comparison

The following algorithms were explored:

- Multinomial Naive Bayes
- Logistic Regression
- Linear SVC
- Random Forest
- XGBoost

XGBoost was further explored using `RandomizedSearchCV` for hyperparameter optimization.

## Investigating Dataset Shortcuts

During feature-importance analysis, the term `reuters` appeared as an important feature.

Further investigation showed that approximately 97.64% of articles containing this term were labelled Real.

This suggested that the model could be learning dataset-specific source patterns rather than relying entirely on the content of an article.

To investigate this, selected terms were removed:

- `reuters`
- `nytimes`
- `via`
- `getty`

A separate shortcut-reduced model was then trained and evaluated.

This experiment does not eliminate all possible dataset biases.

## Final Model Performance

The shortcut-reduced XGBoost model achieved the following results on the held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 92.81% |
| Precision | 92.72% |
| Recall | 91.09% |
| F1-score | 91.90% |

### Confusion Matrix

| | Predicted Real | Predicted Fake |
|---|---:|---:|
| Actual Real | 4,892 | 301 |
| Actual Fake | 375 | 3,836 |

The model correctly classified 8,728 out of 9,404 test articles.

## Error Analysis

Misclassified articles were examined to understand model limitations.

Some observations:
- Certain real articles written in informal or opinion-based styles were classified as Fake.
- Some fake articles written in conventional journalistic styles were classified as Real.
- High model confidence does not necessarily mean that a prediction is factually correct.

## Repository Structure

```text
fake-news-classification/
│
├── fake_news_classification.ipynb
├── xgb_unbiased_model.pkl
├── cv_unbiased_vectorizer.pkl
└── README.md
```

## Running the Project

Install the required dependencies:

```bash
pip install pandas numpy scikit-learn xgboost nltk matplotlib joblib
```

Open `fake_news_classification.ipynb` in Jupyter Notebook or Google Colab to explore the analysis and model development.

The saved model and vectorizer can be loaded using Joblib for inference.

## Limitations

- The model learns patterns from the WELFake dataset and does not independently verify factual claims.
- Removing selected source-related terms does not guarantee the elimination of dataset bias.
- Bag-of-words features have limited ability to capture context, sarcasm, and nuanced meaning.
- The model's confidence scores should not be interpreted as probabilities that an article is factually true.
- Generalization to news from unrelated sources or newer events has not been established.

## Future Improvements

- Evaluate on an independent external dataset.
- Explore TF-IDF and n-gram features.
- Experiment with contextual language models.
- Investigate probability calibration.
- Develop a simple web interface for predictions.

## Disclaimer

This project is intended for educational and research purposes. Its predictions should not be treated as definitive judgments about the factual accuracy of news articles.
