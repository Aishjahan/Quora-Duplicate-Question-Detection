# Duplicate Question Detection on Quora

## Project Overview
Quora is a platform that connects users to share and gain knowledge by asking and answering questions. However, many questions posted on Quora are duplicates, leading to inefficiencies for both seekers and writers. To address this challenge, this project aims to develop a machine learning model to identify duplicate question pairs, enhancing the overall user experience by providing canonical questions with high-quality answers.

This project is inspired by a Kaggle competition focused on Natural Language Processing (NLP) and uses a combination of advanced techniques to classify whether two given questions have the same intent.

---

## Problem Statement
With over 100 million monthly users, Quora receives many similarly worded questions. Multiple versions of the same question can cause:
- Seekers to spend more time finding the best answers.
- Writers to feel the need to answer multiple versions of the same question.

By accurately identifying duplicate questions, Quora can:
- Improve the user experience for both seekers and writers.
- Provide high-quality answers through canonical questions.

---

## Dataset
The dataset consists of question pairs with a label indicating whether the two questions are duplicates (1) or not (0). It includes features such as:
- Question 1
- Question 2
- Label (target variable: 1 for duplicate, 0 for non-duplicate)

**Source:** Kaggle's Quora Question Pairs competition dataset.

---

## Technologies Used
### Programming Language
- **Python**

### Libraries and Frameworks
- **NLP Tools**: NLTK, scikit-learn
- **Data Manipulation**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: Random Forest, Logistic Regression, XGBoost
- **Web Interface**: Streamlit (optional for deployment)

---

## Project Workflow
1. **Exploratory Data Analysis (EDA):**
   - Analyzing question pair distributions.
   - Visualizing duplicate and non-duplicate question pair trends.

2. **Feature Engineering:**
   - Token-based similarity metrics (e.g., common tokens, token set ratio).
   - Length-based features (e.g., question lengths, length difference).
   - Fuzzy string matching.
   - Bag of Words (BoW) transformation for textual comparison.

3. **Model Training:**
   - Random Forest (baseline model used by Quora).
   - Gradient Boosting algorithms (e.g., XGBoost).
   - Hyperparameter tuning to optimize model performance.

4. **Model Evaluation:**
   - Metrics: Accuracy, F1-Score, Precision, Recall, ROC-AUC.
   - Cross-validation to avoid overfitting.

5. **Deployment (Optional):**
   - A user-friendly interface using Streamlit for predicting duplicate questions in real-time.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Aishjahan/Quora-Duplicate-Question-Detection.git
   ```

2. Navigate to the project directory:
   ```bash
   cd duplicate-question-detection
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application (if deploying with Streamlit):
   ```bash
   streamlit run app.py
   ```

---

## Usage
### Training the Model
Run the following command to train the model and save it:
```bash
python train_model.py
```

### Predicting Duplicate Questions
Use the script to input question pairs and predict duplication:
```bash
python predict.py --question1 "What is AI?" --question2 "What is Artificial Intelligence?"
```

---

## Results
The final model achieves:
- **Accuracy:** 87.5%
- **F1-Score:** 0.85
- **ROC-AUC:** 0.89

---

## Future Work
- Experiment with deep learning models like LSTMs and Transformers (e.g., BERT).
- Improve feature extraction using word embeddings such as GloVe or Word2Vec.
- Implement multilingual duplicate detection for global users.


---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- Kaggle for the Quora Question Pairs dataset.
- Open-source contributors for libraries used in this project.

---
