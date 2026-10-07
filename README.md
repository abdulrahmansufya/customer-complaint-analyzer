# Customer Complaint Analyzer

A machine learning system for analyzing customer complaints, predicting
the complaint category, and retrieving similar historical complaints.

## Project Overview

This project analyzes customer complaint messages using Natural Language
Processing (NLP) and Machine Learning.

The system provides two main functions:

1.  **Complaint Classification**
    -   Predicts the product/category related to a customer complaint.
    -   Uses TF-IDF text features and a Linear SVM classifier.
2.  **Similar Complaint Search**
    -   Finds the three most similar historical complaints.
    -   Uses TF-IDF vectors and cosine similarity.

A simple Streamlit web application is included to interact with the
trained system.

## Dataset

The project uses the **Consumer Complaints** dataset from the Consumer
Financial Protection Bureau (CFPB).

The dataset contains customer complaints related to financial products
and services.

The main fields used in this project are:

-   `Consumer complaint narrative` --- customer complaint text.
-   `Product` --- target category used for classification.

The original dataset contains approximately 900,000 records, with around
200,000 complaints containing narrative text.

## Project Workflow

1.  Load and inspect the dataset.
2.  Perform exploratory data analysis (EDA).
3.  Analyze missing values, duplicates, class distribution, and
    complaint length.
4.  Clean and normalize complaint text.
5.  Remove very short and exact duplicate complaint records.
6.  Analyze conflicting labels.
7.  Split the data into training and testing sets using stratified group
    splitting.
8.  Convert complaint text into TF-IDF features.
9.  Train and compare Logistic Regression, Multinomial Naive Bayes, and
    Linear SVM.
10. Evaluate the models using Accuracy, Precision, Recall, F1-score, and
    Confusion Matrix.
11. Perform error analysis on misclassified complaints.
12. Implement similar complaint search using cosine similarity.
13. Test the complete system on unseen complaints.
14. Save the trained model and TF-IDF vectorizer.
15. Build a Streamlit web application.

## Model Performance

  Model                       Test Accuracy   Test Weighted F1   Test Macro F1
  ------------------------- --------------- ------------------ ---------------
  Logistic Regression                78.98%             79.17%          56.18%
  Multinomial Naive Bayes            76.78%             74.94%          44.40%
  **Linear SVM**                 **81.14%**         **80.67%**          55.26%

Linear SVM was selected as the final classification model because it
achieved the highest overall testing accuracy and weighted F1-score.

Macro F1 was also considered because the dataset contains imbalanced
product categories.

## Similarity Search

The system uses TF-IDF vectorization and cosine similarity to retrieve
the three most similar historical complaints.

Each result displays:

-   Complaint text
-   Product category
-   Similarity score

Higher cosine similarity scores indicate greater textual similarity.

## Streamlit Application

The application allows users to:

1.  Enter a customer complaint.
2.  Predict its product category.
3.  Retrieve the three most similar historical complaints.
4.  View the similarity scores.

## Project Structure

``` text
customer-complaint-analyzer/
├── app.py
├── customer_complaint_analysis.ipynb
├── Consumer_Complaints.csv
├── requirements.txt
├── .gitignore
├── .gitattributes
└── model/
    ├── linear_svm_model.pkl
    ├── tfidf_vectorizer.pkl
    └── similarity_data.pkl
```

## Requirements

-   Python 3.10+
-   Git
-   Git LFS

The project dependencies are listed in `requirements.txt`.

## Installation and Setup

### 1. Clone the repository

``` bash
git clone https://github.com/abdulrahmansufya/customer-complaint-analyzer.git
cd customer-complaint-analyzer
```

### 2. Download Git LFS files

Make sure Git LFS is installed.

``` bash
git lfs install
git lfs pull
```

The large dataset and similarity data are stored using Git LFS. After
`git lfs pull`, they are available as normal files in the project
directory.

### 3. Create a virtual environment

Linux/macOS:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

``` powershell
python -m venv .venv
.venv\Scripts\activate
```

### 4. Install dependencies

``` bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

``` bash
streamlit run app.py
```

The application will normally be available at:

``` text
http://localhost:8501
```

## Running the Notebook

Open:

``` text
customer_complaint_analysis.ipynb
```

Make sure the dataset is available in the project directory before
running the notebook.

## Example

Example input:

``` text
I do not recognize this debt and a collection agency keeps contacting me.
```

Example prediction:

``` text
Debt collection
```

The application also returns the three most similar historical
complaints with their similarity scores.

## Error Analysis

The final Linear SVM model achieved a misclassification rate of
approximately 18.86% on the testing set.

Most classification errors occurred between closely related financial
categories, such as:

-   Credit reporting and related credit-report categories
-   Credit reporting and Debt collection
-   Credit card and prepaid card
-   Checking or savings account and Bank account or service

These errors are understandable because some complaints discuss multiple
related financial issues.

## Limitations

The system uses TF-IDF and therefore relies mainly on textual patterns
and word-level features.

Some limitations include:

-   Difficulty distinguishing semantically similar categories.
-   Lower performance on minority classes.
-   Some complaints may involve multiple financial issues.
-   Cosine similarity measures textual similarity rather than deep
    semantic similarity.
-   The dataset is highly imbalanced across product categories.

## Possible Future Improvements

Potential improvements include:

-   Using transformer-based language models for text classification.
-   Using sentence embeddings for semantic similarity search.
-   Applying more advanced class-imbalance techniques.
-   Exploring hierarchical classification.
-   Improving the similarity search with semantic embeddings.
-   Adding confidence scores to predictions.
-   Deploying the application as a public web service.

## Author

**Abdulrahman Al-Sufyani**

GitHub: https://github.com/abdulrahmansufya
