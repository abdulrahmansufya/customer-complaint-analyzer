import streamlit as st
import joblib
import re
from sklearn.metrics.pairwise import cosine_similarity

# Load the trained model, TF-IDF vectorizer, and similarity data
model = joblib.load("model/linear_svm_model.pkl")
tfidf = joblib.load("model/tfidf_vectorizer.pkl")
similarity_data = joblib.load("model/similarity_data.pkl")

historical_complaints = similarity_data["complaints"]
historical_products = similarity_data["products"]
historical_tfidf = similarity_data["tfidf_matrix"]

st.set_page_config(
    page_title="Customer Complaint Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Complaint Analyzer")
st.write(
    "Enter a customer complaint to predict its product category "
    "and find similar historical complaints."
)


def clean_text(text):
    # Apply the same text cleaning steps used during training
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

complaint_text = st.text_area(
    "Customer complaint",
    placeholder="Enter the customer's complaint here...",
    height=150
)

if st.button("Analyze Complaint"):
    if complaint_text.strip():
        # Clean and transform the new complaint
        cleaned_complaint = clean_text(complaint_text)
        complaint_tfidf = tfidf.transform([cleaned_complaint])

        # Predict the product category
        prediction = model.predict(complaint_tfidf)[0]

        st.subheader("Predicted Category")
        st.success(prediction)

        
        # Calculate cosine similarity with historical complaints
        similarity_scores = cosine_similarity(
            complaint_tfidf,
            historical_tfidf
        ).flatten()

        # Get the three most similar historical complaints
        top_indices = similarity_scores.argsort()[-3:][::-1]

        st.subheader("Similar Historical Complaints")

        for rank, index in enumerate(top_indices, start=1):
            with st.container(border=True):
                st.write(f"### Similar Complaint #{rank}")
                st.write(f"**Product:** {historical_products.iloc[index]}")
                st.write( f"**Similarity Score:** {similarity_scores[index]:.4f}"
        )
                st.write(historical_complaints.iloc[index])