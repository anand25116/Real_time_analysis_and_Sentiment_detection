import streamlit as st
import pandas as pd
import time
import joblib
import requests
from PIL import Image
from io import BytesIO
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
)

dark_css = """
<style>
[data-testid="stAppViewContainer"] {background-color: #0e1117; color: #FFFFFF;}
[data-testid="stHeader"] {background-color: #0e1117; color: #FFFFFF;}
[data-testid="stToolbar"] {background-color: #0e1117; color: #FFFFFF;}
.stContainer {background-color: #1c1f26; color: #FFFFFF; border-radius: 10px; padding: 10px;}
.st-expander {background-color: #1c1f26; color: #FFFFFF; border-radius: 5px;}
.css-1d391kg p, .css-1d391kg span, .css-1d391kg div {color: #FFFFFF;}
img {display: block; margin-left: auto; margin-right: auto;}
input[type="range"] {background: #4CAF50;}
.css-1aumxhk, .css-1lcbmhc {color: #FFFFFF;}
[data-testid="stExpander"] > div > div > button {background-color: #1c1f26 !important; color: #FFFFFF !important; border-radius: 5px; padding: 6px 12px; font-weight: bold; text-align: left;}
[data-testid="stExpander"] > div > div > button:hover {background-color: #4CAF50 !important; color: #000000 !important;}
.stButton button {background-color: #333846; color: #FFFFFF; border-radius: 8px; padding: 8px 16px; border: none; transition: all 0.2s ease;}
.stButton button:hover {background-color: #4CAF50; color: #000000;}
.stButton button:focus {outline: none;}
.positive {color: #00FF7F; font-weight: bold;}
.neutral {color: #FFD700; font-weight: bold;}
.negative {color: #FF4500; font-weight: bold;}
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    # BERT
    bert_tokenizer = AutoTokenizer.from_pretrained("models/bert_sentiment")
    bert_model = AutoModelForSequenceClassification.from_pretrained("models/bert_sentiment")
    bert_pipeline = TextClassificationPipeline(model=bert_model, tokenizer=bert_tokenizer, return_all_scores=True)
    
    # Roberta
    roberta_tokenizer = AutoTokenizer.from_pretrained("models/roberta_sentiment")
    roberta_model = AutoModelForSequenceClassification.from_pretrained("models/roberta_sentiment")
    roberta_pipeline = TextClassificationPipeline(model=roberta_model, tokenizer=roberta_tokenizer, return_all_scores=True)
    
    # TF-IDF + Logistic Regression
    logreg_model = joblib.load("models/logreg.pkl")
    tfidf_vectorizer = joblib.load("models/tfidf.pkl")
    baseline_model = {"model": logreg_model, "vectorizer": tfidf_vectorizer}
    
    return bert_pipeline, roberta_pipeline, baseline_model, logreg_model, tfidf_vectorizer

bert_pipeline, roberta_pipeline, baseline_model, logreg_model, tfidf_vectorizer = load_models()


label_map = {0: "negative", 1: "neutral", 2: "positive"}
label_colors = {"negative": "#FF6B6B", "neutral": "#FFA500", "positive": "#4CAF50"}

st.set_page_config(page_title="Amazon Real-Time Review Simulator", layout="wide")
st.title("🛒 Amazon Review Sentiment Simulator (Search & Analyze)")

st.markdown("""
**Legend & Model Explanation:**  
- **BERT:** Transformer-based deep learning model analyzing context in review text.  
- **Roberta:** Transformer model similar to BERT, optimized for sentiment tasks.  
- **Baseline:** Logistic Regression with TF-IDF features.  

**Color Coding of Sentiment:**  
- <span style='color:#4CAF50'>Positive</span>  
- <span style='color:#FFA500'>Neutral</span>  
- <span style='color:#FF6B6B'>Negative</span>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload amazon_sales.csv", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {len(df)} reviews from CSV!")


    product_search = st.text_input("Search for a product (partial or full name):").strip()
    
    if product_search:
        df_filtered = df[df['product_name'].str.contains(product_search, case=False, na=False)]
        if df_filtered.empty:
            st.warning(f"No products found matching '{product_search}'. Showing all products.")
            df_filtered = df.copy()
        else:
            st.info(f"Found {len(df_filtered)} reviews for products matching '{product_search}'.")
    else:
        df_filtered = df.copy()
    

    interval = st.slider("Streaming interval (seconds per review):", 1, 10, 2)
    num_reviews_to_stream = st.slider("Number of reviews to stream:", 10, min(500, len(df_filtered)), 50, 50)
    
    start_sim = st.button("Start Simulation")
    
    if start_sim:
        st.info("Simulation started...")
        placeholder = st.empty()
        progress_bar = st.progress(0)
        reviews_list = df_filtered.sample(n=num_reviews_to_stream, replace=True).to_dict(orient="records")
        
        for i, row in enumerate(reviews_list):
            review_text = str(row.get("review_content", "")) or str(row.get("review_title", ""))
            product_name = row.get("product_name", "Unknown Product")
            rating = row.get("rating", "N/A")
            img_link = row.get("img_link", None)
            
            
            bert_pred = bert_pipeline(review_text, truncation=True, max_length=512)
            bert_scores = {d['label']: d['score'] for d in bert_pred[0]}
            bert_class = label_map[int(max(bert_scores, key=lambda k: bert_scores[k]).split("_")[1])]
            
            roberta_pred = roberta_pipeline(review_text, truncation=True, max_length=512)
            roberta_scores = {d['label']: d['score'] for d in roberta_pred[0]}
            roberta_class = label_map[int(max(roberta_scores, key=lambda k: roberta_scores[k]).split("_")[1])]
            
            rev_vec = baseline_model['vectorizer'].transform([review_text])
            baseline_pred = baseline_model['model'].predict(rev_vec)[0]
            baseline_class = label_map[baseline_pred]
            
            review_vec = tfidf_vectorizer.transform([review_text])
            logreg_pred = logreg_model.predict(review_vec)[0]
            logreg_class = label_map[logreg_pred]
            
            
            with placeholder.container():
                st.markdown("---")
                st.subheader(f"Review {i+1} - {product_name} (Rating: {rating})")
                
                col1, col2 = st.columns([1, 3])
                
                
                with col1:
                    if img_link and str(img_link).startswith(("http://", "https://")):
                        try:
                            response = requests.get(img_link, timeout=3)
                            img = Image.open(BytesIO(response.content))
                            st.image(img, width=200)
                        except:
                            st.write("🖼️ Image cannot be loaded")
                    else:
                        st.write("🖼️ No Image Available")
                
                
                with col2:
                    with st.expander("View Review Content"):
                        st.write(review_text)
                    
                    st.markdown(
                        f"**Predictions:**  \n"
                        f"- BERT: <span style='color:{label_colors[bert_class]};font-weight:bold'>{bert_class}</span>  \n"
                        f"- Roberta: <span style='color:{label_colors[roberta_class]};font-weight:bold'>{roberta_class}</span>  \n"
                        f"- Baseline: <span style='color:{label_colors[baseline_class]};font-weight:bold'>{baseline_class}</span>",
                        unsafe_allow_html=True
                    )
            
            progress_bar.progress((i + 1) / num_reviews_to_stream)
            time.sleep(interval)
        
        st.success("🎉 Simulation completed!")
else:
    st.warning("Please upload the amazon_sales.csv file to start simulation.")
