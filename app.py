import streamlit as st
import pandas as pd
import time
from src.inference import ToxicityPredictor
import os

st.set_page_config(page_title="Toxicity Detector", page_icon="🛡️", layout="wide")

# Custom CSS for aesthetics
st.markdown("""
<style>
    .toxic-high { color: #ff4b4b; font-weight: bold; }
    .toxic-med { color: #ffa421; font-weight: bold; }
    .toxic-low { color: #00c04b; font-weight: bold; }
    .metric-container {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    predictor = ToxicityPredictor()
    success = predictor.load()
    return predictor, success

predictor, is_loaded = load_predictor()

st.title("🛡️ Comment Toxicity Detection")
st.markdown("Analyze online comments for various forms of toxicity including hate speech, insults, and threats using Deep Learning.")

if not is_loaded:
    st.error("""
    **⚠️ Model files not found!** 
    Please follow these steps:
    1. Upload `train.csv` and `colab_training.py` to Google Colab.
    2. Run the script on a GPU runtime to train the model.
    3. Download `toxicity_model.h5` and `vectorizer.pkl`.
    4. Place them in the `Project-2` folder.
    """)
    st.stop()

tab1, tab2 = st.tabs(["💬 Single Comment Analysis", "📁 Bulk CSV Analysis"])

with tab1:
    st.subheader("Analyze a single comment")
    user_input = st.text_area("Enter the comment text here:", height=150, placeholder="e.g. I completely disagree with your point of view.")
    
    if st.button("Analyze Comment", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter a comment to analyze.")
        else:
            with st.spinner("Analyzing..."):
                start_time = time.time()
                results = predictor.predict(user_input)
                latency = time.time() - start_time
                
                st.success(f"Analysis complete in {latency:.2f} seconds!")
                
                st.markdown("### Prediction Results")
                cols = st.columns(3)
                
                for idx, (label, score) in enumerate(results.items()):
                    with cols[idx % 3]:
                        # Format label for display
                        display_label = label.replace('_', ' ').title()
                        
                        # Determine color class
                        if score > 0.7:
                            color_class = "toxic-high"
                        elif score > 0.4:
                            color_class = "toxic-med"
                        else:
                            color_class = "toxic-low"
                            
                        st.markdown(f"""
                        <div class="metric-container">
                            <h4>{display_label}</h4>
                            <h2 class="{color_class}">{score:.2%}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                
with tab2:
    st.subheader("Bulk Toxicity Analysis")
    uploaded_file = st.file_uploader("Upload a CSV file containing a 'comment_text' column", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'comment_text' not in df.columns:
                st.error("The CSV file must contain a 'comment_text' column.")
            else:
                st.write(f"Loaded {len(df)} rows. Here is a preview:")
                st.dataframe(df.head())
                
                if st.button("Process Bulk Upload"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    predictions = {label: [] for label in predictor.labels}
                    
                    # Process in chunks to keep UI responsive
                    total = len(df)
                    for i, text in enumerate(df['comment_text'].fillna('')):
                        res = predictor.predict(text)
                        for label in predictor.labels:
                            predictions[label].append(res[label])
                            
                        if i % 10 == 0 or i == total - 1:
                            progress = (i + 1) / total
                            progress_bar.progress(progress)
                            status_text.text(f"Processed {i + 1}/{total} comments...")
                            
                    for label in predictor.labels:
                        df[label] = predictions[label]
                        
                    st.success("Bulk processing complete!")
                    st.dataframe(df.head(10))
                    
                    # Provide download link
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name='toxicity_predictions.csv',
                        mime='text/csv',
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")
