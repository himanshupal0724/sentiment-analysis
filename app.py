
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
from text_preprocessing import preprocess_text

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "sentiment_pipeline.pkl"
)


# Load existing trained model
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()


# UI 
st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(
                circle at 10% 5%,
                #312e81 0,
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #0b1026 0%,
                #151638 48%,
                #211044 100%
            );
        color: #f3f4ff;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stAppViewContainer"] {
        background: transparent;
    }

    .block-container {
        max-width: 850px;
        padding-top: 3rem;
        padding-bottom: 2rem;
    }

    .hero {
        text-align: center;
        padding: 1.8rem 1rem 1.4rem 1rem;
    }

    .hero-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1.2;
        letter-spacing: -1px;
        color: #ffffff;
    }

    .hero-title span {
        background: linear-gradient(
            90deg,
            #a5b4fc,
            #c4b5fd,
            #67e8f9
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #c4c8e7;
        font-size: 1rem;
        line-height: 1.7;
        margin-top: 0.8rem;
    }

    .section-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(196, 181, 253, 0.22);
        border-radius: 18px;
        padding: 1.4rem;
        margin-top: 1rem;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.16);
    }

    .section-heading {
        color: #f5f3ff;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .section-description {
        color: #b9c0df;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }

    [data-testid="stTextArea"] textarea {
        background: rgba(15, 18, 45, 0.85);
        color: #ffffff;
        border: 1px solid #55518c;
        border-radius: 12px;
        caret-color: #a5b4fc;
    }

    [data-testid="stTextArea"] textarea:focus {
        border-color: #a78bfa;
        box-shadow: 0 0 0 1px #a78bfa;
    }

    [data-testid="stTextArea"] textarea::placeholder {
        color: #929bbd;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(
            90deg,
            #6366f1,
            #8b5cf6
        );
        color: white;
        border: 1px solid rgba(196, 181, 253, 0.4);
        border-radius: 11px;
        min-height: 3rem;
        font-weight: 700;
        transition: 0.2s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        border-color: #c4b5fd;
        filter: brightness(1.12);
        transform: translateY(-1px);
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(196, 181, 253, 0.2);
        border-radius: 13px;
        padding: 1rem;
    }

    div[data-testid="stMetricLabel"] {
        color: #c4c8e7;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }

    [data-testid="stCaptionContainer"] {
        color: #b9c0df;
    }

    .footer {
        text-align: center;
        color: #969fc4;
        font-size: 0.82rem;
        padding-top: 1.5rem;
    }

    hr {
        border-color: rgba(196, 181, 253, 0.2);
    }
</style>
""", unsafe_allow_html=True)


# Hero section
st.markdown("""
<div class="hero">
    <div class="hero-icon">💬</div>
    <div class="hero-title">
        AI-Powered<br><span>Sentiment Analysis</span>
    </div>
    <div class="hero-subtitle">
        Understand customer opinions with Natural Language
        Processing and Machine Learning.
        <br>
        Analyze a review to identify its sentiment.
    </div>
</div>
""", unsafe_allow_html=True)


# Review input
st.markdown("""
<div class="section-card">
    <div class="section-heading">✍️ Analyze a customer review</div>
    <div class="section-description">
        Enter your review below. The trained model will predict
        Positive, Negative, or Neutral sentiment.
    </div>
</div>
""", unsafe_allow_html=True)

review = st.text_area(
    "Customer review",
    placeholder=(
        "Example: The product quality is excellent "
        "and delivery was fast!"
    ),
    height=160,
    max_chars=10000,
    label_visibility="collapsed"
)

st.caption(f"{len(review)} / 10,000 characters")

# Prediction
if st.button(
    "✨  Analyze Sentiment",
    type="primary",
    use_container_width=True
):
    if not review.strip():
        st.warning("Please enter a review first.")

    else:
        try:
            cleaned_review = preprocess_text(review)

            if not cleaned_review:
                st.warning(
                    "Please enter a review containing usable text."
                )
                st.stop()

            prediction = model.predict([cleaned_review])[0]

            st.divider()
            st.markdown(
                '<div class="section-heading">📊 Analysis Result</div>',
                unsafe_allow_html=True
            )

            if prediction == "Positive":
                st.success("😊  Sentiment: Positive")
            elif prediction == "Negative":
                st.error("😞  Sentiment: Negative")
            elif prediction == "Neutral":
                st.info("😐  Sentiment: Neutral")
            else:
                st.write(f"Predicted sentiment: {prediction}")

            # Show probabilities when supported by the model
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(
                    [cleaned_review]
                )[0]

                classes = model.classes_

                probability_df = pd.DataFrame({
                    "Sentiment": classes,
                    "Probability": probabilities
                }).sort_values(
                    "Probability",
                    ascending=False
                )

                st.markdown(
                    '<div class="section-heading">'
                    '📈 Prediction Probabilities</div>',
                    unsafe_allow_html=True
                )

                col1, col2 = st.columns([1, 2])

                top_probability = float(
                    probability_df.iloc[0]["Probability"]
                )

                with col1:
                    st.metric(
                        "Highest class probability",
                        f"{top_probability:.1%}"
                    )

                with col2:
                    st.bar_chart(
                        probability_df.set_index("Sentiment")
                    )

                st.caption(
                    "Probabilities are model estimates, not guaranteed "
                    "measures of prediction correctness."
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# Footer
st.markdown("""
<div class="footer">
    Academic Project &nbsp;•&nbsp; NLP &nbsp;•&nbsp;
    TF-IDF &nbsp;•&nbsp; Machine Learning
</div>
""", unsafe_allow_html=True)
