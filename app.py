import streamlit as st
import pandas as pd
import os
from src import model
import streamlit.components.v1 as components

# ---------- THEME & STYLING ----------
st.set_page_config(page_title="Moviegrad — Movie Predictor", layout="wide", page_icon="🎬")

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

/* Soft gradient background */
.stApp {
    background: linear-gradient(135deg, #19232e, #243447, #1b2838);
}

/* Main content card */
.block-container {
    background: rgba(255, 255, 255, 0.07);
    padding: 28px;
    border-radius: 14px;
    margin-top: 20px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 20px rgba(0,0,0,0.35);
}

/* Title */
h1 {
    color: #66d9ff !important;
    font-weight: 700;
    text-shadow: 0px 0px 6px rgba(102,217,255,0.5);
}

/* Subtitles */
h2, h3 {
    color: #d6eaff !important;
}

/* Label text */
label, p, .css-16idsys, .css-1y4p8pa {
    color: #e6eef4 !important;
    font-size: 16px;
}

/* Inputs: clean & bright */
.stTextInput > div > div,
.stSelectbox > div > div,
.stNumberInput > div > div,
.stFileUploader > div {
    background-color: rgba(255,255,255,0.92);
    color: #000;
    border-radius: 8px;
    padding: 6px;
    border: 1px solid #ccc;
}

/* Slider color */
.stSlider > div[data-baseweb="slider"] > div > div {
    background: #1fa3ff !important;
}

/* Radio inputs */
.stRadio > div {
    background-color: rgba(255,255,255,0.1);
    padding: 8px;
    border-radius: 10px;
}

/* Clean premium buttons */
.stButton>button {
    background: linear-gradient(90deg, #1fa3ff, #007acc);
    color: white;
    padding: 10px 25px;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    font-size: 15px;
    transition: 0.2s ease-in-out;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #3fb6ff, #0088dd);
    transform: translateY(-2px);
}

/* Table readability */
.dataframe {
    background-color: rgba(255,255,255,0.12);
    color: #e9e9e9;
}

/* Prediction result boxes */
.success-box {
    background: rgba(0,255,120,0.12);
    padding: 12px;
    color: #00e676;
    border-left: 4px solid #00e676;
    border-radius: 8px;
}

.error-box {
    background: rgba(255,0,70,0.12);
    padding: 12px;
    color: #ff5570;
    border-left: 4px solid #ff5570;
    border-radius: 8px;
}

/* Sidebar soft dark */
.css-1d391kg, .css-1cypcdb {
    background-color: #162029 !important;
    color: #ffffff !important;
}

</style>
""", unsafe_allow_html=True)


# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("⚙ Setup")

    if st.button("Generate Dataset (100k)"):
        import generate_dataset
        generate_dataset.main()
        st.success("Dataset Created: data/movies_100k.csv")

    if st.button("Train Model"):
        if os.path.exists("data/movies_100k.csv"):
            df = pd.read_csv("data/movies_100k.csv")
            with st.spinner("Training Model..."):
                model.train_model(df)
            st.success("Model Training Completed!")
        else:
            st.error("Dataset not found.")

    if st.button("Load Model"):
        try:
            model.load_model()
            st.success("Model Loaded!")
        except Exception as e:
            st.error(e)

# ---------- MAIN INPUT FORM ----------
st.header("🎬 Enter Movie Details")

genre = st.selectbox("Genre", ['Action','Comedy','Drama','Thriller','Horror','Romance','Sci-Fi','Fantasy'])
budget = st.number_input("Budget (Million $)", 1.0, 500.0, 50.0)
director_score = st.slider("Director Score", 0.0, 10.0, 6.0)
cast_popularity = st.slider("Cast Popularity", 0.0, 100.0, 50.0)
runtime = st.slider("Runtime (Minutes)", 80, 180, 120)
marketing_spend = st.number_input("Marketing Spend (Million $)", 1.0, 100.0, 10.0)
release_month = st.selectbox("Release Month", list(range(1, 13)))
is_sequel = st.radio("Is Sequel?", ["No", "Yes"])
is_sequel = 1 if is_sequel == "Yes" else 0

poster = st.file_uploader("Upload Poster", type=['png','jpg','jpeg'])

# PREVIEW DATA
df_input = pd.DataFrame([{
    "genre": genre,
    "budget_million": budget,
    "director_score": director_score,
    "cast_popularity": cast_popularity,
    "runtime_minutes": runtime,
    "release_month": release_month,
    "marketing_spend": marketing_spend,
    "is_sequel": is_sequel
}])

st.subheader("📊 Preview Input")
st.table(df_input)

if poster:
    st.image(poster, width=250)

# ---------- PREDICT ----------
if st.button("Predict"):
    if not model.is_model_available():
        st.warning("Model not found. Train or Load first.")
        st.stop()

    result = model.predict_single(df_input)
    pred = int(result['prediction'][0])
    score = float(result['score'][0]) * 100

    if pred == 1:
        st.success(f"🎉 HIT — Confidence: {score:.2f}%")
    else:
        st.error(f"❌ FLOP — Confidence: {score:.2f}%")
