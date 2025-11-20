import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.title("📊 Analytics Dashboard — Moviegrad")

DATA_PATH = "data/movies_100k.csv"

if not os.path.exists(DATA_PATH):
    st.error("Dataset not found. Generate it from main page.")
    st.stop()

df = pd.read_csv(DATA_PATH)

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("🎭 Hit Rate by Genre")
genre_hit = df.groupby("genre")["box_office_hit"].mean().reset_index()
st.plotly_chart(px.bar(genre_hit, x="genre", y="box_office_hit", color="genre"))

st.subheader("💰 Budget vs Hit Probability")
st.plotly_chart(px.scatter(df.sample(3000), x="budget_million",
                        y="box_office_hit", color="genre", trendline="lowess"))

st.subheader("📆 Release Month Success Rate")
month_hit = df.groupby("release_month")["box_office_hit"].mean().reset_index()
st.plotly_chart(px.line(month_hit, x="release_month", y="box_office_hit", markers=True))
