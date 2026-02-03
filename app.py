import streamlit as st
import pandas as pd
import plotly.express as px

from utils.interpreter import interpret_3d
from utils.comparator import compare_views
from utils.language import enrich_language
from utils.confidence import confidence_score

st.set_page_config(layout="wide")
st.title("3D Visualization Intelligence Engine")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 3:
        st.error("Need at least 3 numeric columns.")
        st.stop()

    st.sidebar.header("3D View Controls")
    x = st.sidebar.selectbox("X-axis", numeric_cols)
    y = st.sidebar.selectbox("Y-axis", numeric_cols, index=1)
    z = st.sidebar.selectbox("Z-axis", numeric_cols, index=2)

    domain = st.sidebar.selectbox(
        "Domain Language",
        ["General", "Finance", "Supply Chain", "ML"]
    )

    fig = px.scatter_3d(df, x=x, y=y, z=z, opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧠 Interpretation")
    raw_insights = interpret_3d(df, x, y, z)
    enriched = enrich_language(raw_insights, domain)

    for i in enriched:
        st.markdown(f"- {i}")

    st.subheader("📊 Confidence Scores")
    conf = confidence_score(df, x, y, z)
    st.progress(conf / 100)
    st.caption(f"Interpretation confidence: {conf}%")

    st.divider()
    st.subheader("🔄 Compare With Another View")

    x2 = st.selectbox("X-axis (comparison)", numeric_cols, key="x2")
    y2 = st.selectbox("Y-axis (comparison)", numeric_cols, index=1, key="y2")
    z2 = st.selectbox("Z-axis (comparison)", numeric_cols, index=2, key="z2")

    if st.button("Compare Views"):
        diffs = compare_views(df, (x, y, z), (x2, y2, z2))
        for d in diffs:
            st.markdown(f"- {d}")
