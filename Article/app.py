# app.py

import streamlit as st
from recommender import recommend_articles

st.set_page_config(page_title="Smart Article Recommender")

st.title("📚 Smart Article Recommender")
st.write("Paste an article title to get recommendations:")

article_title = st.text_input("Enter article title:")

if st.button("Recommend"):
    if article_title:
        with st.spinner("Finding great reads..."):
            recs = recommend_articles(article_title)
        st.success("Here are some recommendations:")
        for i, rec in enumerate(recs):
            st.markdown(f"**{i+1}.** {rec}")
    else:
        st.warning("Please enter an article title.")
