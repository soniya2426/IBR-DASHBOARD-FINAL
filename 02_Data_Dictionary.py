
import streamlit as st
import pandas as pd

st.title("Data Dictionary")

rows = [
    ("personalisation", "1–5", "Perceived tailoring/personalised experience"),
    ("brand_authenticity", "1–5", "Perceived brand authenticity"),
    ("halal_certified", "1–5", "Trust in halal certification"),
    ("pricing", "1–5", "Perception of pricing/value"),
    ("gen_z_perception", "1–5", "Overall Gen Z perception of the brand"),
    ("brand_loyalty", "1–5", "Self-reported loyalty to the brand"),
    ("willingness_to_buy", "1–5", "Purchase intention / willingness to buy"),
    ("transparency", "1–5", "Perceived transparency"),
    ("consistency", "1–5", "Perceived consistency"),
    ("sustainable", "1–5", "Perceived sustainability"),
    ("would_recommend", "1–5", "Likelihood to recommend (used as NPS proxy)"),
]
df = pd.DataFrame(rows, columns=["Variable", "Scale", "Description"])
st.dataframe(df, use_container_width=True)
