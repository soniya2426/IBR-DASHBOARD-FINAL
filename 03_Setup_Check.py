
import streamlit as st
import os

st.title("Setup Check")

st.write("**Working directory**:", os.getcwd())
st.write("**Files present:**")
for root, dirs, files in os.walk("."):
    level = root.replace(".", "").count(os.sep)
    indent = " " * 2 * level
    st.write(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 2 * (level + 1)
    for f in files:
        st.write(f"{subindent}{f}")
