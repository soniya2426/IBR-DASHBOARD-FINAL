# Gen Z Luxury Skincare – Two-Brand Analytics Dashboard (Streamlit)

This Streamlit app compares two brand datasets (Estée Lauder & Shiffa), tests your hypotheses, runs correlation and linear regressions, builds STP segmentation, and draws a perceptual map.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Datasets are in `data/`. You can also upload new files from the UI.

## Deploy on Streamlit Community Cloud
1. Push this folder to a new GitHub repo.
2. In Streamlit Community Cloud, set the repo and `app.py` as the entrypoint.
3. Add the Python version/runtime as needed (3.10+), and it will auto-install `requirements.txt`.
