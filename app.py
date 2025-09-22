import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import statsmodels.api as sm
import pingouin as pg

# utils.py should be in the repo root (as created earlier)
from utils import (
    harmonize_columns,
    add_brand,
    numeric_subset,
    nps_from_5pt,
    zscore,
    safe_merge,
)

st.set_page_config(page_title="Gen Z Luxury Skincare – Two-Brand Analytics", layout="wide")

st.title("Gen Z Luxury Skincare – Two-Brand Analytics Dashboard")
st.markdown(
    """
**Brands:** Estée Lauder vs Shiffa  
This dashboard compares datasets, runs correlations and regressions, **tests your hypotheses (H1–H6)**, builds **STP segmentation**, and produces a **perceptual map**.
"""
)

# =============================================================================
# Helpers
# =============================================================================
def read_table(uploaded):
    """Read uploaded file (xlsx or csv) robustly."""
    if uploaded is None:
        return None
    name = getattr(uploaded, "name", "").lower()
    try:
        if name.endswith((".xlsx", ".xlsm", ".xltx", ".xltm")):
            return pd.read_excel(uploaded, engine="openpyxl")
        elif name.endswith(".csv"):
            return pd.read_csv(uploaded)
        else:
            # try excel as fallback
            return pd.read_excel(uploaded, engine="openpyxl")
    except Exception as e:
        st.error(f"Error reading {name or 'uploaded file'}: {e}")
        return None

def load_bundled_or_ask():
    """
    Try to load bundled files from ./data; if not present,
    ask the user to upload from the sidebar.
    """
    default_el = "data/estee_lauder-dataset_50.xlsx"
    default_sh = "data/shiffa_synthetic_dataset 50.xlsx"  # note the space
    have_defaults = os.path.exists(default_el) and os.path.exists(default_sh)
    if have_defaults:
        el = pd.read_excel(default_el, engine="openpyxl")
        sh = pd.read_excel(default_sh, engine="openpyxl")
        return el, sh, True
    else:
        st.warning(
            "Bundled data files were not found at:\n\n"
            f"- `{default_el}`\n"
            f"- `{default_sh}`\n\n"
            "Please upload the two datasets using the sidebar (xlsx or csv)."
        )
        return None, None, False

# =============================================================================
# Data loading
# =============================================================================
st.sidebar.header("Upload / Use Data")
up1 = st.sidebar.file_uploader("Upload Estée Lauder file (.xlsx or .csv)", type=["xlsx", "csv"], key="el")
up2 = st.sidebar.file_uploader("Upload Shiffa file (.xlsx or .csv)", type=["xlsx", "csv"], key="sh")

if up1 is not None and up2 is not None:
    estee = read_table(up1)
    shiffa = read_table(up2)
    if estee is None or shiffa is None:
        st.stop()
else:
    estee, shiffa, ok = load_bundled_or_ask()
    if not ok:
        st.info("Waiting for uploads…")
        st.stop()

# Harmonize & combine
estee, shiffa = safe_merge(estee, shiffa)
estee["brand"] = "Estee Lauder"
shiffa["brand"] = "Shiffa"
df = pd.concat([estee, shiffa], ignore_index=True)

st.subheader("Dataset Snapshot")
st.dataframe(df.head(10), use_container_width=True)

# =============================================================================
# Metrics & basic comparison
# =============================================================================
st.header("Topline Comparison")
cols = st.columns(4)
with cols[0]:
    st.metric("Estée rows", len(estee))
with cols[1]:
    st.metric("Shiffa rows", len(shiffa))
with cols[2]:
    st.metric("Shared numeric cols", len(numeric_subset(df).columns))
with cols[3]:
    st.metric("Total rows", len(df))

key_vars = [
    "personalisation","brand_authenticity","halal_certified","pricing",
    "gen_z_perception","brand_loyalty","willingness_to_buy",
    "transparency","consistency","sustainable","would_recommend"
]
existing = [c for c in key_vars if c in df.columns]

st.write("**Key Variables Found:**", ", ".join(existing))

mean_table = df.groupby("brand")[existing].mean(numeric_only=True).T
st.write("### Brand Means (selected variables)")
st.dataframe(mean_table.style.format("{:.2f}"), use_container_width=True)

# =============================================================================
# Analytics Tabs
# =============================================================================
st.header("Analytics")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Distributions & KPIs", "Correlation Heatmap", "Regressions", "Hypotheses Tests", "Perceptual Map & STP"
])

# -----------------------------------------------------------------------------
# Tab 1: KPIs & Distributions
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("1) KPI Cards")
    c1, c2, c3, c4 = st.columns(4)
    for brand, col in zip([estee, shiffa], [c1, c2]):
        with col:
            if "would_recommend" in brand:
                nps, prom, detr = nps_from_5pt(brand["would_recommend"])
                st.metric(f"{brand['brand'].iloc[0]} – NPS (5-pt proxy)", f"{nps:.1f}")
    with c3:
        if "willingness_to_buy" in estee and "willingness_to_buy" in shiffa:
            st.metric("Δ Purchase Intention (EL - Shiffa)",
                      f"{estee['willingness_to_buy'].mean() - shiffa['willingness_to_buy'].mean():.2f}")
    with c4:
        if "brand_loyalty" in estee and "brand_loyalty" in shiffa:
            st.metric("Δ Brand Loyalty (EL - Shiffa)",
                      f"{estee['brand_loyalty'].mean() - shiffa['brand_loyalty'].mean():.2f}")

    st.subheader("2) Distributions")
    pick = st.multiselect(
        "Pick variables to view distributions",
        existing,
        default=[v for v in ["willingness_to_buy","brand_loyalty","pricing"] if v in existing]
    )
    brand_pick = st.radio("Brand", ["Both","Estee Lauder","Shiffa"], horizontal=True)
    if brand_pick == "Estee Lauder":
        data = estee
    elif brand_pick == "Shiffa":
        data = shiffa
    else:
        data = df
    for v in pick:
        fig = px.histogram(
            data, x=v, color="brand" if brand_pick=="Both" else None,
            barmode="overlay", nbins=10, marginal="box"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("3) Boxplots (Compare Brands)")
    bpick = st.multiselect(
        "Variables",
        existing,
        default=[v for v in ["pricing","willingness_to_buy","brand_loyalty"] if v in existing]
    )
    for v in bpick:
        fig = px.box(df, x="brand", y=v, points="all")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 2: Correlation Heatmap
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("4) Correlation Heatmap (per brand & combined)")
    scope = st.radio("Scope", ["Combined","Estee Lauder","Shiffa"], horizontal=True)
    cdf = df if scope=="Combined" else (estee if scope=="Estee Lauder" else shiffa)
    num = numeric_subset(cdf)
    if not num.empty:
        corr = num.corr()
        fig = px.imshow(
            corr, text_auto=True, aspect="auto", title=f"Correlation ({scope})",
            color_continuous_scale="RdBu", zmin=-1, zmax=1
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric variables available for correlation.")

# -----------------------------------------------------------------------------
# Tab 3: Regressions (H1–H6)
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("5) Linear Regression")
    scope_r = st.radio("Run model for", ["Combined","Estee Lauder","Shiffa"], key="regscope", horizontal=True)
    rdf = df if scope_r=="Combined" else (estee if scope_r=="Estee Lauder" else shiffa)

    # Model A (H1–H4): gen_z_perception ~ personalisation + brand_authenticity + halal_certified + pricing
    Xa_cols = [c for c in ["personalisation","brand_authenticity","halal_certified","pricing"] if c in rdf]
    if len(Xa_cols) >= 2 and "gen_z_perception" in rdf:
        X = sm.add_constant(rdf[Xa_cols])
        y = rdf["gen_z_perception"]
        modA = sm.OLS(y, X, missing="drop").fit()
        st.write(modA.summary().as_html(), unsafe_allow_html=True)

    # Model B (H6): willingness_to_buy ~ gen_z_perception
    if "gen_z_perception" in rdf and "willingness_to_buy" in rdf:
        X = sm.add_constant(rdf[["gen_z_perception"]])
        y = rdf["willingness_to_buy"]
        modB = sm.OLS(y, X, missing="drop").fit()
        st.write(modB.summary().as_html(), unsafe_allow_html=True)

    # Model C (H5 moderation): willingness_to_buy ~ gen_z_perception (z) + brand_loyalty (z) + interaction
    if "gen_z_perception" in rdf and "brand_loyalty" in rdf and "willingness_to_buy" in rdf:
        zdf = rdf[["gen_z_perception","brand_loyalty","willingness_to_buy"]].dropna().copy()
        zdf["g"] = (zdf["gen_z_perception"] - zdf["gen_z_perception"].mean())/zdf["gen_z_perception"].std(ddof=0)
        zdf["b"] = (zdf["brand_loyalty"] - zdf["brand_loyalty"].mean())/zdf["brand_loyalty"].std(ddof=0)
        zdf["gx b"] = zdf["g"] * zdf["b"]
        X = sm.add_constant(zdf[["g","b","gx b"]])
        y = zdf["willingness_to_buy"]
        modC = sm.OLS(y, X).fit()
        st.write(modC.summary().as_html(), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Tab 4: Hypotheses Tests (Mediation)
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("6) Mediation (H5a–H5d) via Gen Z Perception")
    drivers = [c for c in ["personalisation","brand_authenticity","halal_certified","pricing"] if c in df.columns]
    scope_m = st.radio("Scope", ["Combined","Estee Lauder","Shiffa"], key="medscope", horizontal=True)
    mdf = df if scope_m=="Combined" else (estee if scope_m=="Estee Lauder" else shiffa)

    if "gen_z_perception" in mdf and "willingness_to_buy" in mdf:
        for x in drivers:
            st.write(f"**X = {x} → M = gen_z_perception → Y = willingness_to_buy**")
            try:
                out = pg.mediation_analysis(
                    data=mdf[[x,"gen_z_perception","willingness_to_buy"]].dropna(),
                    x=x, m="gen_z_perception", y="willingness_to_buy",
                    covar=None, seed=42, n_boot=5000
                )
                st.dataframe(out)
            except Exception as e:
                st.warning(f"Mediation failed for {x}: {e}")
    else:
        st.info("Required columns missing for mediation.")

# -----------------------------------------------------------------------------
# Tab 5: Perceptual Map & STP
# -----------------------------------------------------------------------------
with tab5:
    c1, c2 = st.columns(2)
    # Perceptual Map
    with c1:
        st.subheader("7) Perceptual Map (PCA on attributes)")
        attrs = st.multiselect(
            "Attributes to position on",
            [v for v in ["personalisation","brand_authenticity","halal_certified","pricing",
                         "transparency","consistency","sustainable"] if v in df.columns],
            default=[v for v in ["personalisation","brand_authenticity","halal_certified","pricing"] if v in df.columns]
        )
        if len(attrs) >= 2:
            X = df[attrs].dropna()
            labels = df.loc[X.index, "brand"]
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(Xs)
            pcadf = pd.DataFrame(coords, columns=["PC1","PC2"])
            pcadf["brand"] = labels.values
            centroids = pcadf.groupby("brand")[["PC1","PC2"]].mean(numeric_only=True).reset_index()
            fig = px.scatter(
                pcadf, x="PC1", y="PC2", color="brand", opacity=0.3, title="Respondent-level PCA (2D)"
            )
            fig.add_scatter(
                x=centroids["PC1"], y=centroids["PC2"], mode="markers+text", text=centroids["brand"],
                marker=dict(size=18, line=dict(width=2, color="black")), name="Brand centroids"
            )
            st.plotly_chart(fig, use_container_width=True)

    # STP Segmentation
    with c2:
        st.subheader("8) STP – Segmentation")
        seg_vars = st.multiselect(
            "Variables for clustering",
            [v for v in ["personalisation","brand_authenticity","halal_certified","pricing",
                         "brand_loyalty","willingness_to_buy","transparency","consistency","sustainable"]
             if v in df.columns],
            default=[v for v in ["personalisation","brand_authenticity","halal_certified",
                                 "pricing","brand_loyalty","willingness_to_buy"] if v in df.columns]
        )
        k = st.slider("Number of clusters (segments)", 2, 6, 3)
        if len(seg_vars) >= 2:
            X = df[seg_vars].dropna()
            idx = X.index
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            km = KMeans(n_clusters=k, n_init=20, random_state=42)
            seg = km.fit_predict(Xs)
            segdf = df.loc[idx, ["brand"]].copy()
            segdf["segment"] = seg
            prof = pd.concat([segdf, X.reset_index(drop=True)], axis=1) \
                     .groupby("segment").mean(numeric_only=True).round(2)
            st.write("**Segment sizes (overall)**")
            st.dataframe(segdf["segment"].value_counts().rename("count"))
            st.write("**Segment profiles (mean values)**")
            st.dataframe(prof)

st.caption("Built with ❤️ using Streamlit, Plotly, scikit-learn, statsmodels & pingouin")
