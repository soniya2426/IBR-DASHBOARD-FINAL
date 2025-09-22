import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import statsmodels.api as sm
import pingouin as pg

from utils import harmonize_columns, add_brand, numeric_subset, nps_from_5pt, zscore, safe_merge

st.set_page_config(page_title="Gen Z Luxury Skincare – Two-Brand Analytics", layout="wide")

st.title("Gen Z Luxury Skincare – Two-Brand Analytics Dashboard")

st.markdown(
"""
**Brands:** Estée Lauder vs Shiffa  
This dashboard compares datasets, runs correlations and regressions, **tests your hypotheses (H1–H6)**, builds **STP segmentation**, and produces a **perceptual map**.
"""
)

# --- DATA LOADING ---
st.sidebar.header("1) Upload / Use Sample Data")
up1 = st.sidebar.file_uploader("Upload Estée Lauder file (.xlsx)", type=["xlsx"], key="el")
up2 = st.sidebar.file_uploader("Upload Shiffa file (.xlsx)", type=["xlsx"], key="sh")

if up1 is not None and up2 is not None:
    estee = pd.read_excel(up1)
    shiffa = pd.read_excel(up2)
else:
    estee = pd.read_excel("data/estee_lauder_dataset_50.xlsx")
    shiffa = pd.read_excel("data/shiffa_synthetic_dataset_50.xlsx")

estee, shiffa = safe_merge(estee, shiffa)

estee["brand"] = "Estee Lauder"
shiffa["brand"] = "Shiffa"

df = pd.concat([estee, shiffa], ignore_index=True)

st.subheader("Dataset Snapshot")
st.dataframe(df.head(10), use_container_width=True)

# --- METRICS & BASIC COMPARISON ---
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

key_vars = ["personalisation","brand_authenticity","halal_certified","pricing",
            "gen_z_perception","brand_loyalty","willingness_to_buy",
            "transparency","consistency","sustainable","would_recommend"]
existing = [c for c in key_vars if c in df.columns]

st.write("**Key Variables Found:**", ", ".join(existing))

mean_table = df.groupby("brand")[existing].mean().T
st.write("### Brand Means (selected variables)")
st.dataframe(mean_table.style.format("{:.2f}"), use_container_width=True)

# --- ANALYTICS SECTION ---
st.header("Analytics")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Distributions & KPIs", "Correlation Heatmap", "Regressions", "Hypotheses Tests", "Perceptual Map & STP"
])

with tab1:
    st.subheader("1) KPI Cards")
    c1, c2, c3, c4 = st.columns(4)
    for brand, col in zip([estee, shiffa], [c1, c2]):
        with col:
            nps, prom, detr = nps_from_5pt(brand["would_recommend"]) if "would_recommend" in brand else (np.nan, np.nan, np.nan)
            st.metric(f"{brand['brand'].iloc[0]} – NPS (5-pt proxy)", f"{nps:.1f}")
    with c3:
        st.metric("Δ Purchase Intention (EL - Shiffa)",
                  f"{estee['willingness_to_buy'].mean() - shiffa['willingness_to_buy'].mean():.2f}")
    with c4:
        st.metric("Δ Brand Loyalty (EL - Shiffa)",
                  f"{estee['brand_loyalty'].mean() - shiffa['brand_loyalty'].mean():.2f}")

    st.subheader("2) Distributions")
    pick = st.multiselect("Pick variables to view distributions", existing, default=["willingness_to_buy","brand_loyalty","pricing"])
    brand_pick = st.radio("Brand", ["Both","Estee Lauder","Shiffa"], horizontal=True)
    if brand_pick == "Estee Lauder":
        data = estee
    elif brand_pick == "Shiffa":
        data = shiffa
    else:
        data = df
    for v in pick:
        fig = px.histogram(data, x=v, color="brand" if brand_pick=="Both" else None, barmode="overlay", nbins=10, marginal="box")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("3) Boxplots (Compare Brands)")
    bpick = st.multiselect("Variables", existing, default=["pricing","willingness_to_buy","brand_loyalty"])
    for v in bpick:
        fig = px.box(df, x="brand", y=v, points="all")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("4) Correlation Heatmap (per brand & combined)")
    scope = st.radio("Scope", ["Combined","Estee Lauder","Shiffa"], horizontal=True)
    cdf = df if scope=="Combined" else (estee if scope=="Estee Lauder" else shiffa)
    num = numeric_subset(cdf)
    corr = num.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title=f"Correlation ({scope})", color_continuous_scale="RdBu", zmin=-1, zmax=1)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("5) Linear Regression")
    st.markdown("**Model A (H1–H4):** gen_z_perception ~ personalisation + brand_authenticity + halal_certified + pricing")
    scope_r = st.radio("Run model for", ["Combined","Estee Lauder","Shiffa"], key="regscope", horizontal=True)
    rdf = df if scope_r=="Combined" else (estee if scope_r=="Estee Lauder" else shiffa)

    # Model A
    Xa_cols = [c for c in ["personalisation","brand_authenticity","halal_certified","pricing"] if c in rdf]
    if len(Xa_cols)>=2 and "gen_z_perception" in rdf:
        X = sm.add_constant(rdf[Xa_cols])
        y = rdf["gen_z_perception"]
        modA = sm.OLS(y, X, missing="drop").fit()
        st.write(modA.summary().as_html(), unsafe_allow_html=True)
    else:
        st.info("Required columns missing for Model A.")

    st.markdown("**Model B (H6):** willingness_to_buy ~ gen_z_perception")
    if "gen_z_perception" in rdf and "willingness_to_buy" in rdf:
        X = sm.add_constant(rdf[["gen_z_perception"]])
        y = rdf["willingness_to_buy"]
        modB = sm.OLS(y, X, missing="drop").fit()
        st.write(modB.summary().as_html(), unsafe_allow_html=True)

    st.markdown("**Model C (H5 – Moderation):** willingness_to_buy ~ gen_z_perception + brand_loyalty + gen_z_perception:brand_loyalty")
    if "gen_z_perception" in rdf and "brand_loyalty" in rdf and "willingness_to_buy" in rdf:
        zdf = rdf[["gen_z_perception","brand_loyalty","willingness_to_buy"]].dropna().copy()
        # standardize
        zdf["g"] = (zdf["gen_z_perception"] - zdf["gen_z_perception"].mean())/zdf["gen_z_perception"].std(ddof=0)
        zdf["b"] = (zdf["brand_loyalty"] - zdf["brand_loyalty"].mean())/zdf["brand_loyalty"].std(ddof=0)
        zdf["gx b"] = zdf["g"]*zdf["b"]
        X = sm.add_constant(zdf[["g","b","gx b"]])
        y = zdf["willingness_to_buy"]
        modC = sm.OLS(y, X).fit()
        st.write(modC.summary().as_html(), unsafe_allow_html=True)

        # Simple slopes plot
        st.write("Simple Slopes Visual")
        for lvl, name in [(-1, "Low loyalty (-1σ)"), (0, "Avg loyalty"), (1, "High loyalty (+1σ)")]: 
            pred = zdf.copy()
            pred["b"] = lvl
            pred["gx b"] = pred["g"]*pred["b"]
            yhat = sm.add_constant(pred[["g","b","gx b"]]).dot(modC.params)
            pred[name] = yhat
        fig = px.line(pred.sort_values("g"), x="g", y=["Low loyalty (-1σ)","Avg loyalty","High loyalty (+1σ)"], labels={"g":"Gen Z perception (z)","value":"Predicted willingness"})
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("6) Mediation (H5a–H5d) via Gen Z Perception")
    st.markdown("We test whether **gen_z_perception** mediates the effect of each driver on **willingness_to_buy**. Bootstrapped (5,000) indirect effects via `pingouin.mediation_analysis`.")
    drivers = [c for c in ["personalisation","brand_authenticity","halal_certified","pricing"] if c in df.columns]
    scope_m = st.radio("Scope", ["Combined","Estee Lauder","Shiffa"], key="medscope", horizontal=True)
    mdf = df if scope_m=="Combined" else (estee if scope_m=="Estee Lauder" else shiffa)

    if "gen_z_perception" in mdf and "willingness_to_buy" in mdf:
        for x in drivers:
            st.write(f"**X = {x} → M = gen_z_perception → Y = willingness_to_buy**")
            try:
                out = pg.mediation_analysis(data=mdf[[x,"gen_z_perception","willingness_to_buy"]].dropna(),
                                            x=x, m="gen_z_perception", y="willingness_to_buy",
                                            covar=None, seed=42, n_boot=5000)
                st.dataframe(out)
            except Exception as e:
                st.warning(f"Mediation failed for {x}: {e}")
    else:
        st.info("Required columns missing for mediation.")

with tab5:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("7) Perceptual Map (PCA on attributes)")
        attrs = st.multiselect("Attributes to position on", 
                               [v for v in ["personalisation","brand_authenticity","halal_certified","pricing","transparency","consistency","sustainable"] if v in df.columns],
                               default=[v for v in ["personalisation","brand_authenticity","halal_certified","pricing"] if v in df.columns])
        if len(attrs)>=2:
            X = df[attrs].dropna()
            labels = df.loc[X.index, "brand"]
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(Xs)
            pcadf = pd.DataFrame(coords, columns=["PC1","PC2"])
            pcadf["brand"] = labels.values
            centroids = pcadf.groupby("brand")[["PC1","PC2"]].mean().reset_index()
            fig = px.scatter(pcadf, x="PC1", y="PC2", color="brand", opacity=0.3, title="Respondent-level PCA (2D)")
            fig.add_scatter(x=centroids["PC1"], y=centroids["PC2"], mode="markers+text", text=centroids["brand"],
                            marker=dict(size=18, line=dict(width=2, color="black")), name="Brand centroids")
            st.plotly_chart(fig, use_container_width=True)
            # Attribute loadings (biplot-ish)
            loadings = pd.DataFrame(pca.components_.T, index=attrs, columns=["PC1","PC2"]).reset_index().rename(columns={"index":"attribute"})
            fig2 = px.scatter(loadings, x="PC1", y="PC2", text="attribute", title="Attribute Loadings")
            st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.subheader("8) STP – Segmentation")
        seg_vars = st.multiselect("Variables for clustering",
                                  [v for v in ["personalisation","brand_authenticity","halal_certified","pricing","brand_loyalty","willingness_to_buy","transparency","consistency","sustainable"] if v in df.columns],
                                  default=[v for v in ["personalisation","brand_authenticity","halal_certified","pricing","brand_loyalty","willingness_to_buy"] if v in df.columns])
        k = st.slider("Number of clusters (segments)", 2, 6, 3)
        if len(seg_vars)>=2:
            X = df[seg_vars].dropna()
            idx = X.index
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            km = KMeans(n_clusters=k, n_init=20, random_state=42)
            seg = km.fit_predict(Xs)
            segdf = df.loc[idx, ["brand"]].copy()
            segdf["segment"] = seg
            prof = pd.concat([segdf, X.reset_index(drop=True)], axis=1).groupby("segment").mean().round(2)
            st.write("**Segment sizes (overall)**")
            st.dataframe(segdf["segment"].value_counts().rename("count"))
            st.write("**Segment profiles (mean values)**")
            st.dataframe(prof)
            fig = px.scatter_matrix(pd.DataFrame(Xs, columns=seg_vars), color=seg, title="Clusters in feature space")
            st.plotly_chart(fig, use_container_width=True)
            # Basic naming suggestions
            st.markdown("**Auto labels (suggested):**")
            labels = {0:"Luxury Loyalists", 1:"Value Seekers", 2:"Ethical Purists", 3:"Personalisation Fans", 4:"Skeptical Experimenters", 5:"Trend Seekers"}
            st.write({i: labels.get(i, f"Segment {i}") for i in range(k)})

st.caption("Built with ❤️ using Streamlit, Plotly, scikit-learn, statsmodels & pingouin")
