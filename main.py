# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from io import BytesIO
import plotly.express as px
import matplotlib.pyplot as plt

# =================== Configuración ===================
st.set_page_config(page_title="Panel K-Means + PCA", layout="wide")

st.title("Panel K-Means + PCA — Comparador")
st.caption("Sube tus datos, ajusta parámetros y visualiza el efecto del clustering en PCA (2D o 3D).")

# ---------- Carga de datos ----------
st.sidebar.header("📂 Datos")
uploaded = st.sidebar.file_uploader("CSV", type=["csv"])

# ---------- Parámetros (sidebar) ----------
st.sidebar.header("⚙️ Parámetros del modelo")

k = st.sidebar.slider("Número de clusters (k)", 1, 10, 3)
pca_dims = st.sidebar.radio("Dimensiones de PCA", [2, 3], horizontal=True, index=0)

# init con toggle para que puedas alternar exactamente como pedías
toggle_init = st.sidebar.toggle("Alternar init (k-means++ / random)", value=True)
init_val = "k-means++" if toggle_init else "random"
st.sidebar.caption(f"init = **{init_val}**")

n_init_val = st.sidebar.number_input("n_init (reinicios)", min_value=1, max_value=1000, value=10, step=1)
max_iter_val = st.sidebar.number_input("max_iter (iteraciones máx.)", min_value=1, max_value=5000, value=300, step=10)
random_state_val = st.sidebar.number_input("random_state", min_value=0, max_value=100000, value=0, step=1)

# =================== Lógica ===================
if uploaded is None:
    st.info("Carga un archivo CSV para comenzar.")
else:
    df = pd.read_csv(uploaded)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) < 2:
        st.error("El archivo debe contener al menos dos columnas numéricas.")
        st.stop()

    # Selección de columnas (sidebar)
    with st.sidebar.expander("🎛️ Columnas para el clustering", expanded=True):
        cols = st.multiselect(
            "Selecciona columnas numéricas",
            options=num_cols,
            default=num_cols
        )
    if len(cols) < 2:
        st.warning("Selecciona al menos dos columnas.")
        st.stop()

    X = df[cols].copy().dropna()

    # Ajuste del modelo
    km = KMeans(
        n_clusters=k,
        init=init_val,
        n_init=int(n_init_val),
        max_iter=int(max_iter_val),
        random_state=int(random_state_val)
    )
    km.fit(X)
    df["Cluster"] = km.labels_

    # PCA
    pca = PCA(n_components=pca_dims)
    X_pca = pca.fit_transform(X)
    pca_names = [f"PCA{i+1}" for i in range(pca_dims)]
    pca_df = pd.DataFrame(X_pca, columns=pca_names)
    pca_df["Cluster"] = df.loc[X.index, "Cluster"].astype(str)  # asegurar alineación

    # =================== UI por pestañas ===================
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Exploración", "🟣 Antes / Después", "📍 Centroides", "📈 Elbow", "💾 Descarga"]
    )

    # -------- Exploración --------
    with tab1:
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Vista previa")
            st.dataframe(df.head())
        with right:
            st.subheader("Resumen")
            st.metric("Filas", len(df))
            st.metric("Columnas numéricas", len(num_cols))
            st.metric("Inercia (SSE)", f"{km.inertia_:.4f}")
            st.write("**Parámetros**")
            st.json({
                "k": k,
                "init": init_val,
                "n_init": int(n_init_val),
                "max_iter": int(max_iter_val),
                "random_state": int(random_state_val),
                "columnas": cols
            })

    # -------- Antes / Después --------
    with tab2:
        st.subheader("Distribución original (antes de K-Means)")
        if pca_dims == 2:
            fig_before = px.scatter(
                pca_df, x="PCA1", y="PCA2",
                title="Datos originales proyectados con PCA (sin agrupar)",
                color_discrete_sequence=["gray"]
            )
        else:
            fig_before = px.scatter_3d(
                pca_df, x="PCA1", y="PCA2", z="PCA3",
                title="Datos originales proyectados con PCA (sin agrupar)",
                color_discrete_sequence=["gray"]
            )
        st.plotly_chart(fig_before, use_container_width=True)

        st.subheader(f"Datos agrupados con K-Means (k = {k})")
        if pca_dims == 2:
            fig_after = px.scatter(
                pca_df, x="PCA1", y="PCA2",
                color="Cluster",
                title="Clusters visualizados en 2D con PCA",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
        else:
            fig_after = px.scatter_3d(
                pca_df, x="PCA1", y="PCA2", z="PCA3",
                color="Cluster",
                title="Clusters visualizados en 3D con PCA",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
        st.plotly_chart(fig_after, use_container_width=True)

    # -------- Centroides (en PCA) --------
    with tab3:
        st.subheader("Centroides de los clusters (proyectados en PCA)")
        cent_pca = pd.DataFrame(pca.transform(km.cluster_centers_), columns=pca_names)
        st.dataframe(cent_pca, use_container_width=True)

    # -------- Método del Codo --------
    with tab4:
        st.subheader("Método del Codo (Elbow)")
        if st.button("Calcular número óptimo de clusters", key="elbow"):
            inertias = []
            Ks = range(1, 11)
            for kk in Ks:
                km_tmp = KMeans(
                    n_clusters=kk,
                    init=init_val,
                    n_init=int(n_init_val),
                    max_iter=int(max_iter_val),
                    random_state=int(random_state_val)
                )
                km_tmp.fit(X)
                inertias.append(km_tmp.inertia_)

            fig_elbow, ax = plt.subplots(figsize=(8, 5), dpi=110)
            ax.plot(list(Ks), inertias, "o-")
            ax.set_xlabel("Número de Clusters (k)")
            ax.set_ylabel("Inercia (SSE)")
            ax.set_title("Elbow Method")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig_elbow)

    # -------- Descarga --------
    with tab5:
        st.subheader("Descargar datos con clusters asignados")
        buf = BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        st.download_button(
            label="⬇️ Descargar CSV con Clusters",
            data=buf,
            file_name="datos_clusterizados.csv",
            mime="text/csv"
        )
