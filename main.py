# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
from io import BytesIO

# ---------------- Encabezados (como lo quieres) ----------------
st.set_page_config(layout="wide", page_title="K-Means con PCA")
st.title("Aprendizaje no supervizado: k-means")
st.subheader("By Carlos Alberto Castro Luna 744849")
st.subheader("cargar datos")

# ---------------- Carga de archivo ----------------
uploaded_file = st.file_uploader("Sube un archivo CSV con tus datos", type=["csv"])

# ---------------- Controles (como en tus imágenes) ----------------
n_init_val = st.number_input("ingresa el valor de la varibale n_init:", min_value=1, max_value=1000, value=1, step=1)
max_iter_val = st.number_input("ingresa el valor de maximas iteraciones:", min_value=1, max_value=5000, value=300, step=1)
random_state_val = st.number_input("ingresa el valor de random state:", min_value=0, max_value=100000, value=0, step=1)

choose_init = st.toggle("Elegir init", value=True)
if choose_init:
    init_value = st.radio(" ", options=["k-means++", "random"], index=1, horizontal=True, label_visibility="collapsed")
else:
    init_value = "k-means++"
st.write(f"init = {init_value}")

# ======================= Lógica principal =======================
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Archivo cargado correctamente.")
    st.write("### Vista previa de los datos:")
    st.dataframe(data.head())

    # Tomar TODAS las columnas numéricas (como en tus capturas)
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("⚠️ El archivo debe contener al menos dos columnas numéricas.")
        st.stop()

    X = data[numeric_cols].copy()

    # ----- KMeans (k fijo = 3) con los parámetros elegidos -----
    kmeans = KMeans(
        n_clusters=3,
        init=init_value,
        n_init=int(n_init_val),
        max_iter=int(max_iter_val),
        random_state=int(random_state_val)
    )
    kmeans.fit(X)
    data["cluster"] = kmeans.labels_  # minúscula como se ve en tu preview

    # ----- PCA 2D para visualización -----
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    pca_df = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
    pca_df["Cluster"] = data["cluster"].astype(str)

    # ---------------- Antes de K-Means ----------------
    st.subheader("distribucion original (antes de K-Means)")
    fig_before = px.scatter(
        pca_df, x="PCA1", y="PCA2",
        title="Datos originales proyectados con PCA (sin agrupar)",
        color_discrete_sequence=["gray"]
    )
    st.plotly_chart(fig_before, use_container_width=True)

    # ---------------- Después de K-Means ----------------
    st.subheader("Datos agrupados con K-Means (k = 3)")
    fig_after = px.scatter(
        pca_df, x="PCA1", y="PCA2",
        color="Cluster",
        title="Clusters visualizados en 2D con PCA",
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    st.plotly_chart(fig_after, use_container_width=True)

    # ---------------- Centroides en espacio PCA ----------------
    st.subheader("Centroides de los clusters (en espacio PCA)")
    centroides_pca = pd.DataFrame(
        pca.transform(kmeans.cluster_centers_),
        columns=["PCA1", "PCA2"]
    )
    st.dataframe(centroides_pca)

    # ---------------- Método del Codo ----------------
    st.subheader("metodo del Codo (Elbow Method)")
    if st.button("Calcular numero optimo de clusters"):
        inertias = []
        K = list(range(1, 11))
        for kk in K:
            km = KMeans(
                n_clusters=kk,
                init=init_value,
                n_init=int(n_init_val),
                max_iter=int(max_iter_val),
                random_state=int(random_state_val)
            )
            km.fit(X)
            inertias.append(km.inertia_)
        fig_elbow = px.line(x=K, y=inertias, markers=True,
                            labels={"x": "Número de clusters (k)", "y": "Inercia (SSE)"},
                            title="Método del Codo")
        st.plotly_chart(fig_elbow, use_container_width=True)

    # ---------------- Descarga ----------------
    st.subheader("Descargar datos con clusters asignados")
    buf = BytesIO()
    data.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button(
        "Descargar CSV con Clusters",
        data=buf,
        file_name="datos_clusterizados.csv",
        mime="text/csv"
    )

else:
    st.info("Por favor, sube un archivo CSV para comenzar el análisis")
