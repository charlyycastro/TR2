import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from io import BytesIO
import plotly.express as px
import matplotlib.pyplot as plt

# ConfiguraciÃ³n de la app
st.set_page_config(page_title="K-Means con PCA y Comparativa", layout="wide")
st.title("lustering Interactivo con K-Means y PCA (comparacion Antes/despues)")
st.title("Carlos Alberto Castro Luna 744849")
st.write("""
Sube tus datos, aplica **K-Means**, y observa como el algoritmo agrupa los puntos en un espacio reducido con **PCA (2D o 3D)**.  
tambien puedes comparar la distribuciones **antes y despues del clustering.
""")

# --- Subir archivo ---
uploaded_file = st.file_uploader("Sube un archivo CSV con tus datos", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("… Archivo cargado correctamente.")
    st.write("### Vista previa de los datos:")
    st.dataframe(data.head())

    # Filtrar columnas numÃ©ricas
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("El archivo debe contener al menos dos columnas numericas.")
    else:
        st.sidebar.header("confifuracion del modelo")

        # Seleccionar columnas a usar
        selected_cols = st.sidebar.multiselect(
            "Selecciona las columnas numericas para el clustering:",
            numeric_cols,
            default=numeric_cols
        )


        k = st.sidebar.slider("numero de clusters (k):", 1, 10, 3)
        n_components = st.sidebar.radio("visualizacion de PCA:", [2, 3], index=0)

        # --- Datos y modelo ---
        X = data[selected_cols]

        # aqui es en dond se va cambiar el codigo para meter nuevos parametros
        n = st.number_input(f'ingresa el valor de la varibale n_init: ', value=1, min_value=1)
        m = st.number_input(f'ingresa el valor de maximas iteraciones: ', value=300,min_value=1)
        r = st.number_input(f'ingresa el valor de random state: ', value=0,min_value=0)

        #metodo para cambiar el init
        on = st.toggle("Elegir init")
        if on:
            st.write("init = k-means++")
            inn = 'k-means++'
        else:
            st.write("init = random")
            inn = 'random'

        kmeans = KMeans(n_clusters=k,init=inn, max_iter=m, n_init=n, random_state=r)
        #kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        data['Cluster'] = kmeans.labels_

        # --- PCA ---
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        pca_cols = [f'PCA{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(X_pca, columns=pca_cols)
        pca_df['Cluster'] = data['Cluster']

        # --- VisualizaciÃ³n antes del clustering ---
        st.subheader("distribucion original (antes de K-Means)")
        if n_components == 2:
            fig_before = px.scatter(
                pca_df,
                x='PCA1',
                y='PCA2',
                title="Datos originales proyectados con PCA (sin agrupar)",
                color_discrete_sequence=["gray"]
            )
        else:
            fig_before = px.scatter_3d(
                pca_df,
                x='PCA1',
                y='PCA2',
                z='PCA3',
                title="Datos originales proyectados con PCA (sin agrupar)",
                color_discrete_sequence=["gray"]
            )
        st.plotly_chart(fig_before, use_container_width=True)

        # --- VisualizaciÃ³n despuÃ©s del clustering ---
        st.subheader(f"Datos agrupados con K-Means (k = {k})")
        if n_components == 2:
            fig_after = px.scatter(
                pca_df,
                x='PCA1',
                y='PCA2',
                color=pca_df['Cluster'].astype(str),
                title="Clusters visualizados en 2D con PCA",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
        else:
            fig_after = px.scatter_3d(
                pca_df,
                x='PCA1',
                y='PCA2',
                z='PCA3',
                color=pca_df['Cluster'].astype(str),
                title="Clusters visualizados en 3D con PCA",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
        st.plotly_chart(fig_after, use_container_width=True)

        # --- Centroides ---
        st.subheader("Centroides de los clusters (en espacio PCA)")
        centroides_pca = pd.DataFrame(pca.transform(kmeans.cluster_centers_), columns=pca_cols)
        st.dataframe(centroides_pca)

        # --- MÃ©todo del Codo ---
        st.subheader("metodo del Codo (Elbow Method)")
        if st.button("Calcular numero optimo de clusters"):
            inertias = []
            K = range(1, 11)
            for i in K:
                km = KMeans(n_clusters=i, random_state=42)
                km.fit(X)
                inertias.append(km.inertia_)

            fig2, ax2 = plt.subplots(figsize=(8, 6))
            plt.plot(K, inertias, 'bo-')
            plt.title('metodo del Codo')
            plt.xlabel('numero de Clusters (k)')
            plt.ylabel('Inercia (SSE)')
            plt.grid(True)
            st.pyplot(fig2)

        # --- Descarga de resultados ---
        st.subheader("Descargar datos con clusters asignados")
        buffer = BytesIO()
        data.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button(
            label="Descargar CSV con Clusters",
            data=buffer,
            file_name="datos_clusterizados.csv",
            mime="text/csv"
        )

else:
    st.info("Carga un archivo CSV en la barra lateral para comenzar.")
    st.write("""
    **Ejemplo de formato:**
    | Ingreso_Anual | Gasto_Tienda | Edad |
    |----------------|--------------|------|
    | 45000 | 350 | 28 |
    | 72000 | 680 | 35 |
    | 28000 | 210 | 22 |
    """)

