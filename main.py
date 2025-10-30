# -*- coding: utf-8 -*-
# refactorizado para que pueda funcionar con streamlit
# el archivo original esta en main

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import streamlit as st

# ---- Encabezados originales ----
st.title("Aprendizaje no supervizado: k-means")
st.subheader("By Carlos Alberto Castro Luna 744849")
st.subheader("cargar datos")

# ---- Carga de archivo ----
uploaded_file = st.file_uploader("Sube un archivo CSV con tus datos", type=["csv"])

# ---- Controles adicionales de KMeans ----
st.markdown("### Parámetros de KMeans")
col1, col2, col3, col4 = st.columns(4)

with col1:
    init_method = st.selectbox("init", options=["k-means++", "random"], index=0)

with col2:
    max_iter = st.number_input("max_iter", min_value=10, max_value=5000, value=300, step=10)

with col3:
    n_init_modo = st.selectbox("n_init", options=["auto", "entero"], index=0)
    if n_init_modo == "entero":
        n_init_val = st.number_input("n_init (entero)", min_value=1, max_value=1000, value=10, step=1)
        n_init_param = int(n_init_val)
    else:
        n_init_param = "auto"  # si tu scikit-learn no soporta 'auto', haremos fallback a 10

with col4:
    use_rs = st.checkbox("Usar random_state", value=True)
    if use_rs:
        random_state = st.number_input("random_state", min_value=0, max_value=100000, value=42, step=1)
        random_state = int(random_state)
    else:
        random_state = None

# Utilidad: crear KMeans con fallback si 'auto' no es soportado
def construir_kmeans(n_clusters, init, max_iter, n_init, random_state):
    try:
        return KMeans(
            n_clusters=n_clusters,
            init=init,
            max_iter=int(max_iter),
            n_init=n_init,
            random_state=random_state
        )
    except TypeError:
        # Fallback para scikit-learn < 1.4 cuando n_init="auto"
        if isinstance(n_init, str) and n_init.lower() == "auto":
            st.info("ℹ️ Tu versión de scikit-learn no soporta n_init='auto'. Se usará n_init=10.")
            return KMeans(
                n_clusters=n_clusters,
                init=init,
                max_iter=int(max_iter),
                n_init=10,
                random_state=random_state
            )
        raise

# ---- Lógica principal ----
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Datos")
    st.dataframe(df.head())

    # Validación simple: exactamente 2 columnas
    if df.shape[1] != 2:
        st.error("Este ejemplo espera **exactamente 2 columnas** numéricas en el CSV (p. ej. 'Saldo' y 'transacciones').")
        st.stop()

    # Normalización MinMax
    escalador = MinMaxScaler().fit(df.values)
    df_normalizado = pd.DataFrame(escalador.transform(df.values), columns=["Saldo", "transacciones"])
    st.dataframe(df_normalizado)

    # KMeans con k integrado (=3), usando los parámetros elegidos
    kmeans = construir_kmeans(
        n_clusters=3,  # ← k integrado como en tu ejemplo
        init=init_method,
        max_iter=max_iter,
        n_init=n_init_param,
        random_state=random_state
    ).fit(df_normalizado.values)

    df_normalizado["cluster"] = kmeans.labels_

    st.write(f"{kmeans.cluster_centers_}")
    st.write(f"{kmeans.inertia_}")

    # --- Gráfica de clusters ---
    fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=100)
    colores = ["red", "blue", "orange", "black", "purple", "pink", "brown"]

    for cluster in range(kmeans.n_clusters):
        ax1.scatter(
            df_normalizado[df_normalizado["cluster"] == cluster]["Saldo"],
            df_normalizado[df_normalizado["cluster"] == cluster]["transacciones"],
            marker="o", s=180, color=colores[cluster], alpha=0.5
        )
        ax1.scatter(
            kmeans.cluster_centers_[cluster][0],
            kmeans.cluster_centers_[cluster][1],
            marker="P", s=280, color=colores[cluster]
        )

    ax1.set_title("clientes", fontsize=20)
    ax1.set_xlabel("saldo en cuenta de ahorros", fontsize=15)
    ax1.set_ylabel("veces que uso tarjeta de credito", fontsize=15)
    ax1.text(1.15, 0.2, "k=%i" % kmeans.n_clusters, fontsize=15)
    ax1.text(1.15, 0, "Inercia = %0.2f" % kmeans.inertia_, fontsize=15)
    ax1.set_xlim(-0.1, 1.15)
    ax1.set_ylim(-0.1, 1.15)
    plt.tight_layout()
    st.pyplot(fig1)

    # --- Método del codo (respetando parámetros elegidos) ---
    inercias = []
    for k in range(2, 10):
        km_temp = construir_kmeans(
            n_clusters=k,
            init=init_method,
            max_iter=max_iter,
            n_init=n_init_param,
            random_state=random_state
        ).fit(df_normalizado[["Saldo", "transacciones"]].values)
        inercias.append(km_temp.inertia_)

    fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=100)
    ax2.scatter(range(2, 10), inercias, marker="o", s=180, color="purple")
    ax2.set_xlabel("numero de clusters", fontsize=25)
    ax2.set_ylabel("inercia", fontsize=25)
    plt.tight_layout()
    st.pyplot(fig2)

else:
    st.info("Por favor, sube un archivo CSV para comenzar el análisis")
