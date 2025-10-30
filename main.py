# -*- coding: utf-8 -*-
# refactorizado para que pueda funcionar con streamlit
# el archivo original esta en main

import pandas as pd
import numpy as np
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

# ---- Controles de KMeans (cambiables) ----
st.markdown("### Parámetros de K-Means")
c1, c2, c3, c4 = st.columns(4)

with c1:
    # Mapeo numérico a opciones válidas de scikit-learn
    init_num = st.number_input("ingresa el valor de la variable init:", min_value=1, max_value=2, value=1, step=1)
    INIT_MAP = {1: "k-means++", 2: "random"}
    init_value = INIT_MAP[int(init_num)]

with c2:
    max_iter = st.number_input("ingresa el valor de maximas iteraciones:", min_value=1, max_value=5000, value=300, step=10)

with c3:
    n_init = st.number_input("ingresa el valor de n_init:", min_value=1, max_value=1000, value=10, step=1)

with c4:
    random_state = st.number_input("ingresa el valor de random state:", min_value=0, max_value=100000, value=42, step=1)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Datos")
    st.dataframe(df.head())

    # ---- Usar 'op' y 'co' si existen; si no, primeras 2 numéricas ----
    if {"op", "co"}.issubset(df.columns):
        cols = ["op", "co"]
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            st.error("Se requieren al menos dos columnas numéricas.")
            st.stop()
        cols = num_cols[:2]
        st.info(f"No encontré columnas 'op' y 'co'. Se usarán: {cols}")

    X = df[cols].copy()

    # ---- Normalización MinMax ----
    escalador = MinMaxScaler().fit(X.values)
    X_norm = pd.DataFrame(escalador.transform(X.values), columns=cols)
    st.subheader("Datos normalizados [0,1]")
    st.dataframe(X_norm.head())

    # ---- KMeans con k=3 (integrado) y parámetros elegidos ----
    kmeans = KMeans(
        n_clusters=3,
        init=init_value,                 # ← usa el mapeo 1/2
        max_iter=int(max_iter),
        n_init=int(n_init),
        random_state=int(random_state)
    ).fit(X_norm.values)

    X_norm["cluster"] = kmeans.labels_

    # ---- Centroides e inercia ----
    centros_norm = pd.DataFrame(kmeans.cluster_centers_, columns=cols)
    centros_orig = pd.DataFrame(escalador.inverse_transform(kmeans.cluster_centers_), columns=cols)
    inercia = float(kmeans.inertia_)

    st.subheader("Centroides en espacio normalizado [0,1]")
    st.dataframe(centros_norm)

    st.subheader("Centroides en escala original")
    st.dataframe(centros_orig)

    st.write(f"**Inercia (SSE):** {inercia:.4f}")
    st.write(f"**Columnas usadas:** {cols}")
    st.write(f"**Parámetros:** init={init_value}, max_iter={int(max_iter)}, n_init={int(n_init)}, random_state={int(random_state)}")

    # ---- Gráfica de clusters (normalizado) ----
    fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=100)
    colores = ["red", "blue", "orange", "black", "purple", "pink", "brown"]
    for c in range(kmeans.n_clusters):
        mask = X_norm["cluster"] == c
        ax1.scatter(X_norm.loc[mask, cols[0]], X_norm.loc[mask, cols[1]],
                    marker="o", s=180, color=colores[c], alpha=0.5, label=f"Cluster {c}")
        ax1.scatter(kmeans.cluster_centers_[c][0], kmeans.cluster_centers_[c][1],
                    marker="P", s=280, color=colores[c])
    ax1.set_title("clientes", fontsize=20)
    ax1.set_xlabel("saldo en cuenta de ahorros", fontsize=15)
    ax1.set_ylabel("veces que uso tarjeta de credito", fontsize=15)
    ax1.text(1.15, 0.2, f"k={kmeans.n_clusters}", fontsize=15)
    ax1.text(1.15, 0.0, f"Inercia = {inercia:.2f}", fontsize=15)
    ax1.set_xlim(-0.1, 1.15)
    ax1.set_ylim(-0.1, 1.15)
    ax1.legend()
    plt.tight_layout()
    st.pyplot(fig1)

    # ---- Método del codo (k=2..9) usando los mismos parámetros ----
    inercias = []
    for k in range(2, 10):
        km_temp = KMeans(
            n_clusters=k,
            init=init_value,
            max_iter=int(max_iter),
            n_init=int(n_init),
            random_state=int(random_state)
        ).fit(X_norm[cols].values)
        inercias.append(km_temp.inertia_)

    fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=100)
    ax2.plot(range(2, 10), inercias, marker="o")
    ax2.set_title("Método del Codo (k=2..9)")
    ax2.set_xlabel("numero de clusters")
    ax2.set_ylabel("inercia")
    plt.tight_layout()
    st.pyplot(fig2)

    # ---- Descargar resultados ----
    df_out = df.copy()
    df_out["cluster"] = kmeans.labels_
    st.download_button(
        "⬇️ Descargar CSV con clusters",
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name="analisis_clusterizado.csv",
        mime="text/csv"
    )

else:
    st.info("Por favor, sube un archivo CSV para comenzar el análisis")
