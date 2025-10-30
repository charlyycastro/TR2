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

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Datos")
    st.dataframe(df.head())

    # ---- Selección de columnas (para reproducibilidad usa 'op' y 'co' si existen) ----
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

    # ---- Normalización MinMax (como en tu ejemplo) ----
    escalador = MinMaxScaler().fit(X.values)
    X_norm = pd.DataFrame(escalador.transform(X.values), columns=cols)

    st.subheader("Datos normalizados [0,1]")
    st.dataframe(X_norm.head())

    # ---- KMeans con k integrado (=3) y parámetros fijos para reproducir resultados ----
    kmeans = KMeans(
        n_clusters=3,         # k integrado
        init="k-means++",
        max_iter=300,
        n_init=10,
        random_state=42
    ).fit(X_norm.values)

    # Etiquetas
    X_norm["cluster"] = kmeans.labels_

    # ---- Centros e inercia ----
    centros_norm = pd.DataFrame(kmeans.cluster_centers_, columns=cols)
    centros_orig = pd.DataFrame(
        escalador.inverse_transform(kmeans.cluster_centers_),
        columns=cols
    )
    inercia = float(kmeans.inertia_)

    st.subheader("Centroides en espacio normalizado [0,1]")
    st.dataframe(centros_norm)

    st.subheader("Centroides en escala original")
    st.dataframe(centros_orig)

    st.write(f"**Inercia (SSE):** {inercia:.4f}")
    st.write(f"**Columnas usadas:** {cols}")

    # ---- Gráfica de clusters (en normalizado) ----
    fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=100)
    colores = ["red", "blue", "orange", "black", "purple", "pink", "brown"]

    for c in range(kmeans.n_clusters):
        mask = X_norm["cluster"] == c
        ax1.scatter(
            X_norm.loc[mask, cols[0]],
            X_norm.loc[mask, cols[1]],
            marker="o", s=180, color=colores[c], alpha=0.5, label=f"Cluster {c}"
        )
        ax1.scatter(
            kmeans.cluster_centers_[c][0],
            kmeans.cluster_centers_[c][1],
            marker="P", s=280, color=colores[c]
        )

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

    # ---- Método del codo (k=2..9) en datos normalizados ----
    inercias = []
    for k in range(2, 10):
        km_temp = KMeans(
            n_clusters=k,
            init="k-means++",
            max_iter=300,
            n_init=10,
            random_state=42
        ).fit(X_norm[cols].values)
        inercias.append(km_temp.inertia_)

    fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=100)
    ax2.plot(range(2, 10), inercias, marker="o")
    ax2.set_title("Método del Codo (k=2..9)")
    ax2.set_xlabel("numero de clusters")
    ax2.set_ylabel("inercia")
    plt.tight_layout()
    st.pyplot(fig2)

    # ---- (Opcional) Descargar resultados con cluster asignado ----
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
