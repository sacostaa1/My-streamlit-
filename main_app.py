import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# -----------------------------
# Configuración inicial
# -----------------------------
st.set_page_config(
    page_title="EDA - Análisis Exploratorio de Datos",
    layout="wide"
)

st.title("📊 Análisis Exploratorio de Datos (EDA)")
st.write(
    """
    Aplicación interactiva para explorar, analizar y visualizar conjuntos de datos
    usando **Streamlit**.
    """
)

# -----------------------------
# Carga de datos
# -----------------------------
st.sidebar.header("📂 Cargar datos")
uploaded_file = st.sidebar.file_uploader(
    "Sube tu archivo CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("Archivo cargado correctamente ✅")

    # -----------------------------
    # Vista general del dataset
    # -----------------------------
    st.subheader("🔍 Vista general de los datos")

    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", df.shape[0])
    col2.metric("Columnas", df.shape[1])
    col3.metric("Valores nulos", df.isnull().sum().sum())

    st.dataframe(df.head())

    # -----------------------------
    # Información del dataset
    # -----------------------------
    st.subheader("ℹ️ Información del dataset")

    buffer = []
    df.info(buf=buffer)
    st.text("\n".join(buffer))

    # -----------------------------
    # Estadísticas descriptivas
    # -----------------------------
    st.subheader("📈 Estadísticas descriptivas")
    st.dataframe(df.describe())

    # -----------------------------
    # Análisis de valores nulos
    # -----------------------------
    st.subheader("🧩 Valores nulos por columna")

    nulls = df.isnull().sum()
    nulls_df = pd.DataFrame({
        "Columna": nulls.index,
        "Valores nulos": nulls.values
    })

    fig_nulls = px.bar(
        nulls_df,
        x="Columna",
        y="Valores nulos",
        title="Valores nulos por columna"
    )
    st.plotly_chart(fig_nulls, use_container_width=True)

    # -----------------------------
    # Selección de columnas numéricas
    # -----------------------------
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.subheader("📊 Análisis de variables numéricas")

    if numeric_cols:
        selected_col = st.selectbox(
            "Selecciona una variable numérica",
            numeric_cols
        )

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Distribución**")
            fig_hist = px.histogram(
                df,
                x=selected_col,
                nbins=30,
                title=f"Distribución de {selected_col}"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            st.write("**Boxplot**")
            fig_box = px.box(
                df,
                y=selected_col,
                title=f"Boxplot de {selected_col}"
            )
            st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.warning("No se encontraron columnas numéricas.")

    # -----------------------------
    # Matriz de correlación
    # -----------------------------
    st.subheader("🔗 Matriz de correlación")

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()

        fig_corr = plt.figure(figsize=(10, 6))
        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            fmt=".2f"
        )
        st.pyplot(fig_corr)
    else:
        st.info("No hay suficientes variables numéricas para calcular correlaciones.")

else:
    st.info("⬅️ Sube un archivo CSV desde la barra lateral para comenzar el análisis.")

