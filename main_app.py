import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io

# =============================
# Configuración inicial
# =============================
st.set_page_config(
    page_title="EDA Avanzado",
    layout="wide"
)

st.title("📊 Análisis Exploratorio de Datos (EDA Avanzado)")
st.markdown(
    """
    Este EDA permite **explorar, limpiar y analizar** un conjunto de datos
    mediante estadísticas, visualizaciones y relaciones entre variables.
    """
)

# =============================
# Carga de datos
# =============================
st.sidebar.header("📂 Cargar dataset")
uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Archivo cargado correctamente ✅")

    # =============================
    # Vista general
    # =============================
    st.header("🔍 Visión general")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", df.shape[0])
    c2.metric("Columnas", df.shape[1])
    c3.metric("Duplicados", df.duplicated().sum())
    c4.metric("Valores nulos", df.isnull().sum().sum())

    st.dataframe(df.head(10))

    # =============================
    # Información del dataset
    # =============================
    st.header("ℹ️ Información del dataset")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.code(buffer.getvalue(), language="text")

    # =============================
    # Tipos de variables
    # =============================
    st.header("🧬 Tipos de variables")

    dtypes_df = pd.DataFrame({
        "Columna": df.columns,
        "Tipo de dato": df.dtypes.astype(str)
    })
    st.dataframe(dtypes_df)

    # =============================
    # Valores nulos
    # =============================
    st.header("🧩 Análisis de valores nulos")

    nulls = df.isnull().sum().reset_index()
    nulls.columns = ["Columna", "Valores nulos"]
    nulls["Porcentaje"] = (nulls["Valores nulos"] / len(df)) * 100

    st.dataframe(nulls)

    fig_nulls = px.bar(
        nulls,
        x="Columna",
        y="Valores nulos",
        title="Valores nulos por columna"
    )
    st.plotly_chart(fig_nulls, use_container_width=True)

    # =============================
    # Estadísticas descriptivas
    # =============================
    st.header("📈 Estadísticas descriptivas")
    st.dataframe(df.describe(include="all").transpose())

    # =============================
    # Columnas numéricas
    # =============================
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # =============================
    # Análisis univariado numérico
    # =============================
    st.header("📊 Análisis univariado (Numérico)")

    if numeric_cols:
        num_col = st.selectbox("Selecciona variable numérica", numeric_cols)

        col1, col2 = st.columns(2)

        with col1:
            fig_hist = px.histogram(
                df, x=num_col, nbins=40, marginal="box",
                title=f"Distribución de {num_col}"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            fig_violin = px.violin(
                df, y=num_col, box=True,
                title=f"Violin plot de {num_col}"
            )
            st.plotly_chart(fig_violin, use_container_width=True)

        # Outliers
        Q1 = df[num_col].quantile(0.25)
        Q3 = df[num_col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[num_col] < Q1 - 1.5 * IQR) | (df[num_col] > Q3 + 1.5 * IQR)]

        st.info(f"🔎 Outliers detectados (IQR): **{outliers.shape[0]}**")

    # =============================
    # Análisis categórico
    # =============================
    st.header("🏷️ Análisis categórico")

    if cat_cols:
        cat_col = st.selectbox("Selecciona variable categórica", cat_cols)

        freq_df = df[cat_col].value_counts().reset_index()
        freq_df.columns = [cat_col, "Frecuencia"]

        fig_cat = px.bar(
            freq_df,
            x=cat_col,
            y="Frecuencia",
            title=f"Distribución de {cat_col}"
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    # =============================
    # Análisis bivariado
    # =============================
    st.header("🔀 Análisis bivariado")

    if len(numeric_cols) >= 2:
        x_col = st.selectbox("Variable X", numeric_cols, key="x")
        y_col = st.selectbox("Variable Y", numeric_cols, key="y")

        fig_scatter = px.scatter(
            df, x=x_col, y=y_col, trendline="ols",
            title=f"{x_col} vs {y_col}"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # =============================
    # Correlaciones
    # =============================
    st.header("🔗 Correlaciones")

    corr_method = st.radio(
        "Método de correlación",
        ["pearson", "spearman", "kendall"]
    )

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(method=corr_method)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

    # =============================
    # Exportar datos limpios
    # =============================
    st.header("📥 Exportar datos")

    csv = df.drop_duplicates().to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar dataset sin duplicados",
        data=csv,
        file_name="dataset_limpio.csv",
        mime="text/csv"
    )

else:
    st.info("⬅️ Sube un archivo CSV para iniciar el EDA")
