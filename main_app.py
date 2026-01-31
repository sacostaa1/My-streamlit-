import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
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
    "EDA interactivo, robusto y estable (compatible con Python 3.13 y Streamlit Cloud)."
)

# =============================
# Funciones auxiliares
# =============================
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Forzar unicidad
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        idxs = cols[cols == dup].index.tolist()
        for i, idx in enumerate(idxs):
            if i != 0:
                cols[idx] = f"{dup}_{i}"
    df.columns = cols
    return df

# =============================
# Carga de datos
# =============================
st.sidebar.header("📂 Cargar dataset")
uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = clean_column_names(df)

    st.success("Archivo cargado y columnas normalizadas ✅")

    # =============================
    # Visión general
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
    # Tipos de datos
    # =============================
    st.header("🧬 Tipos de variables")
    st.dataframe(
        pd.DataFrame({
            "Columna": df.columns,
            "Tipo de dato": df.dtypes.astype(str)
        })
    )

    # =============================
    # Valores nulos
    # =============================
    st.header("🧩 Análisis de valores nulos")

    nulls = (
        df.isnull()
        .sum()
        .reset_index()
        .rename(columns={"index": "columna", 0: "nulos"})
    )
    nulls["porcentaje"] = (nulls["nulos"] / len(df)) * 100
    st.dataframe(nulls)

    fig_nulls = px.bar(
        nulls,
        x="columna",
        y="nulos",
        title="Valores nulos por columna"
    )
    st.plotly_chart(fig_nulls, use_container_width=True)

    # =============================
    # Estadísticas descriptivas
    # =============================
    st.header("📈 Estadísticas descriptivas")
    st.dataframe(df.describe(include="all").transpose())

    # =============================
    # Columnas por tipo
    # =============================
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # =============================
    # Análisis univariado numérico
    # =============================
    st.header("📊 Análisis univariado (numérico)")

    if numeric_cols:
        num_col = st.selectbox("Selecciona variable numérica", numeric_cols)

        fig = px.histogram(
            df,
            x=num_col,
            nbins=40,
            marginal="box",
            title=f"Distribución de {num_col}"
        )
        st.plotly_chart(fig, use_container_width=True)

        Q1 = df[num_col].quantile(0.25)
        Q3 = df[num_col].quantile(0.75)
        IQR = Q3 - Q1

        outliers = df[
            (df[num_col] < Q1 - 1.5 * IQR) |
            (df[num_col] > Q3 + 1.5 * IQR)
        ]

        st.info(f"🔎 Outliers detectados (IQR): {outliers.shape[0]}")

    # =============================
    # Análisis categórico
    # =============================
    st.header("🏷️ Análisis categórico")

    if cat_cols:
        cat_col = st.selectbox("Selecciona variable categórica", cat_cols)

        freq = df[cat_col].value_counts().reset_index()
        freq.columns = [cat_col, "frecuencia"]

        fig_cat = px.bar(
            freq,
            x=cat_col,
            y="frecuencia",
            title=f"Distribución de {cat_col}"
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    # =============================
    # Análisis bivariado (SIN NARWHALS)
    # =============================
    st.header("🔀 Análisis bivariado")

    if len(numeric_cols) >= 2:
        x_col = st.selectbox("Variable X", numeric_cols, key="x")
        y_col = st.selectbox("Variable Y", numeric_cols, key="y")

        add_trend = st.checkbox("Agregar línea de tendencia (OLS)", value=False)

        plot_df = df[[x_col, y_col]].dropna()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=plot_df[x_col],
                y=plot_df[y_col],
                mode="markers",
                name="Datos"
            )
        )

        if add_trend and len(plot_df) > 2:
            x = plot_df[x_col].values
            y = plot_df[y_col].values

            m, b = np.polyfit(x, y, 1)
            y_pred = m * x + b

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_pred,
                    mode="lines",
                    name="OLS"
                )
            )

            st.info(f"📐 Ecuación: y = {m:.4f}x + {b:.4f}")

        fig.update_layout(
            title=f"{x_col} vs {y_col}",
            xaxis_title=x_col,
            yaxis_title=y_col
        )

        st.plotly_chart(fig, use_container_width=True)

    # =============================
    # Correlaciones
    # =============================
    st.header("🔗 Correlaciones")

    if len(numeric_cols) >= 2:
        method = st.radio(
            "Método de correlación",
            ["pearson", "spearman", "kendall"]
        )

        corr = df[numeric_cols].corr(method=method)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

    # =============================
    # Exportar datos limpios
    # =============================
    st.header("📥 Exportar datos limpios")

    clean_df = df.drop_duplicates()
    st.download_button(
        label="Descargar dataset limpio",
        data=clean_df.to_csv(index=False).encode("utf-8"),
        file_name="dataset_limpio.csv",
        mime="text/csv"
    )

else:
    st.info("⬅️ Sube un archivo CSV para iniciar el EDA")
