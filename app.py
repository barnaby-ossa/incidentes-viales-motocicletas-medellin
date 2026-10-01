
import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Configuración
# --------------------------------------------------

st.set_page_config(
    page_title="Predicción de gravedad de incidentes viales",
    page_icon="🏍️",
    layout="centered"
)

st.title("Predicción de la gravedad de incidentes viales")
st.write(
    "Aplicación de aprendizaje automático para clasificar "
    "la gravedad de incidentes viales con motocicletas."
)


# --------------------------------------------------
# Cargar modelo
# --------------------------------------------------

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_bagging_ajustado.joblib")


modelo = cargar_modelo()


# --------------------------------------------------
# Obtener categorías aprendidas por el modelo
# --------------------------------------------------

preprocesador = modelo.named_steps["preprocesamiento"]

encoder = preprocesador.named_transformers_["cat"]

categorias_clase = encoder.categories_[0]
categorias_zona = encoder.categories_[1]
categorias_diseno = encoder.categories_[2]


# --------------------------------------------------
# Formulario de entrada
# --------------------------------------------------

st.subheader("Datos del incidente")

anio = st.number_input(
    "Año del incidente",
    min_value=2015,
    max_value=2021,
    value=2021,
    step=1
)

mes = st.number_input(
    "Mes",
    min_value=1,
    max_value=12,
    value=8,
    step=1
)

dia_semana = st.number_input(
    "Día de la semana (0=Lunes, 6=Domingo)",
    min_value=0,
    max_value=6,
    value=1,
    step=1
)

hora = st.number_input(
    "Hora",
    min_value=0,
    max_value=23,
    value=18,
    step=1
)

clase_incidente = st.selectbox(
    "Clase de incidente",
    categorias_clase
)

zona = st.selectbox(
    "Zona",
    categorias_zona
)

diseno_via = st.selectbox(
    "Diseño de vía",
    categorias_diseno
)


# --------------------------------------------------
# Predicción
# --------------------------------------------------

if st.button("Realizar predicción"):

    datos_entrada = pd.DataFrame({
        "AÑO_INCIDENTE": [anio],
        "MES": [mes],
        "DIA_SEMANA": [dia_semana],
        "HORA": [hora],
        "CLASE_INCIDENTE": [clase_incidente],
        "ZONA": [zona],
        "Diseño Vía": [diseno_via]
    })

    prediccion = modelo.predict(datos_entrada)[0]

    st.success(
        f"Gravedad predicha: **{prediccion}**"
    )
