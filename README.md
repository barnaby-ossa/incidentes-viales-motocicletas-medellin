
# Clasificación de la gravedad de los incidentes viales con motocicletas en Medellín mediante aprendizaje automático

## Descripción

Este proyecto desarrolla un modelo de aprendizaje automático para clasificar la gravedad de los incidentes viales con motocicletas registrados en Medellín.

El problema corresponde a una clasificación multiclase, en la que se predicen tres categorías de gravedad:

- HERIDO
- SOLO DAÑOS
- MUERTO

El proyecto se desarrolla siguiendo la metodología CRISP-DM.

## Datos

El conjunto de datos contiene registros de incidentes viales con motocicletas correspondientes al periodo 2015–2021.

Las variables utilizadas por el modelo son:

- AÑO_INCIDENTE
- MES
- DIA_SEMANA
- HORA
- CLASE_INCIDENTE
- ZONA
- Diseño Vía

La variable objetivo es:

- GRAVEDAD_INCIDENTE

La variable DIRECCION no se utiliza como predictor debido a su alta cardinalidad.

## Modelamiento

Se evaluaron cinco modelos de aprendizaje automático:

- Regresión Logística
- SVM
- Red Neuronal
- KNN
- Árbol de Decisión

También se evaluaron métodos de ensamble:

- Voting
- Bagging
- Boosting

La métrica principal utilizada fue F1-macro debido al desbalance existente entre las clases.

El modelo final corresponde a un Bagging ajustado mediante validación cruzada.

### Hiperparámetros finales

- n_estimators = 30
- max_samples = 1.0

## Evaluación

El modelo final obtuvo en el conjunto de prueba:

- Accuracy: 0.824365
- Precision macro: 0.594823
- Recall macro: 0.613655
- F1-macro: 0.603411

## Aplicación

La aplicación fue desarrollada utilizando Streamlit.

Permite ingresar las características de un nuevo incidente y obtener la categoría de gravedad predicha por el modelo.

## Archivos principales

- `app.py`: aplicación Streamlit.
- `modelo_bagging_ajustado.joblib`: modelo completo serializado.
- `requirements.txt`: dependencias y versiones utilizadas.
- `README.md`: documentación del proyecto.

## Ejecución local

Instalar las dependencias:

```bash
pip install -r requirements.txt
