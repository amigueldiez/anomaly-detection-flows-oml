# Detección de tráfico anómalo usando flujos de red con Online Machine Learning

Máster Universitario en Investigación en Ciberseguridad de la Universidad de León.

Este repositorio contiene los scripts que se han desarrollado durante la ejecución del proyecto que permiten determinar si un flujo de red es benigno o maligno con un algoritmo basado en novety detection usando online machine learning.

## Proceso de ejecución

1. Lo primero que se debe realizar es con el fichero `make_dataset.py` generar el dataset con el que se va trabajar. Esto se debe a que los datasets de flujos no están balanceados y, por lo tanto, las métricas que se obtendrían no serían las correctas. Este script permite elegir cuántos flujos coger del dataset principal y el porcentaje de muestras benignas y malignas.

2. Posteriormente, se aplica el preprocesado con el script `preprocessing.py`. Este script coge el conjunto de datos generado anteriormente y lo adapta para el modelo. Se puede configurar las características a utilizar de los flujos y el número de dimensiones a reducir.

3. Finalmente, se utiliza alguno de los Jupyter Notebooks que contienen el código necesario para ejecutar el modelo.

## Hiperparametrización

Se han desarrollado varios script para hiperparametrizar los modelos anteriores. En caso de querer probar el proyecto se recomienda usar esta parte pues es la que tiene las últimas modificaciones y permite obtener los mejores resultados.

Existen los siguientes scripts:

- `hiperparametrizacion_lof.py`: permite hiperparametrizar el modelo Local Outlier Factor
- `hiperparametrizacion_osvm_quantile.py`: permite hiperparametrizar el modelo One-class SVM con el QuantileFilter incluido
- `hiperparametrizacion_osvm.py`: permite hiperparametrizar el modelo One-class SVM sin ningún tipo de filtro. Se aconseja utilizar la solución anterior porque se obtienen mejores resultados.


## Autor

Alberto Miguel Diez - amigud00 \[at\] estudiantes \[dot\] unileon \[dot\] es

También se puede contactar al correo amigd \[at\] unileon \[dot\] es