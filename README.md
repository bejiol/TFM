# TFM

En este repositorio se encuentra alojado el código del Trabajo de Fin de Máster titulado "Detección offline de subtemas en Twitter durante eventos", realizado en el curso 2017/2018. Ha sido llevado a cabo para el Máster en Ingeniería Informática, de la Facultad de Informática de la Universidad Complutense de Madrid.

### Alumna:
* Beatriz Jiménez del Olmo

### Director:
* Rafael Caballero Roldán (Departamento de Sistemas Informáticos y Computación)

### Descripción de los archivos clave:
- **twitter-listener.py** : Script para la recopilación de los tweets del evento a analizar, realizado por el profesor Enrique Martín Martín.
- **credenciales-twitter.py** : Archivo necesario por **twitter-listener.py** donde se definirán las credenciales de acceso necesarias para poder usar la API de Twitter. Estas credenciales serán accesibles una ves se configure una  [aplicación de Twitter](https://apps.twitter.com/).
- **tweet-import-preprocess.py**: Script que importa los tweets recopilados a una base de datos en MongoDB y aplica las operaciones de preprocesado sobre los tweets recogidas en **preprocesado.js**. 
- **tweet-analysis.py**: Script encargado de realizar la fase de detección de temas, una vez se haya especificado el intervalo temporal a estudiar del evento. Hace uso de los archivos adicionales **dateTimePicker.py** y **scroll.py**.
- **config.json**: Fichero de configuración necesario por el script **tweet-analysis.py** para la fase de detección de temas.
- **tweet-time-clustering.py**: Script encargado del proceso de agrupación y representación de los temas según proximidad temporal, generando "momentos" de distintos niveles de concentración de temas.
- **tweet-textual-clustering.py**: Script responsable de la agrupación por relación textual de los temas detectados, representando visualmente los clusters temáticos generados.





