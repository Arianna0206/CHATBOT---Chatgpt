# CHATBOT---Chatgpt

## 1 Propósito del Chatbot

El Chatbot Eli es una herramienta diseñada para proporcionar respuestas a preguntas
específicas basadas en documentos previamente cargados. Este manual te guiará a través de
los pasos necesarios para utilizar el chatbot de manera efectiva.

## 2. Requisitos Previos

Antes de comenzar, es necesario tener lo siguiente:

* Acceso a los archivos de datos necesarios y una conexión a Internet.
* Tener instalado entre las versiones Python 3.8 y Python 3.11 <https://www.python.org/downloads/>
* Tener instalado el manejador de paquetes pip <https://pip.pypa.io/en/stable/installation/>

## 3. Configuración del entorno de ejecución

Como paso opcional se puede establecer un entorno virtual utilizando virtualenv y conda que nos permiten crear estos entornos, uno desde la terminal y el otro desde una interfaz gráfica. Este paso es recomendable, en el caso de que estemos trabajando con versiones distintas de las librerías que se utilizan, y así evitar los conflictos con otros proyectos que estén usando librerías similares.

## 4. Instalaciones necesarias

```bash
  pip install streamlit
  pip install langchain
  pip install openai
  pip install chromadb
  pip install unstructured
  pip install “unstructured[pdf]”
  pip install tiktoken
```

## 5. Ejecución del chatbot

```bash
  streamlit run nombrearchivo.py
```
