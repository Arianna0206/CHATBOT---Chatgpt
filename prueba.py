
import streamlit as st

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI


# Set up API token
API_TOKEN = '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'
os.environ['OPENAI_API_KEY'] = str(API_TOKEN)
data_dir = "data"
query = None

def cargar_datos():
    loader = DirectoryLoader('data', show_progress=True)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 200,
    )

    data = splitter.split_documents(docs)
    return data


def get_model():
    return ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

def initialize_bot():
    progress_bar = st.progress(0)
    progress_text = st.empty()
    progress_text.text("Leyendo documentos...")
    texts = cargar_datos()
    progress_bar.progress(25) 

    progress_text.text("Procesando documentos...")
    
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])


    vectordb = Chroma.from_documents(texts,embeddings)

    retriever = vectordb.as_retriever()

    progress_bar.progress(50) 

    progress_text.text("Entrenando bot...")
    model = get_model()
    preguntar = RetrievalQAWithSourcesChain.from_chain_type(
         llm = model,    
         retriever = retriever,
         return_source_documents=True,
         verbose=True,
         max_tokens_limit=4097
    )
    progress_bar.progress(75)   

    progress_text.text("Terminando detalles finales...")
    progress_bar.progress(100)  
    progress_text.text('Bot entrenado 🎉')
    progress_bar.empty()
    progress_text.empty()
    
    
    return preguntar


def save_uploaded_file(uploaded_file, save_path):
    file_path = os.path.join(save_path, uploaded_file.name)

    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error removing existing file: {e}")
            return False
    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False
    
# Function to obtain response from the LLM
def obtener_respuesta(pregunta, query):
    info_documentos = []
    
    response = query({"question": pregunta})
    print(response)

    respuesta = response.get('answer', 'No hubo respuesta')
    
    # Extracting document information
    nombre_documento = str(response.get('source_documents')[0].metadata['source'])
    similitud = response.get('source_documents')[0].page_content

    info_documentos.append(f"Documento: {nombre_documento}")
    info_documentos.append(f"Similitud: \n {similitud}")
    

    return respuesta, info_documentos
    

def upload_data(archivos): 
    for file in archivos:
        if file is not None:
            if save_uploaded_file(file, data_dir):
                file_uploaded_sucess = st.success(f"Archivo guardado en {data_dir}/{file.name}")
                file_uploaded_sucess.empty()
                st.session_state.file_uploaded = True 

def main():
    st.title("Hola soy Eli tu asistente, ¿En qué te puedo ayudar?")

    # Initialize session state variables
    if 'query' not in st.session_state:
        st.session_state.query = None
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False

    # File uploader
    uploaded_files = st.file_uploader("Cargar archivos", accept_multiple_files=True)

    # Process uploaded files
    if uploaded_files:
        for file in uploaded_files:
            if save_uploaded_file(file, data_dir):
                st.success(f"Archivo guardado en {data_dir}/{file.name}")
        st.session_state.file_uploaded = True

    # Button to train the bot
    if st.session_state.file_uploaded and st.button('Entrenar bot'):
        upload_data(uploaded_files)
        st.session_state.query = initialize_bot()

    # User question input
    pregunta_usuario = st.text_input("Hazme una pregunta:")

    # Respond to the question
    if pregunta_usuario and st.session_state.query:
        respuesta_chatbot, documentos_info = obtener_respuesta(pregunta_usuario, st.session_state.query)
        st.subheader("Respuesta del Chatbot:")
        st.write(respuesta_chatbot)

        st.markdown("#### Información del Documento:")
        for doc_info in documentos_info:
            st.markdown(f"* {doc_info}", unsafe_allow_html=True)

    
if __name__ == '__main__':
    main()
 

