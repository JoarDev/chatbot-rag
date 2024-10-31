import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

gemini_api_key = st.secrets["LLM_API_KEY"]

st.set_page_config(page_title="Chatbot FI UNJu", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Chatbot responde tus dudas del reglamento de tesis de la FI UNJu")
st.info("Hecho por Joaquin Ramos. La informacion del chatbot proviene del [PDF del reglamento de tesis](https://www.fi.unju.edu.ar/proyectos-finales.html)", icon="📃")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Preguntame sobre el reglamento de tesis de la FI UNJu",
        }
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    reader = SimpleDirectoryReader(input_dir="./rag-data", recursive=True)
    docs = reader.load_data()
    Settings.embed_model = GeminiEmbedding(api_key=gemini_api_key)
    Settings.llm = Gemini(api_key=gemini_api_key)
    # Settings.llm = OpenAI(
    #     model="gpt-3.5-turbo",
    #     temperature=0.2,
    #     system_prompt="""You are an expert on 
    #     the Streamlit Python library and your 
    #     job is to answer technical questions. 
    #     Assume that all questions are related 
    #     to the Streamlit Python library. Keep 
    #     your answers technical and based on 
    #     facts – do not hallucinate features.""",
    # )
    index = VectorStoreIndex.from_documents(docs)
    return index


index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )

if prompt := st.chat_input(
    "Escribe tu pregunta aquí"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        # Add response to message history
        st.session_state.messages.append(message)