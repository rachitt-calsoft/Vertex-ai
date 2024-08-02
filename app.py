import os
from typing import List
import streamlit as st
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    Document,
    PromptTemplate,
    StorageContext
)
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Set up environment variables
os.environ["GOOGLE_API_KEY"] = "AIzaSyDjvW_o6-C5SvO5LNGJkdUfyrvzqXDYPzs"

# Available models
MODELS = {
    "Gemini 1.5 Flash": "models/gemini-1.5-flash-latest",
    "Gemini 1.5 Pro": "models/gemini-pro-vision",
    "Gemini-1.0 Pro": "models/gemini-1.5-pro-latest",
}

EMBED_MODEL = "models/embedding-001"
INPUT_DIR = "/docs"
CHROMA_DIR = "./chroma_db"

def initialize_llm(model_name: str) -> Gemini:
    return Gemini(model_name=model_name, temperature=0.5, max_tokens=800)

def initialize_embedding_model(model_name: str) -> GeminiEmbedding:
    return GeminiEmbedding(model_name=model_name)

def load_documents() -> List[Document]:
    reader = SimpleDirectoryReader(input_dir=INPUT_DIR)
    documents = reader.load_data()
    return documents

def create_or_load_index(llm: Gemini, embed_model: GeminiEmbedding) -> VectorStoreIndex:
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = client.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    if not os.path.exists(CHROMA_DIR) or len(os.listdir(CHROMA_DIR)) == 0:
        st.info("Creating new vector store. This may take a moment...")
        documents = load_documents()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        parser = SimpleNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes, storage_context=storage_context)
    else:
        st.info("Loading existing vector store...")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store)
    
    return index

def create_prompt_template():
    template = (
        """You are an assistant for question-answering tasks.
        Use the following context to answer the question.
        If you don't know the answer, just say that you don't know.
        Use five sentences maximum and keep the answer concise.

        Question: {query_str}
        Context: {context_str}
        Answer:"""
    )
    return PromptTemplate(template)

def main():
    st.title("Document Q&A with LlamaIndex and Google LLMs")

    # Model selection
    llm_model = st.selectbox("Select LLM Model", list(MODELS.keys()))

    # Initialize models
    llm = initialize_llm(MODELS[llm_model])
    embed_model = initialize_embedding_model(EMBED_MODEL)

    # Create or load index
    index = create_or_load_index(llm, embed_model)

    # Create prompt template
    prompt_template = create_prompt_template()

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Create query engine with custom prompt template
            query_engine = index.as_query_engine(text_qa_template=prompt_template)
            
            # Execute the query and obtain the response
            response = query_engine.query(prompt)
            
            # Assuming the response object has a method or attribute to get the full response text
            # You need to replace 'response_attribute' with the actual attribute or method name
            full_response = response  # Adjust this line based on the actual response structure
            
            st.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
