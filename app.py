import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


st.set_page_config(page_title="Portfolio Assistant", page_icon="💼", layout="centered")
st.title("💼 Virtual Portfolio Assistant")
st.caption("Powered by RAG Architecture")

PDF_FILE = "priya_sharma_full_profile.pdf"

@st.cache_resource
def initialize_vector_store():
    """
    Initializes and caches the vector database from the PDF source.
    """
    if not os.path.exists(PDF_FILE):
        return None
        
    try:
        loader = PyPDFLoader(PDF_FILE)
        documents = loader.load()
        
       
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(chunks, embeddings)
        return vector_db
        
    except Exception as e:
        st.error(f"Failed to initialize vector store: {str(e)}")
        return None

def main():
    vector_db = initialize_vector_store()
    
    if not vector_db:
        st.error("Document source not found. Please verify the PDF file exists.")
        return

   
    user_query = st.chat_input("Inquire about professional background, skills, or projects...")
    
    if user_query:
      
        results = vector_db.similarity_search(user_query, k=1)
        
        with st.chat_message("user"):
            st.write(user_query)
            
        with st.chat_message("assistant"):
            if results:
                st.markdown("### Relevant Context")
                st.info(results[0].page_content)
            else:
                st.warning("No relevant information found in the portfolio context.")

if __name__ == "__main__":
    main()
