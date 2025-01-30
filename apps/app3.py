import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import time
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']
st.title("Groq RAG")

# User selects input type
input_type = st.selectbox("Select input type:", ["PDF", "DOC", "TXT", "URL"])
uploaded_file = st.file_uploader("Upload a file:", type=["pdf", "docx", "txt"]) if input_type != "URL" else None
url_input = st.text_input("Enter URL:") if input_type == "URL" else None

if "vector" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cuda"})
    
    # Load document based on selected input type
    if input_type == "PDF" and uploaded_file:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.loader = PyPDFLoader(uploaded_file.name)
    elif input_type == "DOC" and uploaded_file:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.loader = UnstructuredWordDocumentLoader(uploaded_file.name)
    elif input_type == "TXT" and uploaded_file:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.loader = TextLoader(uploaded_file.name)
    elif input_type == "URL" and url_input:
        st.session_state.loader = WebBaseLoader(url_input)
    else:
        st.warning("Please upload a file or enter a URL")
        st.stop()
    
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    
    # Option to save embeddings
    save_embeddings = st.checkbox("Save embeddings locally")
    vector_db_path = "faiss_index"
    if save_embeddings:
        if os.path.exists(vector_db_path):
            st.session_state.vectors = FAISS.load_local(vector_db_path, st.session_state.embeddings)
        else:
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.session_state.vectors.save_local(vector_db_path)
    else:
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


llm = ChatGroq(api_key=groq_api_key, model="mixtral-8x7b-32768")

prompt = ChatPromptTemplate.from_template(
    template="""
Answer the question(s) based on the given context only.
Please provide the most accurate answer to the given query.
<context>
{context}
<context>

Queries: 
{input}"""
)

document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Input your prompt here:")

if prompt:
    start_time = time.process_time()
    response = retrieval_chain.invoke({'input': prompt})
    print(f"Response Time: {time.process_time()-start_time}")
    st.write(response['answer'])

    # with streamlit expander
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------------------------")


# import streamlit as st
# import os
# from langchain_groq import ChatGroq
# from langchain_community.document_loaders.web_base import WebBaseLoader  # Updated path
# from langchain_community.embeddings import HuggingFaceEmbeddings # OllamaEmbeddings  # Updated path for community version
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain  # Moved to langchain_core
# from langchain_community.vectorstores import FAISS
# from sentence_transformers import SentenceTransformer
# import time
# from dotenv import load_dotenv
# load_dotenv()

# groq_api_key = os.environ['GROQ_API_KEY']

# if "vector" not in st.session_state:
#     st.session_state.embeddings =  HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
#                                                          model_kwargs={"device": "cuda"})  # OllamaEmbeddings()
#     st.session_state.loader = WebBaseLoader("https://python.langchain.com/api_reference/langchain/index.html")
#     st.session_state.docs = st.session_state.loader.load()
#     st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
#     st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
#     st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

# st.title("Groq RAG")
# llm = ChatGroq(api_key=groq_api_key,
#                model="mixtral-8x7b-32768")
# prompt = ChatPromptTemplate.from_template(
#     template="""
# Answer the question(s) based on the given context only.
# Please provide the most accurate answer to the given query.
# <context>
# {context}
# <context>

# Queries: 
# {input}""")

# document_chain = create_stuff_documents_chain(llm=llm,prompt=prompt)
# retriever = st.session_state.vectors.as_retriever()
# retrieval_chain = create_retrieval_chain(retriever,document_chain)

# prompt = st.text_input("Input your prompt here:")

# if prompt:
#     start_time = time.process_time()
#     response = retrieval_chain.invoke({'input':prompt})
#     print(f"Response Time: {time.process_time()-start_time}")
#     st.write(response['answer'])

#     # with streamlit expander
#     with st.expander("Document Similarity Search"):
#         for i, doc in enumerate(response["context"]):
#             st.write(doc.page_content)
#             st.write("--------------------------------------------------")

