import streamlit as st
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.llms import LLM 
from typing import List
import torch
import time
from dotenv import load_dotenv
load_dotenv()

st.title("Local LLM RAG")

chatmodels_list = ["Llama-2-7b-chat-hf",
                   "deepseek-qwen-1.5B",
                   "Llama-3.1-8B-Instruct",
                   "Llama-3.2-3B",
                   "Llama-3.2-3B-Instruct"]

# Load local LLM
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../models/{chatmodels_list[0]}"))
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16, device_map="auto",max_memory={0: "6GB", "cpu": "10GB"})

# Define a custom LLM wrapper for our local model
class LocalLLM(LLM):
    def _call(self, prompt: str, stop: List[str] = None) -> str:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=1000)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    @property
    def _llm_type(self) -> str:
        return "custom-llm"

# Instantiate our local LLM
local_llm = LocalLLM()

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

# Define LLMChain using the local model
prompt_template = PromptTemplate(
    template="""
Answer the question(s) based on the given context only.
Please provide the most accurate answer to the given query.
<context>
{context}
<context>

Query: {input}


### Response:

""",
    input_variables=["context", "input"],
)

llm_chain = LLMChain(
    llm=local_llm, 
    prompt=prompt_template
)

retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, llm_chain)

prompt = st.text_input("Input your prompt here:")

if prompt:
    start_time = time.process_time()
    response = retrieval_chain.invoke({'input': prompt})
    print(f"Response Time: {time.process_time()-start_time}")
    st.write(response['text'].split("### Response:")[-1].strip())

    # with streamlit expander
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------------------------")
