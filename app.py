import streamlit as st
import torch
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL = 'tinyllama'

llm = ChatOllama(model=MODEL, num_gpu=1)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': device})


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def process_pdf(pdf_file):
    
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.getbuffer())
    
    
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()
    
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store


template="""

    <|system|>
    You are an AI assistant tasked with answering questions based on the provided PDF document. Your primary goal is to provide accurate and helpful information.

    Important guidelines:
    1. Only provide information that is explicitly stated in or can be directly inferred from the document.
    2. If the answer is not in the document or you're unsure, say "I'm sorry, but I don't have enough information from the document to answer that question accurately."
    3. Do not make up or infer information that is not supported by the document.
    4. If asked about topics not related to the document, politely redirect the conversation back to the document's content.
    5. Be concise in your responses while ensuring clarity and completeness.
    ------

    Chat History:
    {chat_history}

    Context:
    {context}

    <|user|>
    {question}

    """

prompt_template = PromptTemplate(
    input_variables=["chat_history", "question", "context"],
    template=template
)


st.title("PDF Chatbot")


st.sidebar.title("Upload PDF")
pdf_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if pdf_file is not None:
    st.sidebar.success("PDF uploaded successfully!")
    if st.sidebar.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            vector_store = process_pdf(pdf_file)
        st.sidebar.success("PDF processed successfully!")
        st.session_state.vector_store = vector_store
else:
    st.sidebar.info("Please upload a PDF file to start.")

# Main chat interface
if 'vector_store' in st.session_state:
    
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer', input_key='question')

    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=st.session_state.vector_store.as_retriever(),
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        output_key='answer',
        get_chat_history=lambda h : h,
        verbose = False
    )

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a question about the PDF"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            response = chain({"question":prompt}, callbacks=[stream_handler])
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})

    
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.sidebar.success("Chat history cleared!")
else:
    st.info("Upload and process a PDF to start chatting.")