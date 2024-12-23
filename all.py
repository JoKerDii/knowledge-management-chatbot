import glob
import os
from langchain.document_loaders import DirectoryLoader, TextLoader

# read in documents using LangChain's loaders
def document_readin():
    folders = glob.glob("knowledge-base/*")
    text_loader_kwargs = {'encoding': 'utf-8'}

    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

MODEL = "gpt-4o-mini"

load_dotenv()

# split document into chunks
def document_splitter(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
    print(f"Document types found: {', '.join(doc_types)}")
    return chunks

# put data chunks into a Vector Store that associates a Vector Embedding with each chunk
def vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    total_vectors = vectorstore.index.ntotal
    dimensions = vectorstore.index.d
    print(f"There are {total_vectors} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore

# conversation chain with RAG and Memory using Langchain
def build_conversation_chain(vectorstore):
    # create a new Chat with OpenAI
    llm = ChatOpenAI(temperature=0.7, model_name=MODEL)
    # set up the conversation memory for the chat
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    # the retriever is an abstraction over the VectorStore that will be used during RAG
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    # putting it together: set up the conversation chain with the GPT 4o-mini LLM, the vector store and memory
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
    return conversation_chain

# main
def RAG():
    documents = document_readin()
    chunks = document_splitter(documents)
    vectorstore = vector_store(chunks)
    conversation_chain = build_conversation_chain(vectorstore)
    return conversation_chain

import gradio as gr
# wrapping in a function - note that history isn't used, as the memory is in the conversation_chain
def chat(message, history):
    conversation_chain = RAG()
    result = conversation_chain.invoke({"question": message})
    return result["answer"]


view = gr.ChatInterface(chat, title="Personal Knowledge Management Chatbot").launch()
