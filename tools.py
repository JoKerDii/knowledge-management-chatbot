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

if __name__ == '__main__':
    from knowledge import document_readin
    documents = document_readin()
    chunks = document_splitter(documents)
    vectorstore = vector_store(chunks)
    conversation_chain = build_conversation_chain(vectorstore)
    print(conversation_chain)