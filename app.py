import gradio as gr
from rag import RAG


# wrapping in a function - note that history isn't used, as the memory is in the conversation_chain
def chat(message, history):
    conversation_chain = RAG()
    result = conversation_chain.invoke({"question": message})
    return result["answer"]


view = gr.ChatInterface(chat, title="Personal Knowledge Management Chatbot").launch()
