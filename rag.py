from knowledge import document_readin
from tools import document_splitter, vector_store, build_conversation_chain

# main
def RAG():
    documents = document_readin()
    chunks = document_splitter(documents)
    vectorstore = vector_store(chunks)
    conversation_chain = build_conversation_chain(vectorstore)
    return conversation_chain

if __name__ == '__main__':
    # query example
    conversation_chain = RAG()
    query = "Is there any resources mentioned scaling law? what are the titles and links of the resources?"
    result = conversation_chain.invoke({"question":query})
    print(result["answer"])