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

if __name__ == '__main__':
    document_readin()