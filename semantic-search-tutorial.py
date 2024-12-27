from langchain_community.document_loaders import PyPDFLoader

file_path = "./TSLA-Q3-2024-Update.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))
