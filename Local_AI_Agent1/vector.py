from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("realistic_restaurant_reviews.csv")    # Read the csv data
embeddings = OllamaEmbeddings(model="mxbai-embed-large")    # Get the embedding model to be used

db_location = "./chroma_db"     # Prepare the vector database location
add_documents = not os.path.exists(db_location)

# Check if the location exists, if it doesnt, we prepare the data into documents
if add_documents:
    documents = []
    ids = []

    for i,row in df.iterrows():
        document = Document(
            page_content = row["Title"] + " " + row["Review"],
            metadata = {"rating": row["Rating"], "date": row["Date"]},
            id = str(i)
        )
        ids.append(str(i))
        documents.append(document)

# Initialise the vector store
vector_store = Chroma(
    collection_name= "restaurant_reviews",
    persist_directory= db_location,
    embedding_function= embeddings
)

# Again, if the database location does not exist, add the data to the vector database
if add_documents:
    vector_store.add_documents(documents= documents, ids= ids)

# A retriever that allows us retrieve relevant documents.
retriever = vector_store.as_retriever(
    search_kwargs = {"k": 5}
)