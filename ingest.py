import voyageai
import chromadb
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 


docs = DirectoryLoader("docs", glob="**/*.txt", loader_cls=TextLoader).load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
chunks = text_splitter.split_documents(docs)

print(len(docs))
print(len(chunks))

load_dotenv()  # Load environment variables from .env file
voyageai.api_key = os.getenv("VOYAGE_API_KEY")

vo =voyageai.Client()
texts = [chunk.page_content for chunk in chunks]
result = vo.embed(texts=texts, model="voyage-4-lite", input_type="document")
print(len(result.embeddings))
print(len(result.embeddings[0])) # Print the first embedding vector


client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(name="tickets")
ids = [f"chunk-{i}" for i in range(len(chunks))]
metadatas = [chunk.metadata for chunk in chunks]

collection.upsert(
    ids=ids,
    embeddings=result.embeddings,
    documents=texts,
    metadatas=metadatas,
)
print(collection.count())

