import chromadb
import voyageai
import ollama
from dotenv import load_dotenv

load_dotenv()
THRESHOLD = 1.2

vo = voyageai.Client()
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="tickets")

ticket = "User cannot log in to App XYZ after a password change"

# Embed the ticket with the SAME model, as a query
query_vector = vo.embed([ticket], model="voyage-4-lite", input_type="query").embeddings[0]

# Search for the 3 closest chunks
results = collection.query(query_embeddings=[query_vector], n_results=3)

best_distance = results["distances"][0][0]
print("Best distance:", best_distance)
if best_distance > THRESHOLD:
    print("I don't know")
else:
    
    context = "\n\n".join(results["documents"][0])

    prompt = f"""You are a support helper. Use the context below to suggest a fix for the ticket.
Say "I don't know" only if the context has nothing related to the ticket.
Give short steps, and say which document you used.

Context:
{context}

Ticket:
{ticket}
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt}],
    )

    print("Answer:", response["message"]["content"])
    print("Sources:")
    for i in range(3):
        print("-", results["metadatas"][0][i]["source"])