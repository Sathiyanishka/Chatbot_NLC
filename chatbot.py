import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load the transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load saved embeddings and text chunks
embeddings = np.load("chunk_embeddings.npy")

with open("nlc_chunks.txt", "r", encoding="utf-8") as f:
    text = f.read()
chunks = [c.strip() for c in text.split("<chunkendhere>") if c.strip()]

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(" AI Chatbot Ready! Ask your questions (type 'exit' to quit)")

# Chat loop
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Bye! Have a great day!")
        break

    query_embedding = model.encode([user_input])
    top_k = 3
    distances, indices = index.search(query_embedding, top_k)

    print("\nAI:")
    for i, idx in enumerate(indices[0]):
        print(f"{i+1}. {chunks[idx]}")
