import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Step 1: Load the sentence-transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 2: Load embeddings
embeddings = np.load("chunk_embeddings.npy")

# Step 3: Create FAISS index
dimension = embeddings.shape[1]  # Typically 384 for MiniLM
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Step 4: Load original text chunks
with open("nlc_chunks.txt", "r", encoding="utf-8") as file:
    text = file.read()
chunks = [chunk.strip() for chunk in text.split("<chunkendhere>") if chunk.strip()]

# Step 5: Get user input query
query = input("Enter your question: ")
query_embedding = model.encode([query])

# Step 6: Search FAISS index
top_k = 3
distances, indices = index.search(query_embedding, top_k)

# Step 7: Show top results
print("\n Top matching chunks:")
for i, idx in enumerate(indices[0]):
    print(f"\nResult {i+1}:")
    print(chunks[idx])

