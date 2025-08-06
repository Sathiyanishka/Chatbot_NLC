from sentence_transformers import SentenceTransformer
import numpy as np

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Read the chunks from the text file
with open("nlc_chunks.txt", "r", encoding="utf-8") as file:
    raw_chunks = file.read()

# Split the chunks using <chunkendhere>
chunks = [chunk.strip() for chunk in raw_chunks.split("<chunkendhere>") if chunk.strip()]

# Generate embeddings
embeddings = model.encode(chunks)

# Save embeddings to a file
np.save("chunk_embeddings.npy", embeddings)

print("✅ 100 Embeddings generated and saved to 'chunk_embeddings.npy'")
