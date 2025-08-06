import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

#  Step 1: Configure your free Gemini API key
genai.configure(api_key="AIzaSyBWtYjYm1o10YrL32VCOjOj17Qs8bsEAPk")

#  Step 2: Load Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")

#  Step 3: Load embedding model
encoder = SentenceTransformer("all-MiniLM-L6-v2")

#  Step 4: Load your FAISS index
embeddings = np.load("chunk_embeddings.npy")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 📄 Step 5: Load your document chunks
with open("nlc_chunks.txt", "r", encoding="utf-8") as file:
    full_text = file.read()
chunks = [chunk.strip() for chunk in full_text.split("<chunkendhere>") if chunk.strip()]

# 💬 Step 6: Chat loop
print("\nGemini Chatbot: Ask me anything about your PDF! (type 'exit' to quit)\n")

while True:
    query = input(" You: ")
    if query.lower() in ['exit', 'quit']:
        print(" Bye!")
        break

    #  Embed and search top chunks
    query_vector = encoder.encode([query])
    distances, indices = index.search(query_vector, 3)

    # Prepare context
    context = "\n\n".join([chunks[i] for i in indices[0]])

    #  Ask Gemini using context
    prompt = f"""You are a helpful assistant. Based on the context below, answer the question as accurately as possible.

Context:
{context}

Question: {query}
Answer:"""

    try:
        response = model.generate_content(prompt)
        print("\n Prompt:\n```\n", prompt, "\n```\n")
        print("\n Gemini:", response.text.strip(), "\n")
    except Exception as e:
        print("Error from Gemini:", e)
