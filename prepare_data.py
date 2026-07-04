import json
import time
from langchain_core.documents import Document
from langchain_chroma import Chroma
# Import Chroma's native embedding function
from chromadb.utils import embedding_functions

# ==========================================
# 1. LOAD AND PROCESS JSON MOVIE DATA
# ==========================================

with open("movies_data.json", "r", encoding="utf-8") as f:
    movies = json.load(f)

def movie_to_text(movie):
    text = f"Title: {movie['title']}\n"
    text += f"Year: {movie['year']}\n"
    text += f"Director: {movie['director']}\n"
    text += f"Genres: {', '.join(movie['genres'])}\n"
    text += f"Rating: {movie['rating']}\n"
    text += f"Original Language: {movie['original_language']}\n"

    cast_list = []
    for actor in movie["cast"]:
        cast_list.append(f"{actor['name']} as {actor['character']}")
    text += f"Cast: {', '.join(cast_list)}\n"

    text += f"Overview: {movie['overview']}"
    return text

# Process all movies into LangChain Documents[cite: 1]
documents = []
for movie in movies:
    text = movie_to_text(movie)
    doc = Document(
        page_content=text,
        metadata={
            "title": movie["title"],
            "movie_id": movie["id"],
            "year": movie["year"],
            "genres": ", ".join(movie["genres"]),
            "director": movie["director"],
            "rating": movie["rating"]
        }
    )
    documents.append(doc)

print(f"Created {len(documents)} documents.")

# ==========================================
# 2. CHROMADB BUILT-IN EMBEDDINGS (ONNX)
# ==========================================

print("Initializing ChromaDB's built-in ONNX embedding function...")

# 1. Grab Chroma's default embedding tool (bypasses sentence-transformers library completely)
chroma_native_ef = embedding_functions.DefaultEmbeddingFunction()

# 2. Wrap it so LangChain's Chroma vector store understands it
class LangChainChromaBuiltInEmbeddings:
    def embed_documents(self, texts):
        return chroma_native_ef(texts)
    def embed_query(self, text):
        return chroma_native_ef([text])[0]

embeddings = LangChainChromaBuiltInEmbeddings()

print("Creating ChromaDB and embedding movies locally using built-in model...")
start_time = time.time()

# This compiles and runs the embeddings locally via ONNX runtime
db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

end_time = time.time()
print(f"\nDone! Vectorized {len(documents)} movies in {end_time - start_time:.2f} seconds.")
print("Database securely saved to './chroma_db' without using external APIs or heavy libraries.")