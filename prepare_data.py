import json
import time
from langchain_core.documents import Document
from langchain_chroma import Chroma
from chromadb.utils import embedding_functions

with open("movies_data.json", "r", encoding="utf-8") as f:
    movies = json.load(f)

def movie_to_text(movie):
    cast_names = [actor["name"] for actor in movie["cast"]]
    cast_with_chars = [f"{a['name']} as {a['character']}" for a in movie["cast"]]
    
    text = f"Title: {movie['title']}\n"
    text += f"Year: {movie['year']}\n"
    text += f"Director: {movie['director']}\n"
    text += f"Genres: {', '.join(movie['genres'])}\n"
    text += f"Rating: {movie['rating']}\n"
    text += f"Original Language: {movie['original_language']}\n"
    text += f"Cast: {', '.join(cast_with_chars)}\n"
    text += f"Overview: {movie['overview']}\n"
    
    text += f"\nSearch terms: movies starring {', '.join(cast_names)}, "
    text += f"directed by {movie['director']}, "
    text += f"{movie['title']}, {', '.join(movie['genres'])} movies"
    
    return text

documents = []
for movie in movies:
    text = movie_to_text(movie)
    cast_names = [actor["name"] for actor in movie["cast"]]
    doc = Document(
        page_content=text,
        metadata={
            "title": movie["title"],
            "movie_id": movie["id"],
            "year": movie["year"],
            "genres": ", ".join(movie["genres"]),
            "director": movie["director"],
            "cast": ", ".join(cast_names),
            "rating": movie["rating"]
        },
        id=str(movie["id"])
    )
    documents.append(doc)

print(f"Created {len(documents)} documents.")

print("Initializing ChromaDB's built-in ONNX embedding function...")

chroma_native_ef = embedding_functions.DefaultEmbeddingFunction()

class LangChainChromaBuiltInEmbeddings:
    def embed_documents(self, texts):
        return chroma_native_ef(texts)
    def embed_query(self, text):
        return chroma_native_ef([text])[0]

embeddings = LangChainChromaBuiltInEmbeddings()

print("Creating ChromaDB and embedding movies locally using built-in model...")
start_time = time.time()

doc_ids = [str(doc.metadata["movie_id"]) for doc in documents]
db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

end_time = time.time()
print(f"\nDone! Vectorized {len(documents)} movies in {end_time - start_time:.2f} seconds.")
print("Database securely saved to './chroma_db' without using external APIs or heavy libraries.")