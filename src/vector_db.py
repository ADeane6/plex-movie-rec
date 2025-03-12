import os
import chromadb
from chromadb.config import Settings
import logging

logger = logging.getLogger(__name__)

def setup_vector_db(movies_df, persist_directory="./chroma_db"):
    """Set up a ChromaDB vector database with movie embeddings"""
    # Ensure the directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    # Initialize ChromaDB with minimal settings to avoid errors
    logger.info(f"Initializing ChromaDB with persist_directory={persist_directory}")
    try:
        chroma_client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
    except TypeError as e:
        logger.warning(f"Error with ChromaDB settings: {e}. Trying with minimal settings.")
        # Try with minimal settings if the above fails
        chroma_client = chromadb.Client(Settings(
            persist_directory=persist_directory
        ))
    
    # Create or get collection
    logger.info("Creating or getting collection 'plex_movies'")
    collection = chroma_client.get_or_create_collection(name="plex_movies")
    
    # Prepare data for ChromaDB
    ids = [str(i) for i in range(len(movies_df))]
    embeddings = movies_df['embedding'].tolist()
    
    # Prepare metadata
    metadatas = []
    for _, row in movies_df.iterrows():
        metadata = {
            'title': row['title'],
            'year': str(row['year']) if row['year'] else "",
            'genres': ','.join(row['genres']),
            'key': row['key']
        }
        metadatas.append(metadata)
    
    # Prepare documents
    documents = movies_df['text_representation'].tolist()
    
    # Add documents to the collection
    logger.info(f"Adding {len(ids)} documents to collection")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )
    
    return collection

def query_vector_db(collection, query_embedding, n=5):
    """Query the vector database for similar movies"""
    if query_embedding is None:
        return []
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n
    )
    
    return results
