import numpy as np
import time
import os
import httpx
import logging
import pickle
import pandas as pd

logger = logging.getLogger(__name__)

def save_embeddings(movies_df, file_path="cached_embeddings.pkl"):
    """Save movie embeddings to a pickle file"""
    logger.info(f"Saving embeddings to {file_path}")
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Extract essential data to save
        cache_data = {
            'movie_keys': movies_df['key'].tolist(),
            'embeddings': movies_df['embedding'].tolist()
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Successfully saved embeddings for {len(movies_df)} movies")
        return True
    except Exception as e:
        logger.error(f"Error saving embeddings: {str(e)}")
        return False

def load_embeddings(file_path="cached_embeddings.pkl"):
    """Load movie embeddings from a pickle file"""
    logger.info(f"Loading embeddings from {file_path}")
    try:
        if not os.path.exists(file_path):
            logger.info(f"No cached embeddings found at {file_path}")
            return None
        
        with open(file_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        logger.info(f"Successfully loaded embeddings for {len(cache_data['movie_keys'])} movies")
        return cache_data
    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}")
        return None

def apply_cached_embeddings(movies_df, cache_data):
    """Apply cached embeddings to the movies dataframe"""
    logger.info("Applying cached embeddings to movies dataframe")
    try:
        # Make sure we have a copy of the DataFrame
        movies_df = movies_df.copy()
        
        # Make sure the embedding column exists
        if 'embedding' not in movies_df.columns:
            movies_df['embedding'] = None
        
        # Create a mapping of movie keys to embeddings
        embedding_map = {key: emb for key, emb in zip(cache_data['movie_keys'], cache_data['embeddings'])}
        
        # Apply embeddings to the dataframe
        for idx, row in movies_df.iterrows():
            if row['key'] in embedding_map:
                movies_df.at[idx, 'embedding'] = embedding_map[row['key']]
        
        # Count how many movies got embeddings
        embedded_count = movies_df['embedding'].notna().sum()
        logger.info(f"Applied embeddings to {embedded_count} out of {len(movies_df)} movies")
        
        return movies_df, embedded_count
    except Exception as e:
        logger.error(f"Error applying cached embeddings: {str(e)}")
        return movies_df, 0

def generate_embeddings(movies_df, api_key, batch_size=20, model="text-embedding-ada-002", 
                        cache_file="cached_embeddings.pkl", use_cache=True):
    """Generate embeddings for movie text representations using the latest OpenAI API"""
    from openai import OpenAI
    import pandas as pd
    
    if not api_key:
        raise ValueError("OpenAI API key is required for generating embeddings")
    
    # Make sure we have a copy of the DataFrame
    movies_df = movies_df.copy()
    
    # Make sure the embedding column exists
    if 'embedding' not in movies_df.columns:
        movies_df['embedding'] = None
    
    # Check if we should use cached embeddings
    if use_cache:
        cache_data = load_embeddings(cache_file)
        if cache_data:
            # Apply cached embeddings
            embedding_map = {key: emb for key, emb in zip(cache_data['movie_keys'], cache_data['embeddings'])}
            
            # Create a new column with the embeddings
            movies_df['embedding'] = movies_df['key'].map(lambda k: embedding_map.get(k))
            
            # Count how many movies got embeddings
            embedded_count = movies_df['embedding'].notna().sum()
            logger.info(f"Applied cached embeddings to {embedded_count} out of {len(movies_df)} movies")
            
            # If all movies have embeddings, return early
            if embedded_count == len(movies_df):
                logger.info("All movies have cached embeddings, skipping API calls")
                return movies_df
            
            # Filter out movies that already have embeddings
            movies_to_embed_indices = movies_df[movies_df['embedding'].isna()].index
            logger.info(f"Need to generate embeddings for {len(movies_to_embed_indices)} new movies")
            
            if len(movies_to_embed_indices) == 0:
                return movies_df
        else:
            movies_to_embed_indices = movies_df.index
    else:
        movies_to_embed_indices = movies_df.index
    
    logger.info(f"Generating embeddings for {len(movies_to_embed_indices)} movies with batch size {batch_size}")
    
    # Create a custom httpx client without proxies
    http_client = httpx.Client(
        timeout=60.0,
        follow_redirects=True
    )
    
    # Initialize OpenAI client with the custom httpx client
    client = OpenAI(
        api_key=api_key,
        http_client=http_client
    )
    
    # Process in batches to avoid rate limits
    for i in range(0, len(movies_to_embed_indices), batch_size):
        batch_indices = movies_to_embed_indices[i:i+batch_size]
        batch_df = movies_df.loc[batch_indices]
        batch = batch_df['text_representation'].tolist()
        
        logger.info(f"Processing batch {i}-{i+min(batch_size, len(movies_to_embed_indices)-i)} of {len(movies_to_embed_indices)}")
        
        try:
            response = client.embeddings.create(
                input=batch,
                model=model
            )
            batch_embeddings = [item.embedding for item in response.data]
            
            # Assign embeddings directly to the DataFrame
            for j, idx in enumerate(batch_indices):
                if j < len(batch_embeddings):
                    movies_df.at[idx, 'embedding'] = batch_embeddings[j]
            
            logger.info(f"Successfully generated {len(batch_embeddings)} embeddings")
            
            # Sleep to avoid rate limits
            if i + batch_size < len(movies_to_embed_indices):
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error generating embeddings for batch {i}-{i+batch_size}: {str(e)}")
            # We don't add None values, just leave them as they are
    
    # Remove rows with failed embeddings
    before_count = len(movies_df)
    movies_df = movies_df.dropna(subset=['embedding'])
    after_count = len(movies_df)
    
    if before_count > after_count:
        logger.warning(f"Dropped {before_count - after_count} rows with failed embeddings")
    
    # Save the updated embeddings to cache
    if use_cache:
        save_embeddings(movies_df, cache_file)
    
    return movies_df


def generate_query_embedding(query_text, api_key, model="text-embedding-ada-002"):
    """Generate embedding for a query string using the latest OpenAI API"""
    from openai import OpenAI
    
    if not api_key:
        raise ValueError("OpenAI API key is required for generating embeddings")
    
    logger.info(f"Generating embedding for query: {query_text}")
    
    try:
        # Create a custom httpx client without proxies
        http_client = httpx.Client(
            timeout=60.0,
            follow_redirects=True
        )
        
        # Initialize OpenAI client with the custom httpx client
        client = OpenAI(
            api_key=api_key,
            http_client=http_client
        )
        
        response = client.embeddings.create(
            input=query_text,
            model=model
        )
        
        embedding = response.data[0].embedding
        logger.info("Successfully generated query embedding")
        return embedding
            
    except Exception as e:
        logger.error(f"Error generating query embedding: {str(e)}")
        return None
