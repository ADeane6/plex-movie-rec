import pandas as pd

def get_movie_recommendations(query, movies_df, collection, openai_api_key, n=5):
    """Get movie recommendations based on a query"""
    from src.embedding import generate_query_embedding
    from src.vector_db import query_vector_db
    
    # Generate embedding for the query
    query_embedding = generate_query_embedding(query, openai_api_key)
    
    # Query the vector database
    results = query_vector_db(collection, query_embedding, n)
    
    if not results or 'ids' not in results or not results['ids']:
        return []
    
    # Get the recommended movies
    recommended_indices = [int(idx) for idx in results['ids'][0]]
    recommended_movies = movies_df.iloc[recommended_indices].copy()
    
    # Format the recommendations
    formatted_recommendations = []
    for _, movie in recommended_movies.iterrows():
        formatted_recommendations.append({
            'title': movie['title'],
            'year': movie['year'],
            'genres': ', '.join(movie['genres']),
            'key': movie['key'],
            'summary': movie['summary']
        })
    
    return formatted_recommendations

def extract_movie_to_play(user_input, recommendations):
    """Extract which movie the user wants to play from their input"""
    user_input_lower = user_input.lower()
    
    # Check if the user wants to play a movie by number
    if "play" in user_input_lower and any(str(i+1) in user_input_lower for i in range(len(recommendations))):
        for i in range(len(recommendations)):
            if f"{i+1}" in user_input_lower:
                return recommendations[i]
    
    # Check if the user mentions a movie title
    for movie in recommendations:
        if movie['title'].lower() in user_input_lower:
            return movie
    
    # If we can't determine which movie, return the first one
    if "play" in user_input_lower and recommendations:
        return recommendations[0]
    
    return None

def find_similar_by_director(movies_df, director, exclude_title=None, limit=3):
    """Find movies by the same director"""
    if not director:
        return []
    
    similar_movies = []
    for _, movie in movies_df.iterrows():
        if director in movie['directors'] and movie['title'] != exclude_title:
            similar_movies.append({
                'title': movie['title'],
                'year': movie['year'],
                'genres': ', '.join(movie['genres']),
                'key': movie['key']
            })
            if len(similar_movies) >= limit:
                break
    
    return similar_movies

def find_similar_by_genre(movies_df, genres, exclude_title=None, limit=3):
    """Find movies with similar genres"""
    if not genres:
        return []
    
    # Calculate genre similarity scores
    similarity_scores = []
    for _, movie in movies_df.iterrows():
        if movie['title'] == exclude_title:
            continue
        
        # Count matching genres
        matching_genres = set(movie['genres']).intersection(set(genres))
        score = len(matching_genres) / max(len(movie['genres']), len(genres))
        
        if score > 0:
            similarity_scores.append({
                'title': movie['title'],
                'year': movie['year'],
                'genres': ', '.join(movie['genres']),
                'key': movie['key'],
                'score': score
            })
    
    # Sort by similarity score
    similarity_scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Return top matches
    return similarity_scores[:limit]

def get_popular_movies(movies_df, limit=5):
    """Get a list of popular movies (placeholder - in a real implementation, 
    this could use ratings or view counts)"""
    # In a real implementation, you might sort by rating or view count
    # Here we'll just return some random movies
    import random
    
    sample_indices = random.sample(range(len(movies_df)), min(limit, len(movies_df)))
    popular_movies = []
    
    for idx in sample_indices:
        movie = movies_df.iloc[idx]
        popular_movies.append({
            'title': movie['title'],
            'year': movie['year'],
            'genres': ', '.join(movie['genres']),
            'key': movie['key']
        })
    
    return popular_movies

def get_recently_added_movies(movies_df, limit=5):
    """Get recently added movies (placeholder - in a real implementation, 
    this would use the 'addedAt' field)"""
    # In a real implementation, you would sort by addedAt
    # Here we'll just return the last rows in the dataframe
    recent_movies = []
    
    for _, movie in movies_df.tail(limit).iterrows():
        recent_movies.append({
            'title': movie['title'],
            'year': movie['year'],
            'genres': ', '.join(movie['genres']),
            'key': movie['key']
        })
    
    return recent_movies