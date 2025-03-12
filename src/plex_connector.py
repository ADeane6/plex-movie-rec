from plexapi.server import PlexServer
from plexapi.myplex import MyPlexAccount
import pandas as pd

def connect_to_plex(baseurl=None, token=None, username=None, password=None, servername=None):
    """Connect to Plex server using either direct connection or via MyPlex account"""
    if baseurl and token:
        return PlexServer(baseurl, token)
    elif username and password and servername:
        account = MyPlexAccount(username, password)
        return account.resource(servername).connect()
    else:
        raise ValueError("Either (baseurl, token) or (username, password, servername) must be provided")

def extract_plex_movies(plex, library_name='Movies'):
    """Extract movie data from Plex library"""
    movies_section = plex.library.section(library_name)
    
    movies_data = []
    for movie in movies_section.all():
        # Extract relevant metadata
        movie_info = {
            'title': movie.title,
            'year': movie.year if hasattr(movie, 'year') else None,
            'summary': movie.summary if hasattr(movie, 'summary') else "",
            'genres': [g.tag for g in movie.genres] if hasattr(movie, 'genres') and movie.genres else [],
            'directors': [d.tag for d in movie.directors] if hasattr(movie, 'directors') and movie.directors else [],
            'actors': [a.tag for a in movie.roles][:5] if hasattr(movie, 'roles') and movie.roles else [],
            'key': movie.key,  # Store the key for later retrieval
            'rating': movie.rating if hasattr(movie, 'rating') else None,
            'duration': movie.duration if hasattr(movie, 'duration') else None,
        }
        
        # Create a rich text representation for embedding
        movie_info['text_representation'] = f"Title: {movie_info['title']}"
        
        if movie_info['year']:
            movie_info['text_representation'] += f" ({movie_info['year']})"
        
        if movie_info['directors']:
            movie_info['text_representation'] += f". Directed by {', '.join(movie_info['directors'])}"
        
        if movie_info['actors']:
            movie_info['text_representation'] += f". Starring {', '.join(movie_info['actors'])}"
        
        if movie_info['genres']:
            movie_info['text_representation'] += f". Genres: {', '.join(movie_info['genres'])}"
        
        if movie_info['summary']:
            movie_info['text_representation'] += f". Summary: {movie_info['summary']}"
        
        movies_data.append(movie_info)
    
    return pd.DataFrame(movies_data)

def get_available_clients(plex):
    """Get a list of available Plex clients"""
    return plex.clients()

def play_movie_by_key(plex, movie_key, client_name):
    """Play a movie on a specified Plex client using the movie's key"""
    try:
        # Fetch the movie using its key
        movie = plex.fetchItem(movie_key)
        
        # Get the client
        client = plex.client(client_name)
        
        # Play the movie
        client.playMedia(movie)
        
        return f"Now playing {movie.title} on {client_name}"
    except Exception as e:
        return f"Error playing movie: {str(e)}"
