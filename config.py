import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Plex Configuration
PLEX_URL = os.getenv('PLEX_URL')
PLEX_TOKEN = os.getenv('PLEX_TOKEN')
PLEX_USERNAME = os.getenv('PLEX_USERNAME')
PLEX_PASSWORD = os.getenv('PLEX_PASSWORD')
PLEX_SERVERNAME = os.getenv('PLEX_SERVERNAME')

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# LLM Configuration
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'anthropic')  # or openai
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')
ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229')

# Vector DB Configuration
VECTOR_DB_PATH = os.getenv('VECTOR_DB_PATH', './chroma_db')

# Movie library section name
MOVIE_LIBRARY_NAME = os.getenv('MOVIE_LIBRARY_NAME', 'Movies')

# Embeddings Configuration
EMBEDDINGS_CACHE_FILE = os.getenv('EMBEDDINGS_CACHE_FILE', os.path.join(VECTOR_DB_PATH, 'cached_embeddings.pkl'))

# Batch size for embeddings
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 100))
