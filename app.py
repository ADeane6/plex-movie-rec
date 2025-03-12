from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import json
import logging
import traceback
import sys
import uuid
from datetime import datetime, timedelta
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import project modules
from src.plex_connector import connect_to_plex, extract_plex_movies, get_available_clients, play_movie_by_key
from src.embedding import generate_embeddings
from src.vector_db import setup_vector_db
from src.llm_service import LLMService
from src.recommendation import get_movie_recommendations, extract_movie_to_play
import config

app = Flask(__name__)

# Global variables to store our connections and data
plex = None
movies_df = None
collection = None
llm_service = None
sessions = {}

# Function to clean up old sessions
def cleanup_old_sessions():
    """Remove sessions older than 30 minutes"""
    current_time = datetime.now()
    expired_sessions = [
        session_id for session_id, session_data in sessions.items()
        if current_time - session_data.get('last_updated', current_time) > timedelta(minutes=30)
    ]
    for session_id in expired_sessions:
        del sessions[session_id]

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize the recommendation system"""
    global plex, movies_df, collection, llm_service
    
    try:
        logger.info("Starting initialization process")
        
        # Connect to Plex
        logger.info("Connecting to Plex server")
        if config.PLEX_URL and config.PLEX_TOKEN:
            logger.info(f"Using direct connection to Plex server at {config.PLEX_URL}")
            plex = connect_to_plex(baseurl=config.PLEX_URL, token=config.PLEX_TOKEN)
        elif config.PLEX_USERNAME and config.PLEX_PASSWORD and config.PLEX_SERVERNAME:
            logger.info(f"Connecting to Plex server {config.PLEX_SERVERNAME} via MyPlex account")
            plex = connect_to_plex(
                username=config.PLEX_USERNAME,
                password=config.PLEX_PASSWORD,
                servername=config.PLEX_SERVERNAME
            )
        else:
            logger.error("No valid Plex credentials provided")
            return jsonify({"error": "No valid Plex credentials provided"}), 400
        
        # Extract movie data
        logger.info(f"Extracting movie data from library: {config.MOVIE_LIBRARY_NAME}")
        movies_df = extract_plex_movies(plex, config.MOVIE_LIBRARY_NAME)
        logger.info(f"Extracted {len(movies_df)} movies from Plex library")
        
        # Generate embeddings with caching
        cache_file = os.path.join(config.VECTOR_DB_PATH, "cached_embeddings.pkl")
        logger.info("Generating embeddings for movies (with caching)")
        movies_df = generate_embeddings(
            movies_df, 
            config.OPENAI_API_KEY,
            batch_size=config.BATCH_SIZE,
            cache_file=cache_file, 
            use_cache=True
        )
        logger.info(f"Generated embeddings for {len(movies_df)} movies")
        
        # Set up vector database
        logger.info(f"Setting up vector database at {config.VECTOR_DB_PATH}")
        collection = setup_vector_db(movies_df, config.VECTOR_DB_PATH)
        logger.info("Vector database setup complete")
        
        # Initialize LLM service
        logger.info(f"Initializing LLM service with provider: {config.LLM_PROVIDER}")
        llm_service = LLMService(
            provider=config.LLM_PROVIDER,
            anthropic_api_key=config.ANTHROPIC_API_KEY,
            openai_api_key=config.OPENAI_API_KEY,
            anthropic_model=config.ANTHROPIC_MODEL,
            openai_model=config.OPENAI_MODEL
        )
        logger.info("LLM service initialized")
        
        logger.info("Initialization complete")
        return jsonify({
            "success": True,
            "message": f"Successfully initialized with {len(movies_df)} movies"
        })
        
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Get movie recommendations based on user input"""
    global plex, movies_df, collection, llm_service, sessions
    
    if not all([plex, movies_df is not None, collection, llm_service]):
        logger.error("System not initialized")
        return jsonify({"error": "System not initialized"}), 400
    
    try:
        data = request.json
        user_input = data.get('message', '')
        session_id = data.get('session_id')
        
        # Create a new session if none exists
        if not session_id or session_id not in sessions:
            session_id = str(uuid.uuid4())
            sessions[session_id] = {
                'last_updated': datetime.now(),
                'recent_recommendations': [],
                'conversation_history': []
            }
        
        # Update session timestamp
        sessions[session_id]['last_updated'] = datetime.now()
        
        # Add user message to conversation history
        sessions[session_id]['conversation_history'].append({
            'role': 'user',
            'content': user_input
        })
        
        logger.info(f"Received recommendation request: {user_input}")
        
        # Check if this is a follow-up command about previous recommendations
        recent_recommendations = sessions[session_id].get('recent_recommendations', [])
        is_play_command = False
        movie_to_play = None
        
        # Check for play commands by index number
        if recent_recommendations and ('play' in user_input.lower() or 'watch' in user_input.lower()):
            is_play_command = True
            
            # Check for number references (e.g., "play the second one" or "play #2")
            number_words = {
                'first': 0, 'second': 1, 'third': 2, 'fourth': 3, 'fifth': 4,
                'sixth': 5, 'seventh': 6, 'eighth': 7, 'ninth': 8, 'tenth': 9,
                '1': 0, '2': 1, '3': 2, '4': 3, '5': 4,
                '6': 5, '7': 6, '8': 7, '9': 8, '10': 9,
                '#1': 0, '#2': 1, '#3': 2, '#4': 3, '#5': 4,
                '#6': 5, '#7': 6, '#8': 7, '#9': 8, '#10': 9,
                'one': 0, 'two': 1, 'three': 2, 'four': 3, 'five': 4,
                'six': 5, 'seven': 6, 'eight': 7, 'nine': 8, 'ten': 9
            }
            
            for word, index in number_words.items():
                if word in user_input.lower() and index < len(recent_recommendations):
                    movie_to_play = recent_recommendations[index]
                    logger.info(f"User wants to play recommendation #{index+1}: {movie_to_play['title']}")
                    break
            
            # Check for movie title mentions
            if not movie_to_play:
                for movie in recent_recommendations:
                    if movie['title'].lower() in user_input.lower():
                        movie_to_play = movie
                        logger.info(f"User wants to play: {movie_to_play['title']}")
                        break
        
        # If this is a play command for a previous recommendation
        if is_play_command and movie_to_play:
            clients = get_available_clients(plex)
            if clients:
                client_name = clients[0].title  # Default to first client
                logger.info(f"Playing on client: {client_name}")
                play_result = play_movie_by_key(plex, movie_to_play['key'], client_name)
                
                response_text = f"Now playing '{movie_to_play['title']}' on {client_name}."
                
                # Add assistant message to conversation history
                sessions[session_id]['conversation_history'].append({
                    'role': 'assistant',
                    'content': response_text
                })
                
                return jsonify({
                    "response": response_text,
                    "recommendations": recent_recommendations,
                    "session_id": session_id
                })
        
        # If not a play command, get new recommendations
        logger.info("Interpreting user request")
        interpreted_query = llm_service.interpret_user_request(user_input)
        logger.info(f"Interpreted query: {interpreted_query}")
        
        # Get recommendations
        logger.info("Getting movie recommendations")
        recommendations = get_movie_recommendations(
            interpreted_query, 
            movies_df, 
            collection, 
            config.OPENAI_API_KEY
        )
        logger.info(f"Found {len(recommendations)} recommendations")
        
        # Store the new recommendations in the session
        sessions[session_id]['recent_recommendations'] = recommendations
        
        # Generate response
        logger.info("Generating response with LLM")
        response_text = llm_service.generate_recommendation_response(user_input, recommendations)
        
        # Add assistant message to conversation history
        sessions[session_id]['conversation_history'].append({
            'role': 'assistant',
            'content': response_text
        })
        
        # Clean up old sessions periodically
        if random.random() < 0.1:  # 10% chance to clean up on each request
            cleanup_old_sessions()
        
        return jsonify({
            "response": response_text,
            "recommendations": recommendations,
            "session_id": session_id
        })
        
    except Exception as e:
        logger.error(f"Error during recommendation: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/api/clients', methods=['GET'])
def clients():
    """Get available Plex clients"""
    global plex
    
    if not plex:
        logger.error("System not initialized")
        return jsonify({"error": "System not initialized"}), 400
    
    try:
        logger.info("Getting available Plex clients")
        clients = get_available_clients(plex)
        client_list = [{"name": client.title, "product": client.product} for client in clients]
        logger.info(f"Found {len(client_list)} clients")
        
        return jsonify({
            "clients": client_list
        })
        
    except Exception as e:
        logger.error(f"Error getting clients: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/api/play', methods=['POST'])
def play():
    """Play a movie on a client"""
    global plex
    
    if not plex:
        logger.error("System not initialized")
        return jsonify({"error": "System not initialized"}), 400
    
    try:
        data = request.json
        movie_key = data.get('movieKey')
        client_name = data.get('clientName')
        logger.info(f"Request to play movie with key {movie_key} on client {client_name}")
        
        if not movie_key or not client_name:
            logger.error("Movie key and client name are required")
            return jsonify({"error": "Movie key and client name are required"}), 400
        
        result = play_movie_by_key(plex, movie_key, client_name)
        logger.info(f"Play result: {result}")
        
        return jsonify({
            "result": result
        })
        
    except Exception as e:
        logger.error(f"Error playing movie: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

if __name__ == '__main__':
    # Ensure the static and templates directories exist
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    app.run(debug=True)
