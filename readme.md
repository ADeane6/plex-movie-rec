# Plex LLM Movie Recommender

A smart movie recommendation system that integrates with your Plex Media Server, using AI to provide personalized movie suggestions and seamless playback.

## Features

- **AI-Powered Recommendations**: Get intelligent movie suggestions based on natural language requests
- **Semantic Understanding**: Ask for movies similar to ones you love or describe the type of movie you're in the mood for
- **Plex Integration**: Seamlessly connects to your Plex Media Server and accesses your movie library
- **One-Click Playback**: Play recommended movies directly on your Plex clients
- **Conversation Memory**: Maintains context between interactions for natural follow-up questions
- **Vector Search**: Uses advanced embedding technology to find semantically similar movies
- **Efficient Caching**: Saves embeddings to avoid regenerating them on restart

## Requirements

- Python 3.8 or higher
- Plex Media Server with a movie library
- OpenAI API key for embeddings
- Anthropic API key (for Claude) or OpenAI API key for the recommendation LLM

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/plex-llm-recommender.git
   cd plex-llm-recommender
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file based on the `.env.example` template:

   ```bash
   cp .env.example .env
   ```

5. Edit the `.env` file with your credentials:

   ```
   # Plex Configuration
   PLEX_URL=http://your-plex-server:32400
   PLEX_TOKEN=your-plex-token
   # Or use username/password authentication
   # PLEX_USERNAME=your-username
   # PLEX_PASSWORD=your-password
   # PLEX_SERVERNAME=your-server-name

   # API Keys
   OPENAI_API_KEY=your-openai-api-key
   ANTHROPIC_API_KEY=your-anthropic-api-key

   # LLM Configuration
   LLM_PROVIDER=anthropic  # or openai
   ANTHROPIC_MODEL=claude-3-sonnet-20240229
   OPENAI_MODEL=gpt-4

   # Vector DB Configuration
   VECTOR_DB_PATH=./chroma_db
   MOVIE_LIBRARY_NAME=Movies
   ```

## Usage

1. Start the application:

   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:

   ```
   http://127.0.0.1:5000
   ```

3. Click the "Initialize System" button to connect to your Plex server and prepare the recommendation engine.

4. Once initialization is complete, you can start asking for movie recommendations:

   - "I'm in the mood for a sci-fi movie like Inception"
   - "Recommend me a comedy to watch tonight"
   - "What are some action movies in my library?"

5. To play a recommended movie, you can say:
   - "Play the first one"
   - "I'd like to watch the second recommendation"
   - "Play Interstellar on my Living Room TV"

## How It Works

1. **Plex Connection**: The app connects to your Plex Media Server and extracts metadata about your movie library.

2. **Embedding Generation**: Each movie's metadata (title, director, actors, genres, summary) is converted into a vector embedding using OpenAI's embedding model.

3. **Vector Database**: These embeddings are stored in a vector database (ChromaDB) for efficient similarity search.

4. **Natural Language Understanding**: When you ask for recommendations, an LLM (Claude or GPT-4) interprets your request to understand what kind of movies you're looking for.

5. **Semantic Search**: Your request is converted to an embedding and used to find the most similar movies in your library.

6. **Response Generation**: The LLM creates a natural, conversational response presenting the recommendations.

7. **Playback Control**: If you ask to play a movie, the app uses the Plex API to start playback on your chosen client.

## Advanced Configuration

### Customizing the LLM

You can choose between Anthropic's Claude and OpenAI's models by changing the `LLM_PROVIDER` setting in your `.env` file. You can also specify which model version to use.

### Embedding Caching

By default, the app caches movie embeddings to avoid regenerating them on restart. The cache is stored in the directory specified by `VECTOR_DB_PATH`. To force regeneration of embeddings, delete the `cached_embeddings.pkl` file in this directory.

### Multiple Libraries

The app is configured to work with a single movie library. If you have multiple movie libraries, you can specify which one to use with the `MOVIE_LIBRARY_NAME` setting.

## Troubleshooting

### Connection Issues

- Verify your Plex server is running and accessible
- Check that your Plex token or credentials are correct
- Ensure your server name matches exactly what's in your Plex account

### API Key Issues

- Verify your OpenAI and Anthropic API keys are valid and have sufficient credits
- Check for any environment variables that might be overriding your settings

### Embedding Generation Errors

- If you encounter errors during embedding generation, try reducing the batch size
- Check your OpenAI API key has permissions for the embedding model

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [python-plexapi](https://github.com/pkkid/python-plexapi) for the Plex API integration
- [OpenAI](https://openai.com/) for the embedding model
- [Anthropic](https://www.anthropic.com/) for Claude
- [ChromaDB](https://www.trychroma.com/) for the vector database

---
