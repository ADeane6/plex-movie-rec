from anthropic import Anthropic
from openai import OpenAI
import httpx
import logging

logger = logging.getLogger(__name__)

class LLMService:
    """Service for interacting with LLMs (Claude or OpenAI)"""
    
    def __init__(self, provider="anthropic", anthropic_api_key=None, openai_api_key=None,
                 anthropic_model="claude-3-sonnet-20240229", openai_model="gpt-4"):
        self.provider = provider
        self.anthropic_api_key = anthropic_api_key
        self.openai_api_key = openai_api_key
        self.anthropic_model = anthropic_model
        self.openai_model = openai_model
        
        # Create a custom httpx client without proxies
        http_client = httpx.Client(
            timeout=60.0,
            follow_redirects=True
        )
        
        # Initialize appropriate client
        if self.provider == "anthropic" and self.anthropic_api_key:
            logger.info(f"Initializing Anthropic client with model: {self.anthropic_model}")
            self.anthropic_client = Anthropic(
                api_key=self.anthropic_api_key,
                http_client=http_client
            )
        elif self.provider == "openai" and self.openai_api_key:
            logger.info(f"Initializing OpenAI client with model: {self.openai_model}")
            self.openai_client = OpenAI(
                api_key=self.openai_api_key,
                http_client=http_client
            )
    
    def interpret_user_request(self, user_input, conversation_history=None):
        """Interpret the user's movie request using an LLM"""
        system_prompt = """
        You are a movie recommendation assistant for a Plex media server.
        The user has a library of movies and wants recommendations.
        
        If the user is asking for movie recommendations, extract what kind of movie they're looking for.
        Focus on extracting genres, themes, moods, or similar movies mentioned.
        
        If the user is referring to previous recommendations (e.g., "play the second one" or "tell me more about the third movie"),
        identify this as a follow-up command, not a new recommendation request.
        
        Return a concise description that captures the essence of what they're looking for,
        or clearly indicate if this is a follow-up command about previous recommendations.
        """
        
        logger.info(f"Interpreting user request: {user_input}")
        
        # Format conversation history if provided
        messages = []
        if conversation_history:
            for message in conversation_history:
                messages.append({
                    "role": message["role"],
                    "content": message["content"]
                })
        
        # Add the current user message
        messages.append({"role": "user", "content": user_input})
        
        if self.provider == "anthropic":
            try:
                response = self.anthropic_client.messages.create(
                    model=self.anthropic_model,
                    max_tokens=300,
                    system=system_prompt,
                    messages=messages if messages else [{"role": "user", "content": user_input}]
                )
                interpreted_query = response.content[0].text
                logger.info(f"Interpreted query: {interpreted_query}")
                return interpreted_query
            except Exception as e:
                logger.error(f"Error interpreting request with Anthropic: {str(e)}")
                return user_input
            
        elif self.provider == "openai":
            try:
                # Prepare messages for OpenAI format
                openai_messages = [{"role": "system", "content": system_prompt}]
                if messages:
                    openai_messages.extend(messages)
                else:
                    openai_messages.append({"role": "user", "content": user_input})
                
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=openai_messages
                )
                interpreted_query = response.choices[0].message.content
                logger.info(f"Interpreted query: {interpreted_query}")
                return interpreted_query
            except Exception as e:
                logger.error(f"Error interpreting request with OpenAI: {str(e)}")
                return user_input
        
        return user_input
    
    def generate_recommendation_response(self, user_input, recommendations):
        """Generate a natural language response with movie recommendations"""
        # Format the recommendations
        recommendation_text = "\n".join([
            f"{i+1}. {movie['title']} ({movie['year']}) - {movie['genres']}"
            for i, movie in enumerate(recommendations)
        ])
        
        prompt = f"""
        The user asked: "{user_input}"
        
        Based on their request, here are some movie recommendations from their Plex library:
        
        {recommendation_text}
        
        Create a friendly, conversational response that presents these recommendations.
        Explain briefly why each movie might match what they're looking for.
        If they mentioned a specific movie, you can reference how these recommendations relate to it.
        """
        
        logger.info("Generating recommendation response")
        
        if self.provider == "anthropic":
            try:
                response = self.anthropic_client.messages.create(
                    model=self.anthropic_model,
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                generated_response = response.content[0].text
                logger.info("Successfully generated recommendation response")
                return generated_response
            except Exception as e:
                logger.error(f"Error generating response with Anthropic: {str(e)}")
                return f"Here are some movie recommendations for you:\n\n{recommendation_text}"
            
        elif self.provider == "openai":
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                generated_response = response.choices[0].message.content
                logger.info("Successfully generated recommendation response")
                return generated_response
            except Exception as e:
                logger.error(f"Error generating response with OpenAI: {str(e)}")
                return f"Here are some movie recommendations for you:\n\n{recommendation_text}"
        
        return f"Here are some movie recommendations for you:\n\n{recommendation_text}"
