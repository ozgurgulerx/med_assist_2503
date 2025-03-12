"""
Main entry point for the Medical Assistant bot - API server and CLI
With enhanced logging for troubleshooting -working---
"""
import os
import asyncio
import logging
import traceback
from typing import Optional, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import custom logging setup
try:
    from troubleshooting import setup_enhanced_logging, log_system_info
    # Set up enhanced logging if available
    log_file = setup_enhanced_logging(log_level="DEBUG" if os.getenv("DEBUG_MODE") == "true" else "INFO")
    log_system_info()
    print(f"Enhanced logging enabled. Log file: {log_file}")
except ImportError:
    # Fall back to basic logging if troubleshooting.py is not available
    logging.basicConfig(
        level=logging.DEBUG if os.getenv("DEBUG_MODE") == "true" else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    print("Basic logging configured. For enhanced logging, add troubleshooting.py")

# Import from core bot
from core_bot import MedicalAssistantBot, interactive_conversation

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical Assistant API",
    description="API for the Medical Assistant Bot with self-reflection based diagnosis confidence",
    version="1.0.0"
)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input data model
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    include_diagnostics: bool = False  # Option to toggle diagnostic info

# Global bot instances cache
bot_instances: Dict[str, MedicalAssistantBot] = {}

# Function to get or create a bot instance for a user
def get_bot_instance(user_id: str) -> MedicalAssistantBot:
    """Get an existing bot instance or create a new one for the user"""
    if user_id not in bot_instances:
        logger.info(f"Creating new bot instance for user: {user_id}")
        bot_instances[user_id] = MedicalAssistantBot()
    return bot_instances[user_id]

@app.get("/")
async def root():
    """Health check endpoint"""
    logger.info("Health check endpoint accessed.")
    return {
        "message": "Medical Assistant API is running",
        "models": {
            "primary": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            "mini": os.getenv("AZURE_OPENAI_MINI_DEPLOYMENT_NAME", "gpt-4o-mini")
        },
        "active_users": len(bot_instances)
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    """Process a message and return the bot's response with optional diagnostic information"""
    # Ensure we have a user ID
    user_id = request.user_id
    if not user_id:
        user_id = f"anon_{hash(request.message) % 10000}"
    
    logger.info(f"Received message from user {user_id}: {request.message}")
    
    try:
        # Get the bot instance for this user
        bot = get_bot_instance(user_id)
        
        # Always include diagnostics for demonstration purposes unless explicitly disabled
        include_diagnostics = True if request.include_diagnostics is None else request.include_diagnostics
        
        # Process the message using the persistent bot instance
        response = await bot.process_message(
            user_id,
            request.message,
            include_diagnostics=include_diagnostics
        )
        
        # Split the response into user-facing content and debug information
        user_response = response
        debug_info = ""
        
        if include_diagnostics and "MEDICAL ASSESSMENT REPORT" in response:
            parts = response.split("==========================================\n", 1)
            if len(parts) > 1:
                user_response = parts[0].strip()
                debug_info = "==========================================\n" + parts[1]
            else:
                debug_info = "Debug information formatting issue"
        
        logger.info(f"Generated response for user {user_id}: {user_response[:100]}...")
        return {
            "response": user_response,
            "debug_info": debug_info,
            "user_id": user_id
        }
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Chatbot processing error for user {user_id}: {str(e)}\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")

@app.get("/users")
async def list_users():
    """List active users with bot instances"""
    return {
        "active_users": len(bot_instances),
        "user_ids": list(bot_instances.keys())
    }

@app.delete("/users/{user_id}")
async def reset_user(user_id: str):
    """Reset a user's conversation and clear all associated state"""
    if user_id in bot_instances:
        # Get the bot instance to reset its internal state before deleting
        if hasattr(bot_instances[user_id], 'dialog_manager'):
            bot_instances[user_id].dialog_manager.reset_user_state(user_id)
            
        # Also clear any state in the intent classifier
        if hasattr(bot_instances[user_id], 'intent_classifier'):
            if hasattr(bot_instances[user_id].intent_classifier, 'dialogue_tracker'):
                if user_id in bot_instances[user_id].intent_classifier.dialogue_tracker.user_states:
                    del bot_instances[user_id].intent_classifier.dialogue_tracker.user_states[user_id]
            if hasattr(bot_instances[user_id].intent_classifier, 'conversation_histories'):
                if user_id in bot_instances[user_id].intent_classifier.conversation_histories:
                    del bot_instances[user_id].intent_classifier.conversation_histories[user_id]
        
        # Finally delete the bot instance
        del bot_instances[user_id]
        logger.info(f"Reset conversation and all state for user: {user_id}")
        return {"status": "success", "message": f"User {user_id} conversation and state completely reset"}
    return {"status": "not_found", "message": "User not found"}

@app.post("/generate_user_id")
async def generate_user_id():
    """Generate a unique user ID for a new session"""
    # Generate a unique ID using a timestamp and random string
    import time
    import random
    import string
    
    timestamp = int(time.time())
    random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    user_id = f"user_{timestamp}_{random_string}"
    
    logger.info(f"Generated new user ID: {user_id}")
    return {"user_id": user_id}

def run_api():
    """Run the API server"""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"Starting Medical Assistant API Server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="debug" if os.getenv("DEBUG_MODE") == "true" else "info")

async def run_interactive_with_recovery():
    """Run the interactive mode with error recovery"""
    print("\nMedical Assistant Bot Interactive Mode")
    print("Type 'exit' to quit, 'debug on/off' to toggle diagnostic info, 'reset' to start over\n")
    print("\033[33mDebug mode is enabled by default for demonstration purposes\033[0m")
    
    bot = MedicalAssistantBot()
    user_id = "interactive_user"
    include_diagnostics = True  # Set to True by default for demonstration
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ")
            
            # Check for special commands
            if user_input.lower() == 'exit':
                print("\nGoodbye!")
                break
            elif user_input.lower() == 'reset':
                print("\nResetting conversation...")
                # Create a new bot instance
                bot = MedicalAssistantBot()
                continue
            elif user_input.lower() == 'debug on':
                include_diagnostics = True
                print("\nDebug mode enabled. Diagnostic information will be shown.")
                continue
            elif user_input.lower() == 'debug off':
                include_diagnostics = False
                print("\nDebug mode disabled. Diagnostic information will be hidden.")
                continue
            
            # Process the message
            response = await bot.process_message(user_id, user_input, include_diagnostics=include_diagnostics)
            
            # Display the response
            if include_diagnostics and "MEDICAL ASSESSMENT REPORT" in response:
                # Split the response into user-facing content and debug information
                parts = response.split("==========================================\n", 1)
                if len(parts) > 1:
                    user_response = parts[0].strip()
                    debug_info = "==========================================\n" + parts[1]
                    
                    # Print the user-facing response
                    print(f"\nAssistant: {user_response}")
                    
                    # Print the debug information in a different color
                    print("\n\033[33m--- DEBUG INFORMATION ---\033[0m")
                    print(f"\033[33m{debug_info}\033[0m")
                else:
                    print(f"\nAssistant: {response}")
            else:
                print(f"\nAssistant: {response}")
                
        except KeyboardInterrupt:
            print("\n\nExiting due to keyboard interrupt...")
            break
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error in interactive mode: {str(e)}\n{error_details}")
            print(f"\nAn error occurred: {str(e)}")
            print("The bot will attempt to continue with the conversation.")

if __name__ == "__main__":
    # Log startup information
    logger.info("Medical Assistant starting up")
    logger.info(f"Run mode: {os.getenv('RUN_MODE', 'interactive')}")
    
    # Check if we should run in API or interactive mode
    if os.getenv("RUN_MODE", "interactive").lower() == "api":
        logger.info("Starting in API mode")
        run_api()
    else:
        logger.info("Starting in interactive CLI mode")
        # Check for environment variables
        if not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"):
            print("\n⚠️ WARNING: Azure OpenAI environment variables not set.")
            print("Using fallback responses instead of actual AI service.")
            print("\nTo use Azure OpenAI, please set in your .env file:")
            print("  AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'")
            print("  AZURE_OPENAI_API_KEY='your-api-key'")
            print("  AZURE_OPENAI_DEPLOYMENT_NAME='gpt-4o'")
            print("  AZURE_OPENAI_MINI_DEPLOYMENT_NAME='gpt-4o-mini'")
            print("\nContinuing with fallback mode...\n")
            
        # Run with error recovery
        asyncio.run(run_interactive_with_recovery())