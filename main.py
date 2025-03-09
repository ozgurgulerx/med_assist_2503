"""
Main entry point for the Medical Assistant bot - API server and CLI
"""
import os
import asyncio
import logging
from typing import Optional, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import from core bot
from core_bot import MedicalAssistantBot, interactive_conversation

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG_MODE") == "true" else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
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
    include_diagnostics: bool = True  # Option to toggle diagnostic info

# Global bot instance cache
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
            "primary": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "o3"),
            "mini": os.getenv("AZURE_OPENAI_MINI_DEPLOYMENT_NAME", "o3-mini")
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
        
        # Process the message using the persistent bot instance
        response = await bot.process_message(
            user_id,
            request.message,
            include_diagnostics=request.include_diagnostics
        )
        
        logger.info(f"Generated response for user {user_id}: {response[:100]}...")
        return {"response": response, "user_id": user_id}
    except Exception as e:
        logger.error(f"Chatbot processing error for user {user_id}: {str(e)}", exc_info=True)
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
    """Reset a user's conversation"""
    if user_id in bot_instances:
        del bot_instances[user_id]
        logger.info(f"Reset conversation for user: {user_id}")
        return {"status": "success", "message": f"User {user_id} conversation reset"}
    return {"status": "not_found", "message": "User not found"}

def run_api():
    """Run the API server"""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"Starting Medical Assistant API Server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="debug" if os.getenv("DEBUG_MODE") == "true" else "info")

if __name__ == "__main__":
    # Check if we should run in API or interactive mode
    if os.getenv("RUN_MODE", "interactive").lower() == "api":
        run_api()
    else:
        # Run in interactive CLI mode
        asyncio.run(interactive_conversation())