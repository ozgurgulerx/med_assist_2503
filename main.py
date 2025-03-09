"""
Main entry point for the Medical Assistant bot - API server and CLI
"""
import os
import asyncio
import logging
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import from core bot
from core_bot import process_message_api, interactive_conversation

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
    include_diagnostics: bool = True  # New option to toggle diagnostic info

@app.get("/")
async def root():
    """Health check endpoint"""
    logger.info("Health check endpoint accessed.")
    return {
        "message": "Medical Assistant API is running",
        "models": {
            "primary": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "o3"),
            "mini": os.getenv("AZURE_OPENAI_MINI_DEPLOYMENT_NAME", "o3-mini")
        }
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    """Process a message and return the bot's response with optional diagnostic information"""
    logger.info(f"Received message: {request.message}")
    
    try:
        logger.info(f"Processing message for user: {request.user_id or 'anonymous'}")
        response = await process_message_api(
            request.message,
            request.user_id,
            include_diagnostics=request.include_diagnostics
        )
        logger.info(f"Generated response: {response[:100]}...")
        return {"response": response}
    except Exception as e:
        logger.error(f"Chatbot processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")

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