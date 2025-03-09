"""
Medical Assistant Bot API & Interactive CLI
"""
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from medical_assistant_bot import interactive_conversation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# 🔹 Allow CORS for your frontend domain
FRONTEND_URL = "https://victorious-coast-00667c603.6.azurestaticapps.net"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],  # Restrict CORS to your UI domain only
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# 🔹 Define input data model
class ChatRequest(BaseModel):
    message: str  # The message sent by the user

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Medical Assistant API is running"}

@app.post("/chat")
async def chat(request: ChatRequest):
    """Receives user message and returns chatbot response"""
    try:
        logger.info(f"Received message: {request.message}")
        response = await interactive_conversation(request.message)  # Pass user input
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in chatbot response: {str(e)}")
        raise HTTPException(status_code=500, detail="Chatbot error")

if __name__ == "__main__":
    logger.info("Starting Medical Assistant API Server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    logger.info("Medical Assistant API Server started")