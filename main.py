"""
Medical Assistant Bot API & Interactive CLI
"""
import os
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from medical_assistant_bot import process_message

# 🔹 Setup detailed logging
#log_file = "/home/LogFiles/myapp.log"  # Save logs for debugging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Log only to stdout
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# 🔹 Allow CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Temporarily allow all origins for debugging
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Define input data model
class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    """Health check endpoint"""
    logger.info("Health check endpoint accessed.")
    return {"message": "Medical Assistant API is running"}

@app.post("/chat")
async def chat(request: ChatRequest):
    """Receives user message and returns chatbot response"""
    logger.info(f"Received message: {request.message}")
    
    # 🔹 Debugging request body
    try:
        logger.debug(f"Raw Request Data: {request.dict()}")
    except Exception as e:
        logger.error(f"Failed to parse request data: {str(e)}")
    
    try:
        logger.info("Calling process_message() function...")
        response = await process_message(request.message)
        logger.info(f"Generated response: {response}")
        return {"response": response}
    except Exception as e:
        logger.error(f"Chatbot processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting Medical Assistant API Server")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
    logger.info("Medical Assistant API Server started")