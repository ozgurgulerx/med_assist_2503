"""
Main entry point for the Medical Assistant Bot application
"""

import asyncio
from dotenv import load_dotenv
from medical_assistant_bot import MedicalAssistantBot, interactive_conversation

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    # Run the interactive conversation
    asyncio.run(interactive_conversation())