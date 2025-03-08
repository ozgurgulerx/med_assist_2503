#!/usr/bin/env python3
import asyncio
import os
from dotenv import load_dotenv

from config.logging_config import setup_logging
from core.bot import MedicalAssistantBot

async def interactive_conversation():
    """Run an interactive conversation with the medical assistant bot"""
    # Set up logging
    setup_logging()
    
    # Check for environment variables
    if not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"):
        print("\nWARNING: Azure OpenAI environment variables not set.")
        print("Using fallback responses instead of actual AI service.")
        print("\nTo use Azure OpenAI, please set:")
        print("  export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'")
        print("  export AZURE_OPENAI_API_KEY='your-api-key'")
        print("  export AZURE_OPENAI_DEPLOYMENT_NAME='gpt-4o'")
    
    # Initialize the bot
    bot = MedicalAssistantBot()
    user_id = "interactive_user"
    
    print("\n----- Starting Interactive Medical Assistant Conversation -----")
    print("Type your messages and press Enter. Type 'exit', 'quit', or 'bye' to end the conversation.\n")
    
    # Initial greeting
    print("Bot: Hello! I'm your medical assistant. How can I help you today?")
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for exit commands
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nBot: Thank you for talking with me. Take care!")
            break
        
        try:
            # Process the message
            response = await bot.process_message(user_id, user_input)
            print(f"\nBot: {response}")
        except Exception as e:
            print(f"\nError processing message: {str(e)}")
            # Print more detailed error information
            import traceback
            print(traceback.format_exc())
            print("\nBot: I'm sorry, I encountered an error. Please try again.")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Run the interactive conversation
    asyncio.run(interactive_conversation())