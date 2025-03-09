"""
Core medical assistant bot using Semantic Kernel

This is the main module that coordinates all components of the medical assistant bot.
It handles message processing, orchestrates dialog flow, and integrates the diagnostic
engine with self-reflection based confidence calculation.
"""
import os
import asyncio
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Semantic Kernel imports
from semantic_kernel.contents.chat_history import ChatHistory

# Local imports - our modular components
from dialog_manager import DialogManager
from llm_handler import LLMHandler
from diagnostic_engine import DiagnosticEngine
from medical_plugins import MedicalKnowledgePlugin
from intent_classifier import IntentClassificationService

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class MedicalAssistantBot:
    """Flexible medical assistant that can handle any medical issue"""
    
    def __init__(self):
        """Initialize the medical assistant bot"""
        # Initialize components
        self.llm_handler = LLMHandler()
        self.medical_plugin = MedicalKnowledgePlugin()
        self.dialog_manager = DialogManager()
        self.diagnostic_engine = DiagnosticEngine(self.llm_handler, self.medical_plugin)
        
        # Initialize intent classifier with its own dedicated service
        self.intent_classifier = IntentClassificationService()
        
        # Chat histories by user ID
        self.chat_histories: Dict[str, ChatHistory] = {}
        
        # User data storage - contains patient_data and other user-specific info
        self.user_data: Dict[str, Dict[str, Any]] = {}
    
    def get_chat_history(self, user_id: str) -> ChatHistory:
        """Get or create chat history for a user"""
        if user_id not in self.chat_histories:
            self.chat_histories[user_id] = ChatHistory()
        return self.chat_histories[user_id]
    
    def get_user_data(self, user_id: str) -> Dict[str, Any]:
        """Get or create user data"""
        if user_id not in self.user_data:
            self.user_data[user_id] = {}
        return self.user_data[user_id]
    
    async def execute_action(self, action_name: str, user_id: str, user_message: str = "") -> str:
        """
        Execute a dialog action and return the response
        
        Args:
            action_name: Name of the action to execute
            user_id: User identifier
            user_message: Original user message
            
        Returns:
            Response text
        """
        # Get user data and ensure patient_data exists
        user_data = self.get_user_data(user_id)
        patient_data = self.diagnostic_engine.get_patient_data(user_data)
        
        logger.info(f"Executing action: {action_name}")
        
        if action_name == "utter_greet":
            return "Hello! I'm your medical assistant. I'm here to help with your health questions."
        
        elif action_name == "utter_how_can_i_help":
            return "How can I help you today?"
        
        elif action_name == "action_handle_out_of_scope":
            # Get the original message for context
            original_message = self.dialog_manager.get_original_message(user_id)
            
            if self.llm_handler.is_available():
                try:
                    # Create a prompt to acknowledge the off-topic message
                    prompt = f"""The user has sent a message that appears to be outside the scope of a medical conversation. 
Their message was: "{original_message}"

Provide a polite, brief response that acknowledges their message but gently redirects the conversation to medical topics.
Make sure your response is concise (max 2 sentences) and ends with a question about their health concerns."""

                    # Get response directly from LLM
                    return await self.llm_handler.execute_prompt(prompt)
                except Exception as e:
                    logger.error(f"Error handling out of scope message with LLM: {str(e)}")
            
            # Fallback response if LLM fails
            return f"I understand you're asking about \"{original_message}\", but I'm primarily designed to help with medical questions."
        
        elif action_name == "utter_redirect_to_medical":
            return "Is there something about your health I can help with today?"
        
        elif action_name == "action_ask_followup_question":
            # Use the diagnostic engine to generate a follow-up question
            response = await self.diagnostic_engine.generate_followup_question(patient_data)
            
            # After asking a follow-up, update the diagnosis confidence
            await self.diagnostic_engine.update_diagnosis_confidence(patient_data)
            
            return response
        
        elif action_name == "action_verify_symptoms":
            # Verify symptoms before providing diagnosis
            return await self.diagnostic_engine.verify_symptoms(patient_data)
        
        elif action_name == "action_provide_medical_info":
            # Extract the topic from user message
            topic = user_message
            
            if self.llm_handler.is_available():
                try:
                    # Create a prompt for medical information
                    prompt = f"""Provide general medical information about the following topic:
Topic: {topic}
Patient demographics: {patient_data.get("demographics", {})}

Give helpful, accurate information while emphasizing this is general advice and not a substitute for professional medical care."""

                    # Use full model for medical information
                    return await self.llm_handler.execute_prompt(prompt, use_full_model=True)
                except Exception as e:
                    logger.error(f"Error providing medical information with LLM: {str(e)}")
            
            # Use the plugin method or fallback
            if self.medical_plugin:
                try:
                    response = await self.medical_plugin.provide_medical_information(
                        topic=topic,
                        patient_demographics=str(patient_data.get("demographics", {}))
                    )
                    return str(response)
                except Exception as e:
                    logger.error(f"Error providing medical information: {str(e)}")
            
            # Last resort fallback
            return f"I can provide general information about {topic}, but remember to consult with a healthcare professional for personalized advice."
        
        elif action_name == "action_provide_diagnosis":
            # Use the diagnostic engine to generate a diagnosis
            return await self.diagnostic_engine.generate_diagnosis(patient_data)
        
        elif action_name == "utter_suggest_mitigations":
            # Use the diagnostic engine to suggest mitigations
            return await self.diagnostic_engine.suggest_mitigations(patient_data)
        
        elif action_name == "utter_anything_else":
            return "Is there anything else you'd like to know or discuss?"
        
        elif action_name == "utter_goodbye":
            return "Take care and don't hesitate to return if you have more questions. Goodbye!"
        
        else:
            logger.warning(f"Unknown action: {action_name}")
            return "I'm not sure how to respond to that."
    
    async def process_message(self, user_id: str, message: str) -> str:
        """
        Process a user message and return the response
        
        Args:
            user_id: The user's identifier
            message: The user's message
            
        Returns:
            Bot response text
        """
        # Get user's history and data
        history = self.get_chat_history(user_id)
        user_data = self.get_user_data(user_id)
        
        # Ensure patient data exists
        patient_data = self.diagnostic_engine.get_patient_data(user_data)
        
        # Store the original message for context in out_of_scope handling
        self.dialog_manager.store_original_message(user_id, message)
        
        # Add user message to history
        history.add_user_message(message)
        
        # Classify intent
        intents = await self.intent_classifier.classify_intent(message)
        top_intent = max(intents.items(), key=lambda x: x[1])[0]
        top_score = max(intents.items(), key=lambda x: x[1])[1]
        
        logger.info(f"User message: {message}")
        logger.info(f"Current state: {self.dialog_manager.get_user_state(user_id)}")
        logger.info(f"Classified intent: {top_intent} (score: {top_score:.2f})")
        
        # Extract symptoms if the intent is about symptoms
        if top_intent == "inform_symptoms":
            # Add to symptoms if not already present
            self.diagnostic_engine.add_symptom(patient_data, message)
            
            # Update confidence after adding a new symptom
            await self.diagnostic_engine.update_diagnosis_confidence(patient_data)
            
            # Check if we should transition to verification based on confidence
            if self.dialog_manager.should_verify_symptoms(user_id, patient_data):
                # This will update the state to verification if needed
                logger.info(f"Confidence threshold reached, transitioning to verification")
        
        # Determine next state based on current state and intent
        next_state = self.dialog_manager.get_next_state(user_id, top_intent)
        
        # Get the next actions to execute
        next_actions = self.dialog_manager.get_next_actions(next_state)
        
        # Execute actions and collect responses
        responses = []
        for action in next_actions:
            response = await self.execute_action(action, user_id, message)
            responses.append(response)
        
        # Update user state
        self.dialog_manager.set_user_state(user_id, next_state)
        
        # Combine responses
        full_response = " ".join(responses)
        
        # Add assistant response to history
        history.add_assistant_message(full_response)
        
        return full_response

# External functions for API and CLI interfaces
async def process_message_api(message: str, user_id: str = None) -> str:
    """
    Process a single message from an API request and return the response
    
    Args:
        message: The user's message text
        user_id: Optional user ID to maintain conversation state between requests
        
    Returns:
        The bot's response as a string
    """
    # Initialize the bot
    bot = MedicalAssistantBot()
    
    # Use the provided user_id or generate a consistent one
    if not user_id:
        user_id = f"api_user_{hash(message) % 10000}"  # Simple hash-based ID if none provided
    
    try:
        # Process the message
        response = await bot.process_message(user_id, message)
        return response
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error processing API message: {str(e)}\n{error_details}")
        return f"I'm sorry, I encountered an error processing your message. Please try again."

async def interactive_conversation():
    """Run an interactive conversation with the medical assistant bot"""
    # Check for environment variables
    if not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"):
        print("\nWARNING: Azure OpenAI environment variables not set.")
        print("Using fallback responses instead of actual AI service.")
        print("\nTo use Azure OpenAI, please set:")
        print("  export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'")
        print("  export AZURE_OPENAI_API_KEY='your-api-key'")
        print("  export AZURE_OPENAI_DEPLOYMENT_NAME='o3'")
        print("  export AZURE_OPENAI_MINI_DEPLOYMENT_NAME='o3-mini'")
    
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
            
            # Print confidence if in debug mode
            if os.getenv("DEBUG_MODE") == "true":
                user_data = bot.get_user_data(user_id)
                if "patient_data" in user_data:
                    confidence = user_data["patient_data"].get("diagnosis_confidence", 0.0)
                    reasoning = user_data["patient_data"].get("confidence_reasoning", "No reasoning provided")
                    print(f"\n[DEBUG] Current diagnosis confidence: {confidence:.2f}")
                    print(f"[DEBUG] Reasoning: {reasoning}")
                
        except Exception as e:
            print(f"\nError processing message: {str(e)}")
            # Print more detailed error information
            import traceback
            print(traceback.format_exc())
            print("\nBot: I'm sorry, I encountered an error. Please try again.")

# Entry point for running the bot directly
if __name__ == "__main__":
    asyncio.run(interactive_conversation())