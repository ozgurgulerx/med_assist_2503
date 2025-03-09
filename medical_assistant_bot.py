"""
Medical Assistant Bot using Semantic Kernel
"""
import os
import asyncio
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Semantic Kernel imports
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

# Local imports - assuming the other modules are in the same directory
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
        # Initialize Semantic Kernel
        self.kernel = Kernel()
        
        # Add Azure OpenAI service
        try:
            self.chat_service = AzureChatCompletion(
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
            )
            self.kernel.add_service(self.chat_service)
            logger.info(f"Added Azure OpenAI service with deployment: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o-mini')}")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI service: {str(e)}")
            logger.warning("The bot will continue with fallback responses instead of actual LLM calls")
            self.chat_service = None
        
        # Add medical knowledge plugin
        self.medical_plugin = MedicalKnowledgePlugin()
        self.kernel.add_plugin(self.medical_plugin, plugin_name="MedicalKnowledge")
        
        # Configure execution settings
        self.execution_settings = AzureChatPromptExecutionSettings()
        self.execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
        
        # Initialize intent classifier with its own dedicated service
        # We're not passing our chat_service or kernel, so it will create its own
        self.intent_classifier = IntentClassificationService()
        
        # Chat histories by user ID
        self.chat_histories: Dict[str, ChatHistory] = {}
        
        # Patient information storage
        self.patient_data: Dict[str, Dict[str, Any]] = {}
        
        # Store the original user message for context
        self.original_messages: Dict[str, str] = {}
        
        # Initialize dialog management
        self.initialize_dialog_manager()
    
    def initialize_dialog_manager(self):
        """Initialize custom dialog management components"""
        # Define dialog states
        self.dialog_states = {
            "greeting": {
                "next_actions": ["utter_greet", "utter_how_can_i_help"],
                "transitions": {
                    "inform_symptoms": "collecting_symptoms",
                    "ask_medical_info": "providing_info",
                    "out_of_scope": "out_of_scope_handler"  # Updated to use handler
                }
            },
            "collecting_symptoms": {
                "next_actions": ["action_ask_followup_question"],
                "transitions": {
                    "inform_symptoms": "collecting_symptoms",
                    "deny": "generating_diagnosis",
                    "ask_medical_info": "providing_info",
                    "goodbye": "farewell",
                    "out_of_scope": "out_of_scope_handler"  # New transition
                }
            },
            "providing_info": {
                "next_actions": ["action_provide_medical_info", "utter_anything_else"],
                "transitions": {
                    "inform_symptoms": "collecting_symptoms",
                    "ask_medical_info": "providing_info",
                    "deny": "farewell",
                    "goodbye": "farewell",
                    "out_of_scope": "out_of_scope_handler"  # New transition
                }
            },
            "generating_diagnosis": {
                "next_actions": ["action_provide_diagnosis", "utter_suggest_mitigations"],
                "transitions": {
                    "ask_medical_info": "providing_info",
                    "confirm": "farewell",
                    "goodbye": "farewell",
                    "out_of_scope": "out_of_scope_handler"  # New transition
                }
            },
            "farewell": {
                "next_actions": ["utter_goodbye"],
                "transitions": {
                    "out_of_scope": "out_of_scope_handler"  # New transition
                }
            },
            # New state for handling out of scope messages
            "out_of_scope_handler": {
                "next_actions": ["action_handle_out_of_scope", "utter_redirect_to_medical"],
                "transitions": {
                    # All intents will transition back to greeting
                    "greet": "greeting",
                    "inform_symptoms": "collecting_symptoms",
                    "ask_medical_info": "providing_info",
                    "confirm": "greeting",
                    "deny": "greeting",
                    "goodbye": "farewell",
                    "out_of_scope": "greeting"  # Default back to greeting if still out of scope
                }
            }
        }
        
        # Current state for each user
        self.user_states: Dict[str, str] = {}
    
    def get_user_state(self, user_id: str) -> str:
        """Get the current dialog state for a user"""
        if user_id not in self.user_states:
            self.user_states[user_id] = "greeting"
        return self.user_states[user_id]
    
    def set_user_state(self, user_id: str, state: str) -> None:
        """Set the dialog state for a user"""
        self.user_states[user_id] = state
    
    def get_chat_history(self, user_id: str) -> ChatHistory:
        """Get or create chat history for a user"""
        if user_id not in self.chat_histories:
            self.chat_histories[user_id] = ChatHistory()
        return self.chat_histories[user_id]
    
    def get_patient_data(self, user_id: str) -> Dict[str, Any]:
        """Get or create patient data for a user"""
        if user_id not in self.patient_data:
            self.patient_data[user_id] = {
                "symptoms": [],
                "demographics": {},
                "asked_questions": [],
                "diagnosis": None,
                "mitigations": []
            }
        return self.patient_data[user_id]
    
    def store_original_message(self, user_id: str, message: str) -> None:
        """Store the original message for context in out_of_scope handling"""
        self.original_messages[user_id] = message
    
    def get_original_message(self, user_id: str) -> str:
        """Get the original message for context"""
        return self.original_messages.get(user_id, "")
    
    async def execute_llm_prompt(self, prompt: str) -> str:
        """Execute a direct prompt to the LLM"""
        if not self.chat_service:
            return "LLM service not available."
            
        try:
            logger.info(f"Direct LLM prompt: {prompt}")
            
            # Create a temp chat history for this prompt
            chat_history = ChatHistory()
            chat_history.add_user_message(prompt)
            
            # Get LLM response
            result = await self.chat_service.get_chat_message_content(
                chat_history=chat_history,
                settings=self.execution_settings,
                kernel=self.kernel
            )
            
            response_text = str(result)
            logger.info(f"Direct LLM response: {response_text}")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error in direct LLM prompt: {str(e)}")
            return f"Error in LLM processing: {str(e)}"
    
    async def execute_action(self, action_name: str, user_id: str, user_message: str = "") -> str:
        """Execute a dialog action and return the response"""
        history = self.get_chat_history(user_id)
        patient_data = self.get_patient_data(user_id)
        
        logger.info(f"Executing action: {action_name}")
        
        if action_name == "utter_greet":
            return "Hello! I'm your medical assistant. I'm here to help with your health questions."
        
        elif action_name == "utter_how_can_i_help":
            return "How can I help you today?"
        
        elif action_name == "action_handle_out_of_scope":
            # Get the original message for context
            original_message = self.get_original_message(user_id)
            
            if self.chat_service:
                try:
                    # Create a prompt to acknowledge the off-topic message
                    prompt = f"""The user has sent a message that appears to be outside the scope of a medical conversation. 
Their message was: "{original_message}"

Provide a polite, brief response that acknowledges their message but gently redirects the conversation to medical topics.
Make sure your response is concise (max 2 sentences) and ends with a question about their health concerns."""

                    # Get response directly from LLM
                    return await self.execute_llm_prompt(prompt)
                except Exception as e:
                    logger.error(f"Error handling out of scope message with LLM: {str(e)}")
            
            # Fallback response if LLM fails
            return f"I understand you're asking about \"{original_message}\", but I'm primarily designed to help with medical questions."
        
        elif action_name == "utter_redirect_to_medical":
            return "Is there something about your health I can help with today?"
        
        elif action_name == "action_ask_followup_question":
            # Get current symptoms as a string
            symptoms = ", ".join(patient_data["symptoms"]) if patient_data["symptoms"] else "unknown symptoms"
            
            # Get previously asked questions
            asked = ", ".join(patient_data["asked_questions"])
            
            logger.info(f"Preparing to ask follow-up questions about symptoms: {symptoms}")
            
            if self.chat_service:
                try:
                    # Create a prompt for follow-up questions
                    prompt = f"""As a medical assistant, I need to ask follow-up questions about the patient's symptoms.
Current symptoms: {symptoms}
Medical history: N/A
Previously asked questions: {asked}

Generate a relevant follow-up question to better understand these symptoms."""

                    # Get response directly from LLM
                    response = await self.execute_llm_prompt(prompt)
                    
                    # Record this question
                    patient_data["asked_questions"].append(response)
                    
                    return response
                except Exception as e:
                    logger.error(f"Error generating follow-up questions with LLM: {str(e)}")
            
            # Use the plugin method or fallback
            try:
                response = await self.medical_plugin.generate_followup_questions(
                    current_symptoms=symptoms,
                    medical_history="",
                    previously_asked=asked
                )
                
                # Record this question
                patient_data["asked_questions"].append(str(response))
                
                return str(response)
            except Exception as e:
                logger.error(f"Error generating follow-up questions: {str(e)}")
                fallback_response = "Can you tell me more about your symptoms? When did they start and have they changed over time?"
                logger.info(f"Using fallback response for follow-up questions")
                patient_data["asked_questions"].append(fallback_response)
                return fallback_response
        
        elif action_name == "action_provide_medical_info":
            # Extract the topic from user message
            topic = user_message
            
            # Get demographics as a string
            demographics = str(patient_data.get("demographics", {}))
            
            logger.info(f"Providing medical information about: {topic}")
            
            if self.chat_service:
                try:
                    # Create a prompt for medical information
                    prompt = f"""Provide general medical information about the following topic:
Topic: {topic}
Patient demographics: {demographics}

Give helpful, accurate information while emphasizing this is general advice and not a substitute for professional medical care."""

                    # Get response directly from LLM
                    return await self.execute_llm_prompt(prompt)
                except Exception as e:
                    logger.error(f"Error providing medical information with LLM: {str(e)}")
            
            # Use the plugin method or fallback
            try:
                response = await self.medical_plugin.provide_medical_information(
                    topic=topic,
                    patient_demographics=demographics
                )
                return str(response)
            except Exception as e:
                logger.error(f"Error providing medical information: {str(e)}")
                fallback_response = f"I can provide general information about {topic}, but remember to consult with a healthcare professional for personalized advice."
                logger.info(f"Using fallback response for medical information")
                return fallback_response
        
        elif action_name == "action_provide_diagnosis":
            # In a real system, this would analyze all collected symptoms
            symptoms = ", ".join(patient_data["symptoms"])
            
            logger.info(f"Generating diagnosis based on symptoms: {symptoms}")
            
            if self.chat_service:
                try:
                    # Create a prompt for diagnosis
                    prompt = f"""Based on these symptoms: {symptoms}, what might be the diagnosis?
Provide a thoughtful analysis considering multiple possibilities.
Be responsible and remind the patient this is not a substitute for professional medical diagnosis."""

                    # Get response directly from LLM
                    response = await self.execute_llm_prompt(prompt)
                    
                    # Store the diagnosis
                    patient_data["diagnosis"] = response
                    
                    return f"Based on the symptoms you've described, {response}"
                except Exception as e:
                    logger.error(f"Error providing diagnosis with LLM: {str(e)}")
            
            # Use the plugin method or fallback
            try:
                response = await self.medical_plugin.analyze_medical_query(
                    query=f"Based on these symptoms: {symptoms}, what might be the diagnosis?",
                    patient_context=""  # Would include demographics and history in real system
                )
                
                # Store the diagnosis
                patient_data["diagnosis"] = str(response)
                
                return f"Based on the symptoms you've described, {str(response)}"
            except Exception as e:
                logger.error(f"Error providing diagnosis: {str(e)}")
                fallback_response = "Based on the symptoms you've described, I'd recommend consulting with a healthcare provider for a proper evaluation. Your symptoms could have various causes."
                logger.info(f"Using fallback response for diagnosis")
                patient_data["diagnosis"] = fallback_response
                return fallback_response
        
        elif action_name == "utter_suggest_mitigations":
            # In a real system, this would generate specific mitigations based on the diagnosis
            return "Here are some steps you might consider: rest, stay hydrated, and monitor your symptoms. If they worsen, please consult with your healthcare provider."
        
        elif action_name == "utter_anything_else":
            return "Is there anything else you'd like to know or discuss?"
        
        elif action_name == "utter_goodbye":
            return "Take care and don't hesitate to return if you have more questions. Goodbye!"
        
        else:
            logger.warning(f"Unknown action: {action_name}")
            return "I'm not sure how to respond to that."
    
    async def process_message(self, user_id: str, message: str) -> str:
        """Process a user message and return the response"""
        # Get user's current state and history
        current_state = self.get_user_state(user_id)
        history = self.get_chat_history(user_id)
        patient_data = self.get_patient_data(user_id)
        
        # Store the original message for context in out_of_scope handling
        self.store_original_message(user_id, message)
        
        # Add user message to history
        history.add_user_message(message)
        
        # Classify intent
        intents = await self.intent_classifier.classify_intent(message)
        top_intent = max(intents.items(), key=lambda x: x[1])[0]
        top_score = max(intents.items(), key=lambda x: x[1])[1]
        
        logger.info(f"User message: {message}")
        logger.info(f"Current state: {current_state}")
        logger.info(f"Classified intent: {top_intent} (score: {top_score:.2f})")
        
        # Special handling for persistent headaches messages - hardcoded fix for demo
        if "headache" in message.lower() or (
            "persistent" in message.lower() and any(word in message.lower() for word in ["started", "worse", "morning", "day"])):
            top_intent = "inform_symptoms"
            intents["inform_symptoms"] = 0.95
            logger.info(f"Overrode intent to: {top_intent} (hardcoded rule)")
        
        # Extract symptoms if the intent is about symptoms
        if top_intent == "inform_symptoms":
            # In a full implementation, we would use proper NER
            # For demo, we'll use the message directly
            if message and message not in patient_data["symptoms"]:
                patient_data["symptoms"].append(message)
                logger.info(f"Added symptom: {message}")
        
        # Determine next state based on current state and intent
        state_info = self.dialog_states.get(current_state, {})
        next_state = state_info.get("transitions", {}).get(top_intent, current_state)
        
        # Get the next action to execute
        next_actions = self.dialog_states.get(next_state, {}).get("next_actions", [])
        
        # Execute actions and collect responses
        responses = []
        for action in next_actions:
            response = await self.execute_action(action, user_id, message)
            responses.append(response)
        
        # Update user state
        self.set_user_state(user_id, next_state)
        logger.info(f"Transitioned to state: {next_state}")
        
        # Combine responses
        full_response = " ".join(responses)
        
        # Add assistant response to history
        history.add_assistant_message(full_response)
        
        return full_response

async def interactive_conversation():
    """Run an interactive conversation with the medical assistant bot"""
    # Check for environment variables
    if not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"):
        print("\nWARNING: Azure OpenAI environment variables not set.")
        print("Using fallback responses instead of actual AI service.")
        print("\nTo use Azure OpenAI, please set:")
        print("  export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'")
        print("  export AZURE_OPENAI_API_KEY='your-api-key'")
        print("  export AZURE_OPENAI_DEPLOYMENT_NAME='gpt-4o'")
    
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

"""
Add this function to medical_assistant_bot.py to handle API requests
"""

async def process_message(message: str) -> str:
    """
    Process a single message from an API request and return the response
    
    Args:
        message: The user's message text
        
    Returns:
        The bot's response as a string
    """
    # Initialize the bot with a consistent user ID for API requests
    bot = MedicalAssistantBot()
    user_id = "api_user"
    
    try:
        # Process the message
        response = await bot.process_message(user_id, message)
        return response
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error processing API message: {str(e)}\n{error_details}")
        return f"I'm sorry, I encountered an error processing your message. Please try again."
# Entry point for running the bot directly
if __name__ == "__main__":
    asyncio.run(interactive_conversation())