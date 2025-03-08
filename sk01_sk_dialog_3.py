import os
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

# Semantic Kernel imports
from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MedicalKnowledgePlugin:
    """Flexible plugin for medical knowledge with no hardcoded medical conditions"""
    
    @kernel_function
    async def analyze_medical_query(self, query: str, patient_context: str = "") -> str:
        """
        Analyze a medical query to determine the appropriate response or follow-up questions.
        
        Args:
            query: The patient's medical query or description of symptoms
            patient_context: Optional context about the patient's history, demographics, or previously mentioned symptoms
        """
        logger.info(f"LLM Request - analyze_medical_query: {query}")
        # This would use the model to provide a flexible response without hardcoding conditions
        response = f"Based on your query about '{query}', I'll need to ask some follow-up questions to better understand your situation."
        logger.info(f"LLM Response - analyze_medical_query: {response}")
        return response
    
    @kernel_function
    async def generate_followup_questions(self, current_symptoms: str, medical_history: str = "", previously_asked: str = "") -> str:
        """
        Generate relevant follow-up questions based on current symptoms and conversation context.
        
        Args:
            current_symptoms: Symptoms mentioned so far
            medical_history: Patient's relevant medical history if available
            previously_asked: Questions already asked to avoid repetition
        """
        logger.info(f"LLM Request - generate_followup_questions - Symptoms: {current_symptoms}, History: {medical_history}, Previous: {previously_asked}")
        # This would dynamically generate questions based on the specific symptoms mentioned
        response = "Could you describe when these symptoms started and if anything makes them better or worse?"
        logger.info(f"LLM Response - generate_followup_questions: {response}")
        return response
    
    @kernel_function
    async def provide_medical_information(self, topic: str, patient_demographics: str = "") -> str:
        """
        Provide general medical information tailored to a patient.
        
        Args:
            topic: The medical topic being discussed
            patient_demographics: Optional demographic information to personalize the response
        """
        logger.info(f"LLM Request - provide_medical_information - Topic: {topic}, Demographics: {patient_demographics}")
        response = f"Here's some general information about {topic}. Remember that this is general advice and not a substitute for professional medical care."
        logger.info(f"LLM Response - provide_medical_information: {response}")
        return response

class IntentClassificationService:
    """Service for classifying user intents using LLM when available, with pattern matching as fallback"""
    
    def __init__(self, chat_service: AzureChatCompletion = None, kernel: Kernel = None):
        """
        Initialize the intent classifier
        
        Args:
            chat_service: Azure Chat Completion service for LLM-based classification
            kernel: Semantic Kernel instance for plugin access
        """
        self.chat_service = chat_service
        self.kernel = kernel
        
        self.intent_options = [
            "greet",
            "inform_symptoms",
            "ask_medical_info",
            "confirm",
            "deny",
            "goodbye",
            "out_of_scope"
        ]
        
        # Enhanced pattern definitions for fallback
        self.patterns = {
            "greet": [
                r"\b(?:hello|hi|hey|greetings|good\s*(?:morning|afternoon|evening))\b"
            ],
            "inform_symptoms": [
                r"\b(?:headache|pain|ache|hurt|sore|dizzy|nausea|vomit|fever|cough|sneeze|rash|swelling|breathing|tired|fatigue|exhausted|symptom|sick|ill|unwell)\b",
                r"\b(?:feeling|experienc(?:e|ing)|suffer(?:ing)?|having)\b",
                r"\b(?:since|started|begin|began|worse|better|worsen|improve)\b",
                r"\b(?:morning|night|day|week|daily|constant|persistent|chronic|acute)\b"
            ],
            "ask_medical_info": [
                r"(?:what|how|why|when|where|which|can|should)\s.+(?:\?|$)",
                r"\b(?:tell me about|explain|information|advice|recommend|suggest)\b",
                r"\b(?:treatment|medicine|medication|drug|therapy|doctor|normal|average|typical|blood pressure|cholesterol|diabetes|heart)\b"
            ],
            "confirm": [
                r"\b(?:yes|yeah|yep|correct|right|exactly|agree|true|affirmative)\b"
            ],
            "deny": [
                r"\b(?:no|nope|not|don't|doesn't|didn't|haven't|hasn't|can't|none|never)\b"
            ],
            "goodbye": [
                r"\b(?:bye|goodbye|farewell|thanks|thank you|appreciate|good night)\b"
            ]
        }
        
        # Compile all patterns
        import re
        self.compiled_patterns = {}
        for intent, pattern_list in self.patterns.items():
            self.compiled_patterns[intent] = [re.compile(pattern, re.IGNORECASE) for pattern in pattern_list]
    
    async def classify_intent_with_patterns(self, utterance: str) -> Dict[str, float]:
        """
        Classify intent using pattern matching as fallback
        
        Args:
            utterance: The user's message
            
        Returns:
            Dictionary of intent names and confidence scores
        """
        logger.info("Using pattern-based intent classification")
        
        # Initialize scores for all intents
        scores = {intent: 0.0 for intent in self.intent_options}
        scores["out_of_scope"] = 0.1  # Default low score for out_of_scope
        
        # Calculate scores based on pattern matches
        for intent, patterns in self.compiled_patterns.items():
            matches = sum(1 for pattern in patterns if pattern.search(utterance))
            if matches > 0:
                # Calculate score based on number of matches
                # More matches = higher confidence
                base_score = 0.5
                match_score = min(matches * 0.15, 0.4)  # Up to 0.4 for multiple matches
                scores[intent] = base_score + match_score
        
        # Special case: if inform_symptoms has evidence but not very strong,
        # and the text is long enough, increase its score
        if 0.2 < scores["inform_symptoms"] < 0.7 and len(utterance.split()) > 5:
            scores["inform_symptoms"] += 0.15
        
        # If asking a question, boost ask_medical_info
        if "?" in utterance:
            scores["ask_medical_info"] = max(scores["ask_medical_info"], 0.75)
        
        # If no intent has significant confidence, boost out_of_scope
        if all(score < 0.4 for intent, score in scores.items() if intent != "out_of_scope"):
            scores["out_of_scope"] = 0.5
        
        # Log the scores
        logger.info(f"Pattern-based intent scores: {scores}")
        
        return scores
    
    async def classify_intent(self, utterance: str) -> Dict[str, float]:
        """
        Classify the intent of a user utterance using pattern matching
        
        Args:
            utterance: The user's message
        
        Returns:
            Dictionary of intent names and confidence scores
        """
        if not utterance.strip():
            return {"out_of_scope": 1.0}
        
        # Use pattern-based classification
        return await self.classify_intent_with_patterns(utterance)

async def assess_symptom_confidence(symptom_list: List[str]) -> Tuple[float, str]:
    """
    Assess the confidence level in the collected symptoms.
    
    Args:
        symptom_list: List of symptoms collected from the user
        
    Returns:
        Tuple of (confidence score, explanation)
    """
    if not symptom_list:
        return 0.0, "No symptoms provided"
    
    # Basic assessment based on symptom count and length
    symptom_count = len(symptom_list)
    total_words = sum(len(symptom.split()) for symptom in symptom_list)
    
    confidence = min(0.3 + symptom_count * 0.15 + total_words * 0.01, 1.0)
    
    explanation = f"Based on {symptom_count} symptoms with {total_words} total words"
    if confidence < 0.5:
        missing = "more specific symptom descriptions, duration, and severity information"
    elif confidence < 0.7:
        missing = "information about triggers and patterns"
    else:
        missing = "additional context might help, but we have good information"
    
    return confidence, f"{explanation}. Missing: {missing}"

class MedicalAssistantBot:
    """Flexible medical assistant that can handle any medical issue"""
    
    def __init__(self):
        # Initialize Semantic Kernel
        self.kernel = Kernel()
        
        # Add new parameters for symptom collection control
        self.max_followup_questions = 3  # Maximum number of follow-up questions before offering diagnosis
        self.symptom_confidence_threshold = 0.7  # Confidence threshold to move to diagnosis
        self.common_affirmative_phrases = [
            "that's all", "that is all", "that's it", "that is it", 
            "no more symptoms", "nothing else", "done", "proceed", 
            "that's everything", "that is everything", "go ahead",
            "what do you think", "what's your assessment", "diagnose", 
            "tell me what it is", "what could it be", "assessment"
        ]
        
        # Add Azure OpenAI service
        try:
            # Debug log to show what we're attempting to use
            logger.info(f"Attempting to initialize Azure OpenAI with endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
            logger.info(f"Using deployment name: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o-mini')}")
            logger.info(f"API version: {os.getenv('AZURE_OPENAI_API_VERSION', '2024-06-01')}")
            
            # Check if key is set (don't log the actual key for security)
            if os.getenv("AZURE_OPENAI_API_KEY"):
                logger.info("API key is set")
            else:
                logger.warning("API key is not set")
                
            self.chat_service = AzureChatCompletion(
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
            )
            self.kernel.add_service(self.chat_service)
            logger.info(f"Successfully added Azure OpenAI service")
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
        
        # Initialize intent classifier with the chat service and kernel
        self.intent_classifier = IntentClassificationService(
            chat_service=self.chat_service,
            kernel=self.kernel
        )
        
        # Chat histories by user ID
        self.chat_histories: Dict[str, ChatHistory] = {}
        
        # Patient information storage
        self.patient_data: Dict[str, Dict[str, Any]] = {}
        
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
                    "out_of_scope": "greeting"
                }
            },
            "collecting_symptoms": {
                "next_actions": ["action_ask_followup_question"],
                "transitions": {
                    "inform_symptoms": "collecting_symptoms",
                    "deny": "generating_diagnosis",
                    "confirm": "generating_diagnosis",  # Added confirmation transition
                    "ask_medical_info": "providing_info",
                    "goodbye": "farewell"
                }
            },
            "providing_info": {
                "next_actions": ["action_provide_medical_info", "utter_anything_else"],
                "transitions": {
                    "inform_symptoms": "collecting_symptoms",
                    "ask_medical_info": "providing_info",
                    "deny": "farewell",
                    "goodbye": "farewell"
                }
            },
            "generating_diagnosis": {
                "next_actions": ["action_provide_diagnosis", "utter_suggest_mitigations"],
                "transitions": {
                    "ask_medical_info": "providing_info",
                    "confirm": "farewell",
                    "goodbye": "farewell"
                }
            },
            "farewell": {
                "next_actions": ["utter_goodbye"],
                "transitions": {}
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
    
    async def check_symptom_completeness(self, user_id: str) -> bool:
        """Check if we have collected enough symptoms to proceed to diagnosis"""
        patient_data = self.get_patient_data(user_id)
        
        # Check if we've asked enough follow-up questions
        if len(patient_data["asked_questions"]) >= self.max_followup_questions:
            logger.info(f"Maximum follow-up questions reached ({self.max_followup_questions}), proceeding to diagnosis")
            return True
        
        # Check if we have at least some symptoms with sufficient details
        if len(patient_data["symptoms"]) >= 2:
            # In a real system, we would do more sophisticated analysis here
            total_symptom_words = sum(len(symptom.split()) for symptom in patient_data["symptoms"])
            if total_symptom_words >= 15:  # If we have at least 15 words describing symptoms
                logger.info(f"Sufficient symptom details collected (words: {total_symptom_words}), proceeding to diagnosis")
                return True
        
        return False
    
    async def check_diagnosis_readiness(self, user_id: str, message: str) -> bool:
        """
        Determine if the system should transition to diagnosis state
        
        Args:
            user_id: The user ID
            message: The current user message
            
        Returns:
            Boolean indicating if ready for diagnosis
        """
        patient_data = self.get_patient_data(user_id)
        
        # 1. Direct user signals
        for signal in self.common_affirmative_phrases:
            if signal in message.lower():
                logger.info(f"User signal for diagnosis detected: '{signal}'")
                return True
        
        # 2. Message indicates "no" in response to "anything else" context
        if patient_data["asked_questions"]:
            last_question = patient_data["asked_questions"][-1].lower()
            asking_for_more = any(phrase in last_question for phrase in [
                "anything else", "any other", "more symptoms", 
                "something else", "additional", "other concern"
            ])
            
            if asking_for_more and any(word in message.lower() for word in [
                "no", "nope", "that's all", "nothing", "none"
            ]):
                logger.info("User indicated no more symptoms to share")
                return True
        
        # 3. Count-based thresholds
        if len(patient_data["symptoms"]) >= 3:
            logger.info(f"Symptom count threshold reached: {len(patient_data['symptoms'])}")
            return True
            
        if len(patient_data["asked_questions"]) >= self.max_followup_questions:
            logger.info(f"Max questions threshold reached: {len(patient_data['asked_questions'])}")
            return True
        
        # 4. Content-based assessment using symptom confidence function
        if len(patient_data["symptoms"]) > 0:
            try:
                confidence, explanation = await assess_symptom_confidence(patient_data["symptoms"])
                
                logger.info(f"Symptom confidence assessment: {confidence:.2f} - {explanation}")
                
                if confidence >= self.symptom_confidence_threshold:
                    return True
                    
            except Exception as e:
                logger.error(f"Error in symptom confidence assessment: {str(e)}")
        
        return False
    
    async def check_and_offer_diagnosis(self, user_id: str) -> str:
        """
        Check if we should offer diagnosis and create appropriate prompt
        
        Args:
            user_id: The user ID
            
        Returns:
            Response text offering diagnosis or asking for more information
        """
        patient_data = self.get_patient_data(user_id)
        
        # If we've asked several questions, offer diagnosis explicitly
        if len(patient_data["asked_questions"]) >= self.max_followup_questions - 1:
            return "I've gathered several details about your symptoms. Would you like me to provide an assessment now, or is there anything else you'd like to tell me about your condition?"
        
        # If few symptoms but many follow-ups, suspect we need to move on
        if len(patient_data["symptoms"]) <= 2 and len(patient_data["asked_questions"]) >= 2:
            return "Based on what you've shared so far, I can offer some initial thoughts. Would you like me to proceed with an assessment, or do you have more symptoms to share?"
        
        # Default follow-up that hints at diagnosis option
        return "Is there anything else about your symptoms you'd like to share? When you're ready for my assessment, just let me know."
    
    async def execute_action(self, action_name: str, user_id: str, user_message: str = "") -> str:
        """Execute a dialog action and return the response"""
        history = self.get_chat_history(user_id)
        patient_data = self.get_patient_data(user_id)
        
        logger.info(f"Executing action: {action_name}")
        
        if action_name == "utter_greet":
            return "Hello! I'm your medical assistant. I'm here to help with your health questions."
        
        elif action_name == "utter_how_can_i_help":
            return "How can I help you today?"
        
        elif action_name == "action_ask_followup_question":
            # Check if the user message indicates they're done sharing symptoms
            for phrase in self.common_affirmative_phrases:
                if phrase in user_message.lower():
                    logger.info(f"User indicated completion of symptom sharing with: '{user_message}'")
                    self.set_user_state(user_id, "generating_diagnosis")
                    return "Thank you for sharing your symptoms. Let me analyze this information to provide an assessment."
            
            # Every other question, check if we should offer diagnosis
            if len(patient_data["asked_questions"]) % 2 == 0 and len(patient_data["asked_questions"]) > 0:
                return await self.check_and_offer_diagnosis(user_id)
            
            # Get current symptoms as a string
            symptoms = ", ".join(patient_data["symptoms"]) if patient_data["symptoms"] else "unknown symptoms"
            
            # Get previously asked questions
            asked = ", ".join(patient_data["asked_questions"])
            
            logger.info(f"Preparing to ask follow-up questions about symptoms: {symptoms}")
            
            # Generate a follow-up question
            fallback_response = "Can you tell me more about your symptoms? When did they start and have they changed over time?"
            
            if len(patient_data["asked_questions"]) == 0:
                question = "Can you describe when these symptoms started and how severe they are?"
            elif len(patient_data["asked_questions"]) == 1:
                question = "Is there anything that makes these symptoms better or worse?"
            elif len(patient_data["asked_questions"]) == 2:
                question = "Have you experienced these symptoms before, or is this the first time?"
            else:
                question = "Would you like me to provide an assessment based on what you've shared so far?"
                
            # Record this question
            patient_data["asked_questions"].append(question)
            return question
        
        elif action_name == "action_provide_medical_info":
            # Extract the topic from user message
            topic = user_message.strip()
            if not topic:
                topic = "general health"
            
            # Provide simple information
            return f"Here's some general information about {topic}. Remember that this is general advice and not a substitute for professional medical care."
        
        elif action_name == "action_provide_diagnosis":
            # In a real system, this would analyze all collected symptoms
            symptoms = ", ".join(patient_data["symptoms"])
            
            logger.info(f"Generating diagnosis based on symptoms: {symptoms}")
            
            # Generate a simple diagnosis based on collected symptoms
            if "headache" in symptoms.lower():
                diagnosis = "Based on your symptoms, you may be experiencing tension headaches. These are common and can be caused by stress, dehydration, or lack of sleep."
            elif "fever" in symptoms.lower():
                diagnosis = "Your symptoms suggest you might have a viral infection. Fevers are a common response to infections as your body fights them off."
            elif "cough" in symptoms.lower() or "sneeze" in symptoms.lower():
                diagnosis = "Your symptoms are consistent with an upper respiratory infection, which could be a common cold or seasonal allergies."
            else:
                diagnosis = "Based on the symptoms you've described, there could be several potential causes. It would be best to consult with a healthcare provider for a proper evaluation."
            
            # Store the diagnosis
            patient_data["diagnosis"] = diagnosis
            
            return diagnosis
        
        elif action_name == "utter_suggest_mitigations":
            # In a real system, this would generate specific mitigations based on the diagnosis
            return "Here are some general steps you might consider: rest, stay hydrated, and monitor your symptoms. If they worsen or persist for more than a few days, please consult with your healthcare provider."
        
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
        
        # Add user message to history
        history.add_user_message(message)
        
        # IMPROVED HANDLING: Check for diagnosis readiness in collecting_symptoms state
        if current_state == "collecting_symptoms":
            ready_for_diagnosis = await self.check_diagnosis_readiness(user_id, message)
            
            if ready_for_diagnosis:
                logger.info("System determined readiness for diagnosis")
                
                # First transition to diagnosis state
                self.set_user_state(user_id, "generating_diagnosis")
                
                # Create a transition message
                transition_response = "Based on the information you've provided, I think I have enough to offer an assessment."
                
                # Execute diagnosis actions
                diagnosis_response = await self.execute_action("action_provide_diagnosis", user_id, message)
                mitigation_response = await self.execute_action("utter_suggest_mitigations", user_id, message)
                
                # Combine responses
                full_response = f"{transition_response} {diagnosis_response} {mitigation_response}"
                
                # Add to history
                history.add_assistant_message(full_response)
                return full_response
        
        # Classify intent
        intents = await self.intent_classifier.classify_intent(message)
        top_intent = max(intents.items(), key=lambda x: x[1])[0]
        top_score = max(intents.items(), key=lambda x: x[1])[1]
        
        logger.info(f"User message: {message}")
        logger.info(f"Current state: {current_state}")
        logger.info(f"Classified intent: {top_intent} (score: {top_score:.2f})")
        
        # Extract symptoms if the intent is about symptoms
        if top_intent == "inform_symptoms":
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
    # Display environment variable status
    print("\n----- Environment Status -----")
    print(f"AZURE_OPENAI_ENDPOINT: {'SET' if os.getenv('AZURE_OPENAI_ENDPOINT') else 'NOT SET'}")
    print(f"AZURE_OPENAI_API_KEY: {'SET' if os.getenv('AZURE_OPENAI_API_KEY') else 'NOT SET'}")
    print(f"AZURE_OPENAI_DEPLOYMENT_NAME: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'NOT SET (will use default)')}")
    print(f"AZURE_OPENAI_API_VERSION: {os.getenv('AZURE_OPENAI_API_VERSION', 'NOT SET (will use default)')}")
    
    print("\nWARNING: If Azure OpenAI environment variables are not set, the bot will use fallback responses.")
    print("This is fine for testing the conversation flow, but responses won't use AI capabilities.")
    
    # Create and initialize the bot
    print("\n----- Initializing Medical Assistant Bot -----")
    bot = MedicalAssistantBot()
    user_id = "interactive_user"
    
    print("\n----- Starting Interactive Medical Assistant Conversation -----")
    print("Type your messages and press Enter. Type 'exit', 'quit', or 'bye' to end the conversation.\n")
    
    # Initial greeting
    initial_response = await bot.execute_action("utter_greet", user_id)
    followup = await bot.execute_action("utter_how_can_i_help", user_id)
    print(f"Bot: {initial_response} {followup}")
    
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

# Entry point for running the bot directly - THIS IS THE CRITICAL PART
if __name__ == "__main__":
    print("Starting Medical Assistant Bot...")
    asyncio.run(interactive_conversation())