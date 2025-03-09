"""
Medical Assistant Bot using Semantic Kernel with dual-model verification approach
Only provides diagnosis when both primary and verification models have high confidence
"""
import os
import asyncio
import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

# Semantic Kernel imports
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

# Local imports
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
    """LLM-driven medical assistant with dual-model verification for high confidence diagnosis"""
    
    def __init__(self):
        # Initialize Semantic Kernel
        self.kernel = Kernel()
        
        # Add primary Azure OpenAI service (gpt-4o)
        try:
            self.primary_chat_service = AzureChatCompletion(
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
            )
            self.kernel.add_service(self.primary_chat_service)
            logger.info(f"Added primary Azure OpenAI service with deployment: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o')}")
        except Exception as e:
            logger.error(f"Failed to initialize primary Azure OpenAI service: {str(e)}")
            logger.warning("The bot will continue with fallback responses instead of actual LLM calls")
            self.primary_chat_service = None
        
        # Add verification Azure OpenAI service (o1)
        try:
            self.verification_chat_service = AzureChatCompletion(
                deployment_name=os.getenv("VERIFICATION_OPENAI_DEPLOYMENT_NAME", "o1"),
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2025-01-01-preview"
            )
            logger.info(f"Added verification Azure OpenAI service with deployment: {os.getenv('VERIFICATION_OPENAI_DEPLOYMENT_NAME', 'o1')}")
        except Exception as e:
            logger.error(f"Failed to initialize verification Azure OpenAI service: {str(e)}")
            logger.warning("The bot will continue without verification model checks")
            self.verification_chat_service = None
        
        # Configure execution settings
        self.execution_settings = AzureChatPromptExecutionSettings()
        self.execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
        
        # Initialize intent classifier
        self.intent_classifier = IntentClassificationService()
        
        # Chat histories by user ID
        self.chat_histories: Dict[str, ChatHistory] = {}
        
        # Patient information storage
        self.patient_data: Dict[str, Dict[str, Any]] = {}
        
        # Initialize dialog management
        self.initialize_dialog_manager()
    
    def initialize_dialog_manager(self):
        """Initialize dialog management components"""
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
                "next_actions": ["action_assess_and_follow_up"],
                "transitions": {
                    "inform_symptoms": "collecting_symptoms",
                    "ask_medical_info": "answering_patient_question",
                    "diagnosis_ready": "generating_diagnosis",
                    "goodbye": "farewell"
                }
            },
            "answering_patient_question": {
                "next_actions": ["action_answer_patient_question", "action_return_to_assessment"],
                "transitions": {
                    "inform_symptoms": "collecting_symptoms",
                    "ask_medical_info": "answering_patient_question",
                    "diagnosis_ready": "generating_diagnosis",
                    "goodbye": "farewell"
                }
            },
            "providing_info": {
                "next_actions": ["action_provide_medical_info", "utter_anything_else"],
                "transitions": {
                    "inform_symptoms": "collecting_symptoms",
                    "ask_medical_info": "providing_info",
                    "goodbye": "farewell"
                }
            },
            "generating_diagnosis": {
                "next_actions": ["action_provide_diagnosis", "action_suggest_mitigations"],
                "transitions": {
                    "inform_symptoms": "collecting_symptoms",
                    "ask_medical_info": "answering_followup_question",
                    "goodbye": "farewell"
                }
            },
            "answering_followup_question": {
                "next_actions": ["action_answer_followup_question"],
                "transitions": {
                    "inform_symptoms": "collecting_symptoms",
                    "ask_medical_info": "answering_followup_question",
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
                "conversation_history": [],
                "symptoms": [],
                "primary_diagnosis_confidence": 0.0,
                "verification_diagnosis_confidence": 0.0,
                "potential_diagnoses": [],
                "diagnosis": None,
                "waiting_for_answer": False,
                "current_question": None,
                "next_intent": None,
                "question_count": 0,
                "verification_attempted": False,
                "verification_diagnoses": []
            }
        return self.patient_data[user_id]
    
    def _build_conversation_context(self, user_id: str) -> str:
        """Build a conversation context string"""
        patient_data = self.get_patient_data(user_id)
        
        # Use saved conversation history
        if patient_data.get("conversation_history"):
            context_parts = []
            for entry in patient_data["conversation_history"]:
                role = "User" if entry["role"] == "user" else "Assistant"
                context_parts.append(f"{role}: {entry['content']}")
            return "\n".join(context_parts)
        
        # If no history available yet
        return ""
    
    async def execute_llm_prompt(self, prompt: str, use_verification_model: bool = False) -> str:
        """Execute a direct prompt to the LLM"""
        # Choose which model to use
        chat_service = self.verification_chat_service if use_verification_model else self.primary_chat_service
        model_name = "verification model (o1)" if use_verification_model else "primary model (o1-mini)"
        
        if not chat_service:
            return f"LLM service for {model_name} not available."
            
        try:
            logger.info(f"Direct LLM prompt to {model_name}: {prompt[:100]}...")
            
            # Create a temp chat history for this prompt
            chat_history = ChatHistory()
            chat_history.add_user_message(prompt)
            
            # Get LLM response
            result = await chat_service.get_chat_message_content(
                chat_history=chat_history,
                settings=self.execution_settings,
                kernel=self.kernel
            )
            
            response_text = str(result)
            logger.info(f"Direct LLM response from {model_name}: {response_text[:100]}...")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error in direct LLM prompt to {model_name}: {str(e)}")
            return f"Error in LLM processing with {model_name}: {str(e)}"
    
    async def execute_structured_llm_prompt(self, prompt: str, use_verification_model: bool = False) -> Dict[str, Any]:
        """Execute a prompt that expects structured JSON output from the LLM"""
        # Choose which model to use
        chat_service = self.verification_chat_service if use_verification_model else self.primary_chat_service
        model_name = "verification model (o1)" if use_verification_model else "primary model (o1-mini)"
        
        if not chat_service:
            return {"error": f"LLM service for {model_name} not available."}
            
        try:
            logger.info(f"Structured LLM prompt to {model_name}: {prompt[:100]}...")
            
            # Create a temp chat history for this prompt
            chat_history = ChatHistory()
            chat_history.add_user_message(prompt)
            
            # Get LLM response
            result = await chat_service.get_chat_message_content(
                chat_history=chat_history,
                settings=self.execution_settings,
                kernel=self.kernel
            )
            
            response_text = str(result)
            logger.info(f"Structured LLM response received from {model_name}")
            
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed_result = json.loads(json_str)
                    return parsed_result
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from {model_name} response: {e}")
                    return {"error": f"Failed to parse JSON: {str(e)}", "raw_response": response_text}
            else:
                logger.warning(f"No JSON found in {model_name} response")
                return {"error": f"No JSON found in {model_name} response", "raw_response": response_text}
            
        except Exception as e:
            logger.error(f"Error in structured LLM prompt to {model_name}: {str(e)}")
            return {"error": f"Error in LLM processing with {model_name}: {str(e)}"}
    
    async def perform_verification_assessment(self, user_id: str) -> Tuple[bool, float]:
        """
        Perform verification assessment with second model (o1)
        Returns (success, confidence)
        """
        patient_data = self.get_patient_data(user_id)
        conversation_context = self._build_conversation_context(user_id)
        
        # Mark that verification was attempted
        patient_data["verification_attempted"] = True
        
        if not self.verification_chat_service:
            logger.warning("Verification model not available - skipping verification")
            return (False, 0.0)
        
        # Create assessment prompt for verification model
        verification_prompt = f"""You are a verification model assessing a patient's symptoms for a second opinion.

CONVERSATION HISTORY:
{conversation_context}

Your task is to:
1. Analyze all symptoms and information provided in the conversation
2. Make your own independent assessment of potential diagnoses
3. Determine your confidence level in a diagnosis (0-100%)

Respond with a JSON object in this format:
{{
  "symptom_summary": "Brief summary of key symptoms identified",
  "diagnosis_confidence": 45.5,
  "potential_diagnoses": ["condition1", "condition2", "condition3"],
  "diagnosis_reasoning": "Explanation of your diagnostic reasoning",
  "diagnosis_ready": false
}}

Focus on making your own independent assessment based solely on the information in the conversation history."""
        
        # Get assessment from verification model
        verification_assessment = await self.execute_structured_llm_prompt(verification_prompt, use_verification_model=True)
        
        # Check if assessment succeeded
        if "error" in verification_assessment:
            logger.error(f"Error in verification assessment: {verification_assessment['error']}")
            return (False, 0.0)
        
        # Extract confidence and diagnoses
        verification_confidence = verification_assessment.get("diagnosis_confidence", 0.0)
        
        if "potential_diagnoses" in verification_assessment:
            patient_data["verification_diagnoses"] = verification_assessment["potential_diagnoses"]
        
        # Update patient data
        patient_data["verification_diagnosis_confidence"] = verification_confidence
        
        # Log verification assessment
        logger.info(f"Verification model diagnosis confidence: {verification_confidence}%")
        logger.info(f"Verification model diagnoses: {patient_data.get('verification_diagnoses', [])}")
        
        # Determine if verification passed threshold
        verification_passed = verification_confidence >= 90.0
        
        return (True, verification_confidence)
    
    async def execute_action(self, action_name: str, user_id: str, user_message: str = "") -> str:
        """Execute a dialog action and return the response"""
        patient_data = self.get_patient_data(user_id)
        conversation_context = self._build_conversation_context(user_id)
        
        logger.info(f"Executing action: {action_name}")
        
        if action_name == "utter_greet":
            return "Hello! I'm your medical assistant. I'm here to help with your health questions."
        
        elif action_name == "utter_how_can_i_help":
            return "How can I help you today?"
        
        elif action_name == "action_assess_and_follow_up":
            """
            Core action that:
            1. Assesses symptoms and updates confidence with primary model
            2. If primary confidence > 90%, verifies with second model
            3. If both models > 90% confidence, provides diagnosis
            4. Otherwise asks follow-up questions
            """
            # If we received an answer to a previous question, record it
            if patient_data["waiting_for_answer"] and user_message:
                patient_data["waiting_for_answer"] = False
                
                # Add this to the knowledge base for the next assessment
                if "last_question" in patient_data:
                    logger.info(f"Received answer to question: {patient_data['last_question']}")
                    logger.info(f"Answer: {user_message}")
            
            # Create assessment prompt for primary model
            assessment_prompt = f"""You are a medical assistant assessing a patient's symptoms and determining appropriate next steps.

CONVERSATION HISTORY:
{conversation_context}

LATEST USER MESSAGE:
{user_message}

Your task is to:
1. Analyze all symptoms and information provided so far
2. Assess the likelihood of possible diagnoses
3. Determine your confidence level in a diagnosis (0-100%)
4. If confidence is below 90%, identify the SINGLE most information-rich follow-up question
5. If confidence is 90% or higher, indicate that a diagnosis can be considered

Respond with a JSON object in this format:
{{
  "symptom_summary": "Brief summary of key symptoms identified",
  "diagnosis_confidence": 45.5,
  "potential_diagnoses": ["condition1", "condition2", "condition3"],
  "diagnosis_reasoning": "Explanation of your diagnostic reasoning",
  "diagnosis_ready": false,
  "follow_up_question": "Single most informative follow-up question if needed",
  "question_reasoning": "Why this specific question will provide the most diagnostic value"
}}

The follow-up question should be the ONE question that would best help discriminate between potential diagnoses or confirm a specific diagnosis. Focus on maximizing information gain with each question."""
            
            # Get assessment from primary LLM
            assessment = await self.execute_structured_llm_prompt(assessment_prompt)
            
            # Check if assessment succeeded
            if "error" in assessment:
                logger.error(f"Error in symptom assessment: {assessment['error']}")
                return "I'm having trouble assessing your symptoms. Could you please tell me more about what you're experiencing?"
            
            # Update patient data with assessment
            if "symptom_summary" in assessment and assessment["symptom_summary"]:
                logger.info(f"Symptom summary: {assessment['symptom_summary']}")
            
            # Update confidence and potential diagnoses
            patient_data["primary_diagnosis_confidence"] = assessment.get("diagnosis_confidence", 0.0)
            
            if "potential_diagnoses" in assessment:
                patient_data["potential_diagnoses"] = assessment["potential_diagnoses"]
            
            # Log primary model confidence assessment
            logger.info(f"Primary model diagnosis confidence: {patient_data['primary_diagnosis_confidence']}%")
            logger.info(f"Primary model potential diagnoses: {patient_data.get('potential_diagnoses', [])}")
            
            # Check if primary model's confidence reaches threshold for verification
            primary_confidence = patient_data["primary_diagnosis_confidence"]
            verification_needed = primary_confidence >= 90.0 and not patient_data["verification_attempted"]
            
            # If primary model is confident, verify with second model
            if verification_needed:
                logger.info("Primary model confidence threshold reached - performing verification")
                
                verification_success, verification_confidence = await self.perform_verification_assessment(user_id)
                
                # Check if both models are confident
                if verification_success and verification_confidence >= 90.0:
                    # Both models have high confidence - ready for diagnosis
                    patient_data["next_intent"] = "diagnosis_ready"
                    logger.info(f"Both models have high confidence - ready for diagnosis")
                    logger.info(f"Primary: {primary_confidence}%, Verification: {verification_confidence}%")
                    return "Based on our conversation and thorough analysis, I believe I have enough information to offer some insights about your symptoms."
                else:
                    # Verification model not confident enough - continue with questions
                    logger.info(f"Verification model confidence too low: {verification_confidence}% - continuing assessment")
            
            # Check if diagnosis is ready based on previous verification
            diagnosis_ready = assessment.get("diagnosis_ready", False)
            both_models_confident = (
                patient_data["primary_diagnosis_confidence"] >= 90.0 and 
                patient_data["verification_diagnosis_confidence"] >= 90.0
            )
            
            if both_models_confident or (diagnosis_ready and patient_data["primary_diagnosis_confidence"] >= 95.0):
                # Signal state transition to diagnosis
                patient_data["next_intent"] = "diagnosis_ready"
                logger.info(f"Diagnosis ready - both models confident or primary model highly confident")
                
                # Immediately proceed with diagnosis rather than waiting for user input
                intro = "Based on our conversation and thorough analysis, I believe I have enough information to offer some insights about your symptoms. "
                
                # Generate diagnosis immediately
                diagnosis = await self.execute_action("action_provide_diagnosis", user_id, "")
                mitigations = await self.execute_action("action_suggest_mitigations", user_id, "")
                
                # Combine all parts
                return intro + diagnosis + " " + mitigations
            else:
                # Ask a single follow-up question
                next_question = assessment.get("follow_up_question", "")
                
                if not next_question:
                    # Fallback if no question provided
                    next_question = "Could you tell me more about when these symptoms started and how they've progressed?"
                
                # Store the question and mark that we're waiting for an answer
                patient_data["last_question"] = next_question
                patient_data["waiting_for_answer"] = True
                patient_data["question_count"] += 1
                
                # Also store in conversation history
                patient_data["conversation_history"].append({"role": "assistant", "content": next_question})
                
                # Track question reasoning for debugging
                question_reasoning = assessment.get("question_reasoning", "")
                if question_reasoning:
                    logger.info(f"Question reasoning: {question_reasoning}")
                
                return next_question
        
        elif action_name == "action_answer_patient_question":
            """Answer the patient's question while maintaining the diagnostic context"""
            answer_prompt = f"""You are a medical assistant. The patient has been discussing health concerns and has asked a question.

CONVERSATION HISTORY:
{conversation_context}

PATIENT QUESTION:
{user_message}

Provide a helpful, factually accurate answer that:
1. Addresses their specific question
2. Is medically sound and responsible
3. Avoids making definitive diagnoses without sufficient information
4. Is conversational and empathetic

Keep your response focused on their question while maintaining appropriate clinical caution."""
            
            answer = await self.execute_llm_prompt(answer_prompt)
            
            # Record the answer in conversation history
            patient_data["conversation_history"].append({"role": "assistant", "content": answer})
            
            return answer
        
        elif action_name == "action_return_to_assessment":
            """After answering a patient question, return to the diagnostic assessment"""
            # Create prompt for next follow-up question
            return_prompt = f"""You are a medical assistant who has just answered a patient question. Now you need to continue assessing their symptoms.

CONVERSATION HISTORY:
{conversation_context}

Based on this conversation, determine the SINGLE most important follow-up question to ask next that would:
1. Help distinguish between potential diagnoses
2. Gather critical missing information about their symptoms
3. Provide the maximum diagnostic value

The question should be direct, specific, and focused on a single aspect of their condition."""
            
            next_question = await self.execute_llm_prompt(return_prompt)
            
            # Store the question and mark that we're waiting for an answer
            patient_data["last_question"] = next_question
            patient_data["waiting_for_answer"] = True
            patient_data["question_count"] += 1
            
            # Add to conversation history
            patient_data["conversation_history"].append({"role": "assistant", "content": next_question})
            
            return next_question
        
        elif action_name == "action_provide_medical_info":
            """Provide general medical information in response to questions"""
            info_prompt = f"""You are a medical assistant providing general health information.

CONVERSATION HISTORY:
{conversation_context}

USER QUESTION/TOPIC:
{user_message}

Provide helpful, accurate medical information that:
1. Is evidence-based and factual
2. Avoids making specific diagnoses
3. Includes appropriate disclaimers about general information vs. personalized medical advice
4. Is conversational and accessible

Keep your response focused on providing valuable health information while maintaining appropriate clinical caution."""
            
            info_response = await self.execute_llm_prompt(info_prompt)
            
            # Record in conversation history
            patient_data["conversation_history"].append({"role": "assistant", "content": info_response})
            
            return info_response
        
        elif action_name == "action_provide_diagnosis":
            """Generate a diagnosis based on collected symptoms when confidence threshold is met"""
            # Prepare diagnostic context with both models' assessments
            primary_confidence = patient_data.get("primary_diagnosis_confidence", 0)
            verification_confidence = patient_data.get("verification_diagnosis_confidence", 0)
            primary_diagnoses = patient_data.get("potential_diagnoses", ["Unknown"])
            verification_diagnoses = patient_data.get("verification_diagnoses", [])
            
            # Format diagnoses lists
            primary_diagnoses_str = ", ".join(primary_diagnoses) if primary_diagnoses else "Unknown"
            verification_diagnoses_str = ", ".join(verification_diagnoses) if verification_diagnoses else "Not available"
            
            # Create diagnosis prompt with multi-model context - UPDATED FOR SIMPLER FORMAT
            diagnosis_prompt = f"""You are a medical assistant providing a preliminary assessment of symptoms.

CONVERSATION HISTORY:
{conversation_context}

DIAGNOSTIC ASSESSMENT:
Primary model confidence: {primary_confidence}%
Primary model diagnoses: {primary_diagnoses_str}
Verification model confidence: {verification_confidence}%
Verification model diagnoses: {verification_diagnoses_str}

Provide a clear, easy-to-understand assessment using these guidelines:

1. Start with a brief, simple summary of the key symptoms in 1-2 sentences
2. Explain the most likely condition(s) in everyday language, avoiding medical jargon when possible
3. Present your assessment in 2-3 concise paragraphs total
4. Use conversational language as if speaking directly to the patient
5. Include a simple statement that this is a preliminary assessment, not a definitive diagnosis
6. DO NOT use bullet points or numbered lists - write in complete paragraphs only

Remember that patients need clear, straightforward information. Focus on being helpful and informative without causing unnecessary concern."""
            
            diagnosis = await self.execute_llm_prompt(diagnosis_prompt)
            
            # Store the diagnosis
            patient_data["diagnosis"] = diagnosis
            
            # Add to conversation history
            patient_data["conversation_history"].append({"role": "assistant", "content": diagnosis})
            
            return diagnosis
        
        elif action_name == "action_suggest_mitigations":
            """Suggest appropriate self-care and next steps based on the diagnosis"""
            mitigation_prompt = f"""You are a medical assistant providing guidance after a preliminary assessment of symptoms.

CONVERSATION HISTORY:
{conversation_context}

DIAGNOSTIC ASSESSMENT:
{patient_data.get("diagnosis", "Unknown")}

Provide simple, actionable next steps for the patient in a few short paragraphs, covering:

1. Self-care measures that might help relieve symptoms
2. Clear signs that would indicate they should seek immediate medical attention
3. What type of healthcare provider would be appropriate to consult (general practitioner, specialist, etc.)
4. A brief, reassuring closing statement

Use everyday language, avoid medical jargon, and format your response as 2-3 natural paragraphs (no bullet points or numbered lists). Focus on being supportive, practical, and easy to understand."""
            
            mitigations = await self.execute_llm_prompt(mitigation_prompt)
            
            # Add to conversation history
            patient_data["conversation_history"].append({"role": "assistant", "content": mitigations})
            
            return mitigations
        
        elif action_name == "action_answer_followup_question":
            """Answer follow-up questions after providing a diagnosis"""
            followup_prompt = f"""You are a medical assistant who has just provided a diagnosis assessment to a patient. They now have a follow-up question.

CONVERSATION HISTORY:
{conversation_context}

PATIENT QUESTION:
{user_message}

THE DIAGNOSIS YOU PROVIDED:
{patient_data.get("diagnosis", "Unknown diagnosis")}

Answer their question with these guidelines:
1. Respond directly to their specific question in a conversational, empathetic manner
2. Maintain consistency with the diagnosis you've already shared
3. Provide practical, actionable information when appropriate
4. Use simple, everyday language rather than medical terminology
5. Remind them that consulting with a healthcare provider is important for proper diagnosis and treatment

Your answer should be helpful and informative while maintaining appropriate clinical caution."""
            
            followup_answer = await self.execute_llm_prompt(followup_prompt)
            
            # Add to conversation history
            patient_data["conversation_history"].append({"role": "assistant", "content": followup_answer})
            
            return followup_answer
        
        elif action_name == "utter_anything_else":
            return "Is there anything else you'd like to know about this topic?"
        
        elif action_name == "utter_goodbye":
            return "Take care and don't hesitate to return if you have more questions. Remember to seek professional medical care if your symptoms persist or worsen. Goodbye!"
        
        else:
            logger.warning(f"Unknown action: {action_name}")
            return "I'm not sure how to respond to that. Could you please rephrase your question or concern?"
    
    async def process_message(self, user_id: str, message: str) -> str:
        """Process a user message and return the response"""
        # Get user's current state and history
        current_state = self.get_user_state(user_id)
        history = self.get_chat_history(user_id)
        patient_data = self.get_patient_data(user_id)
        
        # Add user message to history
        history.add_user_message(message)
        
        # Also store in patient data backup
        patient_data["conversation_history"].append({"role": "user", "content": message})
        
        # Classify intent
        intents = await self.intent_classifier.classify_intent(message)
        top_intent = max(intents.items(), key=lambda x: x[1])[0]
        top_score = max(intents.items(), key=lambda x: x[1])[1]
        
        logger.info(f"User message: {message}")
        logger.info(f"Current state: {current_state}")
        logger.info(f"Classified intent: {top_intent} (score: {top_score:.2f})")
        
        # Check for override from assessment
        if "next_intent" in patient_data:
            top_intent = patient_data.pop("next_intent")
            logger.info(f"Overriding intent to: {top_intent} based on assessment")
        
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

async def interactive_conversation(user_message):
    """Run an interactive conversation with the medical assistant bot"""
    # Check for environment variables
    if not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"):
        print("\nWARNING: Azure OpenAI environment variables not set.")
        print("Using fallback responses instead of actual AI service.")
        print("\nTo use Azure OpenAI, please set:")
        print("  export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'")
        print("  export AZURE_OPENAI_API_KEY='your-api-key'")
        print("  export AZURE_OPENAI_DEPLOYMENT_NAME='gpt-4o'")
        print("  export VERIFICATION_OPENAI_DEPLOYMENT_NAME='o1'")
    
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
    asyncio.run(interactive_conversation())