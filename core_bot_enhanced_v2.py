import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

# Import our enhanced implementations
from llm_function_handler_v3 import LLMFunctionHandler
from diagnostic_engine_enhanced_v2 import DiagnosticEngine
from intent_classification_enhanced import IntentClassificationService

logger = logging.getLogger(__name__)

class DialogState:
    INITIAL = "initial"
    SYMPTOM_GATHERING = "symptom_gathering"
    DIAGNOSIS = "diagnosis"
    VERIFICATION = "verification"
    MITIGATION = "mitigation"
    REPORT = "report"
    ENDED = "ended"

class DialogManager:
    """Manages conversation flow and state transitions."""
    
    def __init__(self):
        self.state_data = {}
    
    def get_state(self, user_id: str) -> str:
        """Get the current state for a user."""
        return self.state_data.get(user_id, {}).get("state", DialogState.INITIAL)
    
    def set_state(self, user_id: str, state: str):
        """Set the state for a user."""
        if user_id not in self.state_data:
            self.state_data[user_id] = {}
        self.state_data[user_id]["state"] = state
    
    def get_patient_data(self, user_id: str) -> Dict[str, Any]:
        """Get patient data for a user."""
        return self.state_data.get(user_id, {}).get("patient_data", {})
    
    def update_patient_data(self, user_id: str, data: Dict[str, Any]):
        """Update patient data for a user."""
        if user_id not in self.state_data:
            self.state_data[user_id] = {}
        if "patient_data" not in self.state_data[user_id]:
            self.state_data[user_id]["patient_data"] = {}
        
        # Update patient data with new data
        self.state_data[user_id]["patient_data"].update(data)
    
    def reset_session(self, user_id: str):
        """Reset the session for a user."""
        self.state_data[user_id] = {}
        self.set_state(user_id, DialogState.INITIAL)

class MedicalAssistantBot:
    """Enhanced Medical Assistant Bot using function calling for diagnosis and follow-up."""
    
    def __init__(self):
        """Initialize the medical assistant bot."""
        self.llm_handler = LLMFunctionHandler()
        self.diagnostic_engine = DiagnosticEngine()
        self.intent_classifier = IntentClassificationService()
        self.dialog_manager = DialogManager()
        
        # Configure medical assistant settings
        self.follow_up_limit = 3
        self.diagnosis_confidence_threshold = 0.75
        self.init_bot()
    
    def init_bot(self):
        """Initialize the bot configuration."""
        try:
            # Register an additional plugin for conversation management
            conversation_plugin = self.llm_handler.kernel.create_plugin("conversation", "Medical conversation functions")
            
            # Add function for generating a medical report
            self.llm_handler.kernel.add_function_to_plugin(
                plugin=conversation_plugin,
                function=self.llm_handler.kernel.create_function(
                    plugin_name="conversation",
                    function_name="generate_report",
                    description="Generates a medical report summarizing the patient interaction",
                    prompt=self._get_report_prompt()
                )
            )
            
            logger.info("Medical Assistant Bot initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing bot: {str(e)}")
    
    def _get_report_prompt(self):
        return """Generate a concise medical report for a patient interaction.\n\n
Patient symptoms: {{$symptoms}}\n
Diagnosis: {{$diagnosis}}\n
Diagnosis confidence: {{$confidence}}\n
Diagnosis verification: {{$verification}}\n\n
Your report should include:\n
1. A summary of the presenting symptoms\n
2. The diagnosis and its confidence level\n
3. Whether the diagnosis was verified and any relevant details\n
4. General recommendations appropriate for the condition\n\n
Format your response as a professional medical summary."""
    
    async def process_message(self, user_id: str, message: str, include_diagnostics: bool = False) -> Dict[str, Any]:
        """Process a user message and generate a response.
        
        Args:
            user_id: The user's unique identifier
            message: The user's message
            include_diagnostics: Whether to include diagnostic information in the response
            
        Returns:
            Dictionary containing the bot's response and diagnostic information if requested
        """
        try:
            # Get the current state and patient data
            current_state = self.dialog_manager.get_state(user_id)
            patient_data = self.dialog_manager.get_patient_data(user_id)
            
            # Initialize response data
            response_data = {
                "response": "",
                "state": current_state
            }
            
            # Include diagnostic information if requested
            if include_diagnostics:
                response_data["diagnostics"] = {
                    "patient_data": patient_data,
                    "current_state": current_state
                }
            
            # Handle conversation based on current state
            if current_state == DialogState.INITIAL:
                # Classify intent
                intent_data = await self.intent_classifier.classify_intent(message)
                intent = intent_data.get("intent", "smallTalk")
                
                # Handle initial state based on intent
                if intent == "symptomReporting":
                    # Extract symptoms from the message
                    symptoms = await self.diagnostic_engine.extract_symptoms(message)
                    
                    # Update patient data with extracted symptoms
                    if symptoms:
                        self.dialog_manager.update_patient_data(user_id, {"symptoms": symptoms, "asked_questions": []})
                        
                        # Generate a follow-up question
                        follow_up_question = await self.diagnostic_engine.generate_follow_up_question(
                            symptoms, 
                            []
                        )
                        
                        # Update patient data with the asked question
                        patient_data = self.dialog_manager.get_patient_data(user_id)
                        asked_questions = patient_data.get("asked_questions", [])
                        asked_questions.append(follow_up_question)
                        self.dialog_manager.update_patient_data(user_id, {"asked_questions": asked_questions})
                        
                        # Transition to symptom gathering state
                        self.dialog_manager.set_state(user_id, DialogState.SYMPTOM_GATHERING)
                        
                        response_data["response"] = (f"I understand you're experiencing {', '.join(symptoms)}. "
                                                   f"{follow_up_question}")
                    else:
                        # No symptoms extracted, ask for symptoms
                        self.dialog_manager.set_state(user_id, DialogState.SYMPTOM_GATHERING)
                        response_data["response"] = "I'd like to help you with your health concerns. Could you please describe your symptoms in detail?"
                
                elif intent == "medicalInquiry":
                    # Handle medical inquiry without symptom reporting
                    multiple_intents = await self.intent_classifier.extract_multiple_intents(message)
                    extracted_entities = multiple_intents.get("extracted_entities", [])
                    
                    if extracted_entities:
                        # There are medical entities, ask if they're experiencing these symptoms
                        self.dialog_manager.update_patient_data(user_id, {"potential_symptoms": extracted_entities})
                        self.dialog_manager.set_state(user_id, DialogState.SYMPTOM_GATHERING)
                        response_data["response"] = f"I notice you mentioned {', '.join(extracted_entities)}. Are you currently experiencing these symptoms?"
                    else:
                        # General medical inquiry
                        result = await self.llm_handler.execute_chat_prompt(
                            f"Respond to this medical question briefly and professionally: {message}",
                            service_id="mini"
                        )
                        response_data["response"] = result.get("text", "I'm not sure I understand your medical question. Could you please rephrase?")
                
                elif intent == "emergency":
                    # Emergency response
                    response_data["response"] = "This sounds like a medical emergency. Please call your local emergency services (like 911) immediately instead of consulting with me. Seek professional medical help right away."
                    self.dialog_manager.set_state(user_id, DialogState.ENDED)
                
                elif intent in ["greeting", "smallTalk"]:
                    # Handle greetings and small talk
                    result = await self.llm_handler.execute_chat_prompt(
                        f"Respond to this greeting or small talk as a medical assistant: {message}",
                        service_id="mini"
                    )
                    response_data["response"] = result.get("text", "Hello! I'm your medical assistant. How can I help you today?")
                
                elif intent == "farewell" or intent == "endConversation":
                    # Handle farewells and end conversation
                    response_data["response"] = "Thank you for consulting with me. Take care, and don't hesitate to return if you have more health concerns."
                    self.dialog_manager.reset_session(user_id)
                
                else:
                    # Default response for unhandled intents
                    response_data["response"] = "I'm here to help with your health concerns. Could you tell me what symptoms you're experiencing?"
            
            elif current_state == DialogState.SYMPTOM_GATHERING:
                # Check for conversation-ending intents
                intent_data = await self.intent_classifier.classify_intent(message)
                intent = intent_data.get("intent", "symptomReporting")
                
                if intent in ["farewell", "endConversation"]:
                    # End the conversation if requested
                    response_data["response"] = "Thank you for consulting with me. Take care, and don't hesitate to return if you have more health concerns."
                    self.dialog_manager.reset_session(user_id)
                    response_data["state"] = DialogState.ENDED
                    return response_data
                
                # Extract symptoms from the response
                new_symptoms = await self.diagnostic_engine.extract_symptoms(message)
                
                # Update patient data
                patient_data = self.dialog_manager.get_patient_data(user_id)
                current_symptoms = patient_data.get("symptoms", [])
                asked_questions = patient_data.get("asked_questions", [])
                
                # Combine symptoms, avoiding duplicates
                combined_symptoms = list(set(current_symptoms + new_symptoms))
                
                # Check if we've asked enough follow-up questions
                if len(asked_questions) >= self.follow_up_limit or not new_symptoms:
                    # If we have enough info or no new symptoms, generate a diagnosis
                    self.dialog_manager.update_patient_data(user_id, {"symptoms": combined_symptoms})
                    diagnosis_data = await self.diagnostic_engine.generate_diagnosis(
                        combined_symptoms,
                        diagnosis_confidence_threshold=self.diagnosis_confidence_threshold
                    )
                    
                    # Update patient data with diagnosis
                    self.dialog_manager.update_patient_data(user_id, {"diagnosis": diagnosis_data.get("diagnosis", {})})
                    
                    # Check if high confidence diagnosis
                    if diagnosis_data.get("high_confidence", False):
                        # Move to verification state for high confidence diagnoses
                        self.dialog_manager.set_state(user_id, DialogState.VERIFICATION)
                        
                        patient_data = self.dialog_manager.get_patient_data(user_id)
                        verification_response, is_verified = await self.diagnostic_engine.verify_symptoms(patient_data)
                        
                        # Update patient data with verification result
                        self.dialog_manager.update_patient_data(user_id, {"verification": verification_response, "is_verified": is_verified})
                        
                        # Generate mitigations for the verified diagnosis
                        diagnosis_name = patient_data.get("diagnosis", {}).get("name", "Unknown")
                        mitigation_response = await self.diagnostic_engine.generate_mitigations(diagnosis_name, combined_symptoms)
                        
                        # Move to mitigation state
                        self.dialog_manager.set_state(user_id, DialogState.MITIGATION)
                        
                        # Prepare diagnosis response
                        diagnosis = patient_data.get("diagnosis", {})
                        confidence = diagnosis.get("confidence", 0.0)
                        diagnosis_name = diagnosis.get("name", "Unknown")
                        reasoning = diagnosis.get("reasoning", "")
                        
                        # Format response
                        response_data["response"] = (
                            f"Based on your symptoms ({', '.join(combined_symptoms)}), "
                            f"I believe you may have {diagnosis_name} "
                            f"(confidence: {confidence:.0%})\n\n"
                            f"{reasoning}\n\n"
                            f"{verification_response}\n\n"
                            f"Recommendations:\n{mitigation_response}"
                        )
                    else:
                        # Generate another follow-up question for low confidence diagnoses
                        if len(asked_questions) < self.follow_up_limit:
                            # Ask another follow-up question
                            follow_up_question = await self.diagnostic_engine.generate_follow_up_question(
                                combined_symptoms, 
                                asked_questions
                            )
                            
                            # Update patient data with the asked question
                            asked_questions.append(follow_up_question)
                            self.dialog_manager.update_patient_data(user_id, {
                                "symptoms": combined_symptoms,
                                "asked_questions": asked_questions
                            })
                            
                            # Remain in symptom gathering state
                            response_data["response"] = f"Thank you for that information. {follow_up_question}"
                        else:
                            # Move to diagnosis state with low confidence
                            self.dialog_manager.set_state(user_id, DialogState.DIAGNOSIS)
                            
                            # Prepare diagnosis response
                            diagnosis = patient_data.get("diagnosis", {})
                            confidence = diagnosis.get("confidence", 0.0)
                            diagnosis_name = diagnosis.get("name", "Unknown")
                            reasoning = diagnosis.get("reasoning", "")
                            
                            # Check if we have a valid diagnosis
                            if diagnosis_name and diagnosis_name.lower() != "unknown":
                                # Generate mitigations for the diagnosis
                                mitigation_response = await self.diagnostic_engine.generate_mitigations(diagnosis_name, combined_symptoms)
                                
                                # Format response with low confidence warning
                                response_data["response"] = (
                                    f"Based on the information you've provided ({', '.join(combined_symptoms)}), "
                                    f"it's possible you may have {diagnosis_name}, though I'm not entirely certain "
                                    f"(confidence: {confidence:.0%}).\n\n"
                                    f"{reasoning}\n\n"
                                    f"Some general recommendations:\n{mitigation_response}\n\n"
                                    f"Please consult with a healthcare professional for a proper examination and diagnosis."
                                )
                            else:
                                # No clear diagnosis
                                response_data["response"] = (
                                    f"After analyzing your symptoms ({', '.join(combined_symptoms)}), "
                                    f"I don't have enough information to provide a specific diagnosis. "
                                    f"The symptoms you've described could be associated with various conditions. "
                                    f"I recommend consulting with a healthcare professional who can perform a proper examination."
                                )
                else:
                    # Ask another follow-up question
                    follow_up_question = await self.diagnostic_engine.generate_follow_up_question(
                        combined_symptoms, 
                        asked_questions
                    )
                    
                    # Update patient data with new symptoms and the asked question
                    asked_questions.append(follow_up_question)
                    self.dialog_manager.update_patient_data(user_id, {
                        "symptoms": combined_symptoms,
                        "asked_questions": asked_questions
                    })
                    
                    # Remain in symptom gathering state
                    response_data["response"] = f"I see you have {', '.join(combined_symptoms)}. {follow_up_question}"
            
            elif current_state in [DialogState.DIAGNOSIS, DialogState.VERIFICATION, DialogState.MITIGATION]:
                # Generate a medical report for the patient
                patient_data = self.dialog_manager.get_patient_data(user_id)
                
                # Check if user wants a report
                intent_data = await self.intent_classifier.classify_intent(message)
                intent = intent_data.get("intent", "medicalInquiry")
                
                if intent in ["farewell", "endConversation"]:
                    # End the conversation
                    response_data["response"] = "Thank you for consulting with me. I hope the information was helpful. Take care, and don't hesitate to return if you have more health concerns."
                    self.dialog_manager.reset_session(user_id)
                    response_data["state"] = DialogState.ENDED
                    return response_data
                
                # Extract additional information and handle medical inquiries
                if intent == "medicalInquiry":
                    # Get diagnosis information
                    diagnosis = patient_data.get("diagnosis", {})
                    diagnosis_name = diagnosis.get("name", "Unknown")
                    
                    # Generate a response based on the inquiry and diagnosis context
                    result = await self.llm_handler.execute_chat_prompt(
                        f"Given a diagnosis of {diagnosis_name}, respond to this medical question professionally: {message}",
                        service_id="full"
                    )
                    response_data["response"] = result.get("text", "I'm not sure I understand your question. Could you please clarify what you'd like to know about your condition?")
                    
                elif intent == "symptomReporting":
                    # Handle additional symptom reporting after diagnosis
                    new_symptoms = await self.diagnostic_engine.extract_symptoms(message)
                    if new_symptoms:
                        # Update patient data with new symptoms
                        current_symptoms = patient_data.get("symptoms", [])
                        combined_symptoms = list(set(current_symptoms + new_symptoms))
                        self.dialog_manager.update_patient_data(user_id, {"symptoms": combined_symptoms})
                        
                        # Re-evaluate diagnosis with new symptoms
                        diagnosis_data = await self.diagnostic_engine.generate_diagnosis(
                            combined_symptoms,
                            diagnosis_confidence_threshold=self.diagnosis_confidence_threshold
                        )
                        
                        # Update patient data with new diagnosis
                        self.dialog_manager.update_patient_data(user_id, {"diagnosis": diagnosis_data.get("diagnosis", {})})
                        
                        # Generate mitigations for the diagnosis
                        diagnosis = diagnosis_data.get("diagnosis", {})
                        diagnosis_name = diagnosis.get("name", "Unknown")
                        mitigation_response = await self.diagnostic_engine.generate_mitigations(diagnosis_name, combined_symptoms)
                        
                        # Format response with updated information
                        response_data["response"] = (
                            f"I've updated your symptoms to include {', '.join(new_symptoms)}. "
                            f"Based on this new information, I believe you may have {diagnosis_name} "
                            f"(confidence: {diagnosis.get('confidence', 0.0):.0%}).\n\n"
                            f"{diagnosis.get('reasoning', '')}\n\n"
                            f"Recommendations:\n{mitigation_response}"
                        )
                    else:
                        # No new symptoms extracted
                        response_data["response"] = "I understand you're providing more information, but I'm not able to identify new symptoms. Would you like me to summarize my current assessment?"
                    
                else:
                    # Generate a medical report
                    try:
                        # Extract report data
                        symptoms = ", ".join(patient_data.get("symptoms", []))
                        diagnosis = patient_data.get("diagnosis", {})
                        diagnosis_name = diagnosis.get("name", "Unknown")
                        confidence = str(diagnosis.get("confidence", 0.0))
                        verification = patient_data.get("verification", "Verification not performed")
                        
                        # Generate the report
                        result = await self.llm_handler.invoke_function(
                            "conversation.generate_report",
                            {
                                "symptoms": symptoms,
                                "diagnosis": diagnosis_name,
                                "confidence": confidence,
                                "verification": verification
                            },
                            service_id="full"
                        )
                        
                        # Set state to report
                        self.dialog_manager.set_state(user_id, DialogState.REPORT)
                        response_data["response"] = f"Here's a summary of our consultation:\n\n{result.get('text', '')}"
                        
                    except Exception as e:
                        logger.error(f"Error generating report: {str(e)}")
                        response_data["response"] = "I'm having trouble generating a summary at the moment. Is there something specific about your condition you'd like me to explain?"
            
            elif current_state == DialogState.REPORT:
                # Check if user wants to end the conversation
                intent_data = await self.intent_classifier.classify_intent(message)
                intent = intent_data.get("intent", "medicalInquiry")
                
                if intent in ["farewell", "endConversation"]:
                    # End the conversation
                    response_data["response"] = "Thank you for consulting with me. I hope the information was helpful. Take care, and don't hesitate to return if you have more health concerns."
                    self.dialog_manager.reset_session(user_id)
                    response_data["state"] = DialogState.ENDED
                else:
                    # Handle follow-up inquiries after report
                    result = await self.llm_handler.execute_chat_prompt(
                        f"After providing a medical report to a patient, respond to this follow-up: {message}",
                        service_id="mini"
                    )
                    response_data["response"] = result.get("text", "Is there anything else about your condition you'd like me to explain?")
            
            elif current_state == DialogState.ENDED:
                # New conversation after previous one ended
                self.dialog_manager.reset_session(user_id)
                
                # Process the message as a new conversation
                return await self.process_message(user_id, message, include_diagnostics)
            
            # Update state in response data
            response_data["state"] = self.dialog_manager.get_state(user_id)
            return response_data
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "response": "I apologize, but I encountered an error processing your message. Could you please try again?",
                "state": self.dialog_manager.get_state(user_id)
            }

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create an instance of the bot
    bot = MedicalAssistantBot()
    
    # Simple CLI for testing
    import asyncio
    
    async def main():
        print("Medical Assistant Bot initialized. Type 'exit' to quit.")
        user_id = "test_user"
        
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
                
            result = await bot.process_message(user_id, user_input, include_diagnostics=True)
            print(f"Bot: {result['response']}")
            print(f"State: {result['state']}")
    
    # Run the main function
    asyncio.run(main())
