import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple

# Import our enhanced LLM function handler
from llm_function_handler_v3 import LLMFunctionHandler

logger = logging.getLogger(__name__)

class DiagnosticEngine:
    """
    Enhanced diagnostic engine that uses function calling for medical diagnostics,
    follow-up questioning, and verification.
    """
    
    def __init__(self, max_follow_up_questions: int = 5):
        """Initialize the diagnostic engine.
        
        Args:
            max_follow_up_questions: Maximum number of follow-up questions to ask
        """
        self.llm_handler = LLMFunctionHandler()
        self.max_follow_up_questions = max_follow_up_questions
    
    async def extract_symptoms(self, message: str) -> List[str]:
        """Extract symptoms from a user message.
        
        Args:
            message: The user's message
            
        Returns:
            List of extracted symptoms
        """
        try:
            # Call the symptom extraction function
            result = await self.llm_handler.invoke_function(
                "medical.extract_symptoms",
                {"message": message}
            )
            
            response_text = result.get("text", "[]")
            
            # Extract symptoms array from the response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                symptoms = json.loads(json_match.group(0))
                logger.info(f"Extracted symptoms: {symptoms}")
                return symptoms
            else:
                logger.warning(f"No symptoms array found in extraction response: '{response_text}'")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting symptoms: {str(e)}")
            return []
    
    async def generate_diagnosis(self, symptoms: List[str], 
                              diagnosis_confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """Generate a diagnosis based on symptoms.
        
        Args:
            symptoms: List of patient symptoms
            diagnosis_confidence_threshold: Confidence threshold for diagnosis
            
        Returns:
            Dictionary containing diagnosis information
        """
        try:
            # Format the symptoms for the function call
            symptoms_text = ", ".join(symptoms) if symptoms else "No symptoms provided"
            
            # Call the diagnosis generation function
            result = await self.llm_handler.invoke_function(
                "medical.generate_diagnosis",
                {"symptoms": symptoms_text},
                service_id="full"  # Use the full model for better diagnoses
            )
            
            response_text = result.get("text", "")
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                diagnosis_data = json.loads(json_match.group(0))
                
                # Log diagnosis results
                primary_diagnosis = diagnosis_data.get("diagnosis", {})
                diagnosis_name = primary_diagnosis.get("name")
                confidence = primary_diagnosis.get("confidence", 0.0)
                logger.info(f"Generated diagnosis: {diagnosis_name} with confidence {confidence}")
                
                # Check if diagnosis meets confidence threshold
                if diagnosis_name and confidence >= diagnosis_confidence_threshold:
                    diagnosis_data["high_confidence"] = True
                else:
                    diagnosis_data["high_confidence"] = False
                    
                return diagnosis_data
            else:
                logger.warning(f"No valid diagnosis JSON found in response: '{response_text}'")
                return {"diagnosis": {"name": None, "confidence": 0.0}, "high_confidence": False}
                
        except Exception as e:
            logger.error(f"Error generating diagnosis: {str(e)}")
            return {"diagnosis": {"name": None, "confidence": 0.0}, "high_confidence": False}
    
    async def generate_follow_up_question(self, symptoms: List[str], asked_questions: List[str]) -> str:
        """Generate a follow-up question to gather more symptom information.
        
        Args:
            symptoms: Current list of patient symptoms
            asked_questions: Previously asked questions
            
        Returns:
            Follow-up question text
        """
        try:
            # Format the inputs for the function call
            symptoms_text = ", ".join(symptoms) if symptoms else "No symptoms provided"
            asked_questions_text = ". ".join(asked_questions) if asked_questions else "No previous questions"
            
            # Call the follow-up question generation function
            result = await self.llm_handler.invoke_function(
                "medical.generate_followup_question",
                {
                    "symptoms": symptoms_text,
                    "asked_questions": asked_questions_text
                },
                service_id="mini"  # Use the smaller model for faster responses
            )
            
            # Extract the question from the response
            question = result.get("text", "").strip()
            
            # Remove any quotes or special characters that might be in the response
            question = question.strip('"')
            logger.info(f"Generated follow-up question: {question}")
            
            return question
                
        except Exception as e:
            logger.error(f"Error generating follow-up question: {str(e)}")
            return "Can you tell me more about your symptoms?"
    
    async def verify_symptoms(self, patient_data: Dict[str, Any]) -> Tuple[str, bool]:
        """Verify symptoms with the O1 model for high-confidence verification.
        
        Args:
            patient_data: Dictionary containing patient information including symptoms and diagnosis
            
        Returns:
            Tuple containing (verification_response, is_verified)
        """
        try:
            # Extract relevant data
            symptoms = patient_data.get("symptoms", [])
            diagnosis = patient_data.get("diagnosis", {})
            diagnosis_name = diagnosis.get("name", "Unknown")
            confidence = diagnosis.get("confidence", 0.0)
            
            # Only use the verifier if we have a valid diagnosis with high confidence
            if not diagnosis_name or diagnosis_name == "Unknown" or confidence < 0.75:
                return "Cannot verify diagnosis with insufficient confidence.", False
            
            # Format the symptoms for verification
            symptoms_text = ", ".join(symptoms) if symptoms else "No symptoms provided"
            
            # Call the verification function with the O1 verifier model
            result = await self.llm_handler.invoke_function(
                "medical.verify_high_confidence_diagnosis",
                {
                    "symptoms": symptoms_text,
                    "diagnosis": diagnosis_name,
                    "confidence": str(confidence)
                },
                service_id="verifier"  # Use the medical verifier model (O1)
            )
            
            response_text = result.get("text", "")
            
            # Extract JSON verification result
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                verification_data = json.loads(json_match.group(0))
                verification = verification_data.get("verification", "disagree")
                
                # Check verification result
                is_verified = verification.lower() == "agree"
                
                # Construct verification response
                if is_verified:
                    verification_response = f"Verification complete: The diagnosis of {diagnosis_name} is confirmed."
                    if "notes" in verification_data:
                        verification_response += f" {verification_data['notes']}"
                else:
                    refined_diagnosis = verification_data.get("diagnosis_name", "")
                    verification_response = f"Verification complete: The initial diagnosis of {diagnosis_name} needs refinement."
                    if refined_diagnosis:
                        verification_response += f" A more accurate diagnosis may be {refined_diagnosis}."
                    if "notes" in verification_data:
                        verification_response += f" {verification_data['notes']}"
                
                logger.info(f"Verification result: {verification} for diagnosis {diagnosis_name}")
                return verification_response, is_verified
            else:
                logger.warning(f"No valid verification JSON found in response: '{response_text}'")
                return "Unable to verify the diagnosis at this time.", False
                
        except Exception as e:
            logger.error(f"Error verifying symptoms: {str(e)}")
            return f"Error in verification process: {str(e)}", False
    
    async def generate_mitigations(self, diagnosis: str, symptoms: List[str]) -> str:
        """Generate mitigations and management steps for a diagnosed condition.
        
        Args:
            diagnosis: The name of the diagnosed condition
            symptoms: List of patient symptoms
            
        Returns:
            Mitigation recommendations text
        """
        try:
            if not diagnosis or diagnosis.lower() == "unknown":
                return "Without a clear diagnosis, I cannot provide specific recommendations."
            
            # Format the symptoms for the function call
            symptoms_text = ", ".join(symptoms) if symptoms else "No symptoms provided"
            
            # Call the mitigation suggestion function
            result = await self.llm_handler.invoke_function(
                "medical.suggest_mitigations",
                {
                    "diagnosis": diagnosis,
                    "symptoms": symptoms_text
                },
                service_id="full"  # Use the full model for better recommendations
            )
            
            mitigation_response = result.get("text", "")
            return mitigation_response
                
        except Exception as e:
            logger.error(f"Error generating mitigations: {str(e)}")
            return "I'm unable to provide specific recommendations at this time."
    
    async def calculate_diagnosis_confidence(self, patient_data: Dict[str, Any]) -> float:
        """Calculate the confidence level for a diagnosis based on patient data.
        
        Args:
            patient_data: Dictionary containing patient information
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Extract diagnosis from patient data
        diagnosis = patient_data.get("diagnosis", {})
        
        # Return the confidence from the diagnosis
        return diagnosis.get("confidence", 0.0)
