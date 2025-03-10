"""
Diagnostic engine for medical assistant bot
"""
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class DiagnosticEngine:
    """Handles medical diagnosis and symptom analysis"""
    
    def __init__(self, llm_handler, medical_plugin=None):
        """
        Initialize the diagnostic engine
        
        Args:
            llm_handler: Handler for LLM interactions
            medical_plugin: Optional medical knowledge plugin for fallbacks
        """
        self.llm_handler = llm_handler
        self.medical_plugin = medical_plugin
    
    def get_patient_data(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure patient data structure is properly initialized
        
        Args:
            user_data: The user's data dictionary
            
        Returns:
            Patient data dictionary with all required fields
        """
        # Make sure we have the right structure
        if "patient_data" not in user_data:
            user_data["patient_data"] = {
                "symptoms": [],
                "demographics": {},
                "asked_questions": [],
                "diagnosis": None,
                "mitigations": [],
                "diagnosis_confidence": 0.0
            }
        
        return user_data["patient_data"]
    
    def add_symptom(self, patient_data: Dict[str, Any], symptom: str) -> None:
        """
        Add a symptom to patient data, or update existing symptoms with new details
        
        Args:
            patient_data: The patient data dictionary
            symptom: The symptom to add
        """
        if not symptom:
            return
            
        # First check if this is new information to be combined with existing symptoms
        symptoms = patient_data.get("symptoms", [])
        
        # If we already have symptoms and this adds details, combine them
        if symptoms and symptom not in symptoms:
            # Check for key symptom terms to determine if this is additional detail
            # about an existing symptom or a completely new symptom
            existing_keywords = []
            for existing in symptoms:
                # Extract key terms from existing symptoms
                words = existing.lower().split()
                existing_keywords.extend([w for w in words if len(w) > 3])
            
            # If the new symptom contains keywords from existing symptoms
            # treat it as additional details
            symptom_words = symptom.lower().split()
            has_overlap = any(word in existing_keywords for word in symptom_words if len(word) > 3)
            
            if has_overlap:
                # This is likely additional detail about the same symptoms
                # Replace the most relevant symptom with a combined version
                most_similar = None
                highest_similarity = 0
                
                for i, existing in enumerate(symptoms):
                    # Simple token overlap similarity
                    existing_tokens = set(existing.lower().split())
                    new_tokens = set(symptom.lower().split())
                    overlap = len(existing_tokens.intersection(new_tokens))
                    similarity = overlap / len(existing_tokens.union(new_tokens))
                    
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        most_similar = i
                
                if most_similar is not None and highest_similarity > 0.2:
                    # Combine the existing symptom with the new details
                    combined = f"{symptoms[most_similar]} ({symptom})"
                    symptoms[most_similar] = combined
                    patient_data["symptoms"] = symptoms
                    logger.info(f"Updated symptom with details: {combined}")
                    return
        
        # If it's a new symptom or we don't have any symptoms yet, add it
        if symptom not in symptoms:
            symptoms.append(symptom)
            patient_data["symptoms"] = symptoms
            logger.info(f"Added new symptom: {symptom}")
    
    def get_symptoms_text(self, patient_data: Dict[str, Any]) -> str:
        """
        Get all symptoms as a comma-separated string
        
        Args:
            patient_data: The patient data dictionary
            
        Returns:
            String of symptoms
        """
        symptoms = patient_data.get("symptoms", [])
        if not symptoms:
            return "unknown symptoms"
        return ", ".join(symptoms)
    
    def get_asked_questions_text(self, patient_data: Dict[str, Any]) -> str:
        """
        Get all asked questions as a comma-separated string
        
        Args:
            patient_data: The patient data dictionary
            
        Returns:
            String of asked questions
        """
        questions = patient_data.get("asked_questions", [])
        if not questions:
            return ""
        return ", ".join(questions)

    async def calculate_diagnosis_confidence(self, symptoms: str) -> Dict[str, Any]:
        """
        Calculate confidence in diagnosis based on provided symptoms using LLM.
        
        Args:
            symptoms: The symptoms text to analyze
            
        Returns:
            Dictionary with confidence score and reasoning
        """
        if not self.llm_handler.is_available():  # Check the LLM handler, not self
            return {
                "confidence": 0.3,
                "reasoning": "Limited confidence due to unavailable LLM service"
            }
        
        try:
            prompt = f"""Analyze these symptoms to assess how confident a medical professional would be in providing a diagnosis:

    Symptoms: {symptoms}

    Consider factors like:
    - Specificity of symptoms described
    - Whether duration and severity are mentioned
    - Presence of characteristic patterns
    - Whether symptoms could match multiple conditions

    Respond with a confidence score between 0.0 and 1.0 where:
    - 0.1-0.3: Very low confidence (vague symptoms, minimal details)
    - 0.3-0.5: Low confidence (some details but ambiguous presentation)
    - 0.5-0.7: Moderate confidence (specific symptoms with some context)
    - 0.7-0.9: High confidence (detailed, specific symptoms with context)
    - 0.9+: Very high confidence (detailed symptoms forming a clear pattern)

    Return a JSON object:
    {{
    "confidence": 0.7,
    "reasoning": "Brief explanation of the confidence score"
    }}
    """
            # Use mini model to save costs
            response_data = await self.llm_handler.execute_prompt(prompt, use_full_model=False, temperature=0.3)  # Call the LLMHandler's method
            response_text = response_data.get("text", "")
            
            # Extract the JSON
            import re
            import json
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    confidence = float(result.get("confidence", 0.3))
                    # Ensure confidence is within valid range
                    confidence = max(0.0, min(1.0, confidence))
                    
                    return {
                        "confidence": confidence,
                        "reasoning": result.get("reasoning", "No reasoning provided")
                    }
                except (json.JSONDecodeError, ValueError):
                    return {
                        "confidence": 0.3,
                        "reasoning": "Error parsing confidence calculation"
                    }
            
            return {
                "confidence": 0.3,
                "reasoning": "Could not calculate confidence from LLM response"
            }
        
        except Exception as e:
            logger.error(f"Error calculating diagnosis confidence: {str(e)}")
            return {
                "confidence": 0.2,
                "reasoning": f"Error in confidence calculation: {str(e)}"
            }
    

    async def update_diagnosis_confidence(self, patient_data: Dict[str, Any]) -> None:
        """
        Update diagnosis confidence for patient data
        
        Args:
            patient_data: The patient data dictionary
        """
        symptoms = self.get_symptoms_text(patient_data)
        
        # If we have no symptoms or just blank entries, set confidence to 0
        if not symptoms or symptoms == "unknown symptoms":
            patient_data["diagnosis_confidence"] = 0.0
            patient_data["confidence_reasoning"] = "Insufficient information"
            return
        
        # Count how many questions we've asked
        question_count = len(patient_data.get("asked_questions", []))
        
        # If we have symptoms but haven't asked any follow-up questions yet,
        # set a minimum baseline confidence based on symptom count
        if question_count == 0:
            symptom_count = len(patient_data.get("symptoms", []))
            # Start with a low baseline confidence
            baseline_confidence = min(0.1 * symptom_count, 0.4)
            patient_data["diagnosis_confidence"] = baseline_confidence
            patient_data["confidence_reasoning"] = "Initial symptoms provided, awaiting more details"
            logger.info(f"Set baseline confidence: {baseline_confidence:.2f}")
            return
        
        # Otherwise, use the LLM to calculate confidence
        confidence_data = await self.calculate_diagnosis_confidence(symptoms)  # Call local method instead
        
        # Extract confidence and reasoning
        confidence = confidence_data.get("confidence", 0.0)
        reasoning = confidence_data.get("reasoning", "No reasoning available")
        
        patient_data["diagnosis_confidence"] = confidence
        patient_data["confidence_reasoning"] = reasoning
        
        logger.info(f"Updated diagnosis confidence: {confidence:.2f}")
        logger.info(f"Confidence reasoning: {reasoning}")
    
    async def generate_followup_question(self, patient_data: Dict[str, Any]) -> str:
        """
        Generate a follow-up question based on collected symptoms
        
        Args:
            patient_data: The patient data dictionary
            
        Returns:
            A follow-up question
        """
        # Get current symptoms as a string
        symptoms = self.get_symptoms_text(patient_data)
        
        # Get previously asked questions
        asked = self.get_asked_questions_text(patient_data)
        
        logger.info(f"Generating follow-up question about symptoms: {symptoms}")
        
        if self.llm_handler.is_available():
            try:
                # Create a prompt for follow-up questions
                prompt = f"""As a medical assistant, I need to ask follow-up questions about the patient's symptoms.
Current symptoms: {symptoms}
Medical history: N/A
Previously asked questions: {asked}

Generate a relevant follow-up question to better understand these symptoms."""

                # Use mini model for follow-up questions to save cost
                response_data = await self.llm_handler.execute_prompt(prompt, use_full_model=False)
                response = response_data.get("text", "")
                
                # Add this question to the list of asked questions
                if response and "Error" not in response:
                    patient_data["asked_questions"].append(response)
                    
                return response
            except Exception as e:
                logger.error(f"Error generating follow-up questions with LLM: {str(e)}")
        
        # Use the plugin method or fallback
        if self.medical_plugin:
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
        
        # Last resort fallback
        fallback_response = "Can you tell me more about your symptoms? When did they start and have they changed over time?"
        logger.info(f"Using fallback response for follow-up questions")
        patient_data["asked_questions"].append(fallback_response)
        return fallback_response
    
    async def verify_symptoms(self, patient_data: Dict[str, Any]) -> str:
        """
        Verify collected symptoms with the patient before diagnosis
        
        Args:
            patient_data: The patient data dictionary
            
        Returns:
            Verification response
        """
        symptoms = self.get_symptoms_text(patient_data)
        confidence = patient_data.get("diagnosis_confidence", 0.0)
        
        # Determine if we should use the verifier model (for high confidence cases)
        use_verifier = confidence >= 0.9 and self.llm_handler.is_verifier_model_available()
        
        if self.llm_handler.is_available() and (self.llm_handler.is_full_model_available() or use_verifier):
            try:
                prompt = f"""Based on our conversation, I understand you're experiencing these symptoms:
    {symptoms}

    I want to make sure I understand correctly before providing my assessment.
    Please review these symptoms and let me know if this accurately represents what you're experiencing, 
    or if there's anything I've missed or misunderstood.

    Format your response as a direct question to the patient that summarizes the symptoms and asks for confirmation."""
                
                if use_verifier:
                    logger.info(f"Using verifier model for high-confidence symptom verification ({confidence:.2f})")
                    response_data = await self.llm_handler.execute_prompt(prompt, use_verifier_model=True)
                else:
                    # Use full model for normal verification
                    response_data = await self.llm_handler.execute_prompt(prompt, use_full_model=True)
                
                # Store which model was used for verification
                patient_data["verification_model"] = response_data.get("model", "unknown")
                
                return response_data.get("text", "")
            except Exception as e:
                logger.error(f"Error verifying symptoms with LLM: {str(e)}")
        
        # Simple fallback
        return f"I understand you're experiencing: {symptoms}. Is that correct, or would you like to add any other symptoms before I provide my assessment?"
    
    async def suggest_mitigations(self, patient_data: Dict[str, Any]) -> str:
        """
        Suggest mitigations based on diagnosis
        
        Args:
            patient_data: The patient data dictionary
            
        Returns:
            Mitigation suggestions
        """
        diagnosis = patient_data.get("diagnosis", "")
        symptoms = self.get_symptoms_text(patient_data)
        
        if not diagnosis or diagnosis == "unknown":
            return "Without a clear understanding of your condition, I'd recommend rest, staying hydrated, and consulting with a healthcare provider if your symptoms persist or worsen."
        
        if self.llm_handler.is_available():
            try:
                prompt = f"""Based on this potential diagnosis: 
"{diagnosis}"

And these symptoms:
"{symptoms}"

Suggest appropriate general care measures or mitigations that might help the patient.
Focus on evidence-based, conservative recommendations.
Emphasize these are general suggestions and not a replacement for professional medical care."""

                # Use full model for mitigations if available
                response_data = await self.llm_handler.execute_prompt(prompt, use_full_model=True)
                
                return response_data.get("text", "")
            except Exception as e:
                logger.error(f"Error generating mitigations: {str(e)}")
        
        # Fallback
        return "Here are some general steps you might consider: rest, stay hydrated, and monitor your symptoms. If they worsen, please consult with your healthcare provider."