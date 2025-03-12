import logging
import time
import json
import re
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def current_utc_timestamp() -> str:
    """
    Simple helper to generate a UTC timestamp string, e.g. 2025-03-10T09:00:00Z
    """
    import datetime
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

class DiagnosticEngine:
    """Handles medical diagnosis and symptom analysis using function calling."""
    
    def __init__(self, llm_function_handler, medical_plugin=None):
        """
        Initialize the diagnostic engine
        
        Args:
            llm_function_handler: Handler for LLM function calling interactions
            medical_plugin: Optional medical knowledge plugin for specialized knowledge
        """
        self.llm_handler = llm_function_handler
        self.medical_plugin = medical_plugin
    
    def get_patient_data(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure patient data structure is properly initialized.

        Args:
            user_data: The user's data dictionary

        Returns:
            Patient data dictionary with all required fields
        """
        if "patient_data" not in user_data:
            user_data["patient_data"] = {
                "patient_id": "",
                "demographics": {
                    "age": 0,
                    "gender": "",
                    "weight": 0.0,
                    "height": 0.0,
                    "other_demographics": {}
                },
                "symptoms": [],
                "medical_history": [],
                "asked_questions": [],
                "diagnosis": {
                    "name": None,
                    "confidence": 0.0
                },
                "mitigations": []
            }
        
        return user_data["patient_data"]
    
    def add_symptom(self, patient_data: Dict[str, Any], symptom: str) -> None:
        """
        Add or refine a symptom in the patient data structure.
        """
        if not symptom:
            logger.warning("Attempted to add empty symptom")
            return
        
        # Normalize symptom text
        symptom = symptom.strip().lower()
        
        # Add to symptoms list if it doesn't already exist
        symptoms = patient_data.get("symptoms", [])
        if symptom not in symptoms:
            symptoms.append(symptom)
            patient_data["symptoms"] = symptoms
            logger.info(f"Added symptom: '{symptom}' to patient data")
        else:
            logger.info(f"Symptom '{symptom}' already exists in patient data")
    
    def get_symptoms_text(self, patient_data: Dict[str, Any]) -> str:
        """
        Convert patient symptoms to a concise text format for prompts.
        """
        symptoms = patient_data.get("symptoms", [])
        if not symptoms:
            return "unknown symptoms"
        
        return ", ".join(symptoms)
    
    async def extract_symptoms_from_message(self, message: str) -> List[str]:
        """
        Extract symptoms from a user message using the symptom extraction function.
        """
        try:
            # Call the symptom extraction function
            result = await self.llm_handler.invoke_function(
                "medical.extract_symptoms",
                {"message": message},
                service_id="mini"
            )
            
            # Parse the result to extract symptoms array
            response_text = result.get("text", "[]")
            
            # Extract JSON array from response if needed
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
                
            # Parse the JSON array
            symptoms = json.loads(response_text)
            
            # Ensure list format
            if isinstance(symptoms, list):
                return symptoms
            else:
                logger.warning(f"Extracted symptoms not in expected list format: {symptoms}")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting symptoms from message: {str(e)}")
            return []
            
    async def calculate_diagnosis_confidence(self, symptoms_text: str) -> Dict[str, Any]:
        """
        Calculate confidence in diagnosis based on current symptoms.
        Uses a semantic function to evaluate diagnostic confidence.
        """
        try:
            # Generate a preliminary diagnosis with confidence
            result = await self.llm_handler.invoke_function(
                "medical.generate_diagnosis",
                {"symptoms": symptoms_text},
                service_id="full"
            )
            
            # Parse the result to extract diagnosis data
            response_text = result.get("text", "{}")
            
            # Extract JSON from response if needed
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
                
            # Parse the JSON and extract confidence information
            diagnosis_data = json.loads(response_text)
            diagnosis = diagnosis_data.get("diagnosis", {})
            
            return {
                "confidence": diagnosis.get("confidence", 0.0),
                "reasoning": diagnosis.get("reasoning", "")
            }
            
        except Exception as e:
            logger.error(f"Error calculating diagnosis confidence: {str(e)}")
            return {"confidence": 0.2, "reasoning": f"Exception: {e}"}
    
    async def update_diagnosis_confidence(self, patient_data: Dict[str, Any]) -> None:
        """
        Update the patient_data's diagnosis confidence based on current symptoms
        and how many follow-up questions were asked/answered.
        """
        # Ensure diagnosis structure exists
        if "diagnosis" not in patient_data:
            logger.info("Initializing diagnosis structure in patient_data")
            patient_data["diagnosis"] = {"confidence": 0.0, "name": None}
        
        symptoms_str = self.get_symptoms_text(patient_data)
        logger.info(f"Symptoms text for confidence calculation: '{symptoms_str}'")
        
        if not symptoms_str or symptoms_str == "unknown symptoms":
            # No real symptoms => confidence = 0
            logger.warning("No valid symptoms found for diagnosis confidence calculation")
            patient_data["diagnosis"]["confidence"] = 0.0
            patient_data["confidence_reasoning"] = "No valid symptoms"
            return
        
        # If no follow-up questions asked, do a baseline calculation
        asked_questions = patient_data.get("asked_questions", [])
        question_count = len(asked_questions)
        
        if question_count == 0:
            # baseline confidence based on number of symptoms
            scount = len(patient_data["symptoms"])
            baseline_conf = min(0.1 * scount, 0.4)
            patient_data["diagnosis"]["confidence"] = baseline_conf
            patient_data["confidence_reasoning"] = f"Initial baseline confidence based on {scount} symptoms"
            logger.info(f"Baseline confidence set to {baseline_conf:.2f} based on {scount} symptoms")
            return
        
        # Otherwise, call the LLM to compute confidence using function calling
        logger.info(f"Calculating diagnosis confidence for symptoms: '{symptoms_str}'")
        conf_data = await self.calculate_diagnosis_confidence(symptoms_str)
        confidence = conf_data.get("confidence", 0.0)
        reasoning = conf_data.get("reasoning", "")
        
        patient_data["diagnosis"]["confidence"] = confidence
        patient_data["confidence_reasoning"] = reasoning
        
        # If we have diagnosis data, also update the name
        if "name" in conf_data and conf_data["name"]:
            patient_data["diagnosis"]["name"] = conf_data["name"]
            
        logger.info(f"Diagnosis confidence updated to {confidence:.2f} with reasoning: {reasoning}")
    
    async def generate_followup_question(self, patient_data: Dict[str, Any]) -> str:
        """
        Generate a follow-up question about the user's symptoms using function calling.
        The question is appended to asked_questions in 'patient_data'.
        Uses semantic functions to generate high-value questions.
        """
        symptoms_str = self.get_symptoms_text(patient_data)
        
        # Create a simple textual representation of asked questions
        asked_list = patient_data.get("asked_questions", [])
        asked_str = ", ".join(q["question"] for q in asked_list if isinstance(q, dict))
        
        logger.info(f"Generating follow-up question for: {symptoms_str}")
        
        try:
            # Call the follow-up question function
            result = await self.llm_handler.invoke_function(
                "medical.generate_followup_question",
                {
                    "symptoms": symptoms_str,
                    "asked_questions": asked_str
                },
                service_id="full"  # Use the full model for better follow-up questions
            )
            
            question_text = result.get("text", "").strip()
            if not question_text:
                question_text = "Can you tell me more about your symptoms?"
                
        except Exception as e:
            logger.error(f"Error generating follow-up question: {str(e)}")
            question_text = "Can you tell me more about your symptoms?"
        
        # Record it in asked_questions with timestamps and mark as symptom-related
        asked_dict = {
            "question": question_text,
            "answer": "",
            "timestamp_asked": current_utc_timestamp(),
            "timestamp_answered": "",
            "is_symptom_related": True,
            "is_answered": False
        }
        asked_list.append(asked_dict)
        
        # Save back to patient data
        patient_data["asked_questions"] = asked_list
        
        return question_text
    
    async def verify_symptoms(self, patient_data: Dict[str, Any]) -> str:
        """
        Verify symptoms using semantic function calls to the verification model (O1).
        
        This method handles two cases:
        1. High confidence (>=0.85): Verify if the diagnosis is correct
        2. Low confidence (<0.85): Explain why symptoms don't fit a known condition 
           and recommend medical consultation
        """
        symptoms_str = self.get_symptoms_text(patient_data)
        diagnosis_data = patient_data.get("diagnosis", {})
        confidence = float(diagnosis_data.get("confidence", 0.0))
        diagnosis_name = diagnosis_data.get("name", "Unknown")
        verification_trigger = patient_data.get("verification_info", {}).get("trigger_reason", "unknown")
        
        try:
            # Record which model is being used for verification
            patient_data["verification_model"] = "O1/medical-verification"
            
            if verification_trigger == "high_confidence" or confidence >= 0.85:
                # High confidence case - verify the diagnosis
                logger.info(f"High confidence verification ({confidence:.2f} >= 0.85), using verifier function")
                
                # Call the high confidence verification function
                result = await self.llm_handler.invoke_function(
                    "medical.verify_high_confidence_diagnosis",
                    {
                        "symptoms": symptoms_str,
                        "diagnosis": diagnosis_name,
                        "confidence": str(confidence)
                    },
                    service_id="verifier"  # Use the O1 model
                )
                
                verification_text = result.get("text", "")
                
                # Parse JSON response
                json_match = re.search(r'\{.*?\}', verification_text, re.DOTALL)
                if json_match:
                    verification_data = json.loads(json_match.group(0))
                    verification = verification_data.get("verification", "disagree")
                    
                    if verification.lower() == "agree":
                        # Verifier agrees - can finalize diagnosis
                        diag_name = verification_data.get("diagnosis_name", diagnosis_name)
                        diagnosis_data["name"] = diag_name
                        patient_data["diagnosis"] = diagnosis_data
                        
                        # Mark for immediate report generation
                        patient_data["ready_for_report"] = True
                        patient_data["verification_complete"] = True
                        
                        return f"I've confirmed your symptoms likely represent: {diag_name}. Based on this assessment, I'll provide you with a medical report."
                    else:
                        # Verifier disagrees - need more information
                        return f"I need a bit more information to be certain. Could you please clarify any missing or incorrect symptom details?"
                else:
                    # Couldn't parse verifier response
                    logger.warning("Could not parse JSON from verifier response")
                    return f"I need to verify your symptoms further. Can you confirm if these accurately describe your condition: {symptoms_str}?"
            
            else:  # Low confidence case
                # Low confidence case - explain why symptoms don't fit a known condition
                logger.info(f"Low confidence verification ({confidence:.2f} < 0.85), using explanation function")
                
                # Call the low confidence explanation function
                result = await self.llm_handler.invoke_function(
                    "medical.explain_low_confidence",
                    {
                        "symptoms": symptoms_str,
                        "confidence": str(confidence)
                    },
                    service_id="verifier"  # Use the O1 model
                )
                
                explanation = result.get("text", "")
                
                # Store the explanation in patient data
                if "verification_info" not in patient_data:
                    patient_data["verification_info"] = {}
                patient_data["verification_info"]["low_confidence_explanation"] = explanation
                
                # Add a referral note to the patient data
                patient_data["referral_needed"] = True
                
                return explanation
                
        except Exception as e:
            logger.error(f"Error during verification process: {str(e)}")
            return f"An error occurred while analyzing your symptoms. Let's clarify: Do these symptoms accurately describe your condition: {symptoms_str}?"
    
    async def generate_diagnosis(self, patient_data: Dict[str, Any]) -> str:
        """
        Generate a comprehensive diagnosis report based on patient symptoms.
        Uses function calling for consistent, structured output.
        """
        symptoms_str = self.get_symptoms_text(patient_data)
        diagnosis_data = patient_data.get("diagnosis", {})
        diagnosis_name = diagnosis_data.get("name", "Unknown")
        confidence = float(diagnosis_data.get("confidence", 0.0))
        
        try:
            # Use generate_diagnosis function to create a structured diagnosis
            result = await self.llm_handler.invoke_function(
                "medical.generate_diagnosis",
                {"symptoms": symptoms_str},
                service_id="full"  # Use the full model for comprehensive diagnosis
            )
            
            response_text = result.get("text", "")
            
            # Extract JSON from the response if needed
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if json_match:
                diagnosis_data = json.loads(json_match.group(0))
                
                # Update the patient's diagnosis information
                main_diagnosis = diagnosis_data.get("diagnosis", {})
                patient_data["diagnosis"] = {
                    "name": main_diagnosis.get("name", diagnosis_name),
                    "confidence": main_diagnosis.get("confidence", confidence),
                    "reasoning": main_diagnosis.get("reasoning", "")
                }
                
                # Update differential diagnoses
                patient_data["differential_diagnoses"] = diagnosis_data.get("differential_diagnoses", [])
                
                # Format a user-friendly report
                main_name = patient_data["diagnosis"]["name"]
                reasoning = patient_data["diagnosis"]["reasoning"]
                
                report = f"Based on your symptoms ({symptoms_str}), I believe you may have **{main_name}**. \n\n{reasoning}"
                
                # Add differential diagnoses if available
                differentials = patient_data.get("differential_diagnoses", [])
                if differentials:
                    report += "\n\nI've also considered these alternatives:\n"
                    for diff in differentials[:3]:  # Show top 3 alternatives
                        report += f"- {diff.get('name', 'Unknown')}: {int(diff.get('confidence', 0.0) * 100)}% confidence\n"
                
                return report
            else:
                # No valid JSON found, return a simple diagnosis
                return f"Based on your symptoms ({symptoms_str}), I believe you may have {diagnosis_name}. However, this is not a definitive diagnosis, and I recommend consulting a healthcare professional."
                
        except Exception as e:
            logger.error(f"Error generating diagnosis: {str(e)}")
            return f"I've analyzed your symptoms ({symptoms_str}), but I'm unable to provide a definitive diagnosis at this time. Please consult with a healthcare professional for proper evaluation."
    
    async def suggest_mitigations(self, patient_data: Dict[str, Any]) -> str:
        """
        Suggest mitigations for the diagnosed condition using function calling.
        """
        symptoms_str = self.get_symptoms_text(patient_data)
        diagnosis_name = patient_data.get("diagnosis", {}).get("name", "Unknown")
        
        try:
            # Call the mitigations suggestion function
            result = await self.llm_handler.invoke_function(
                "medical.suggest_mitigations",
                {
                    "symptoms": symptoms_str,
                    "diagnosis": diagnosis_name
                },
                service_id="full"  # Use the full model for comprehensive suggestions
            )
            
            mitigations = result.get("text", "")
            
            # Store in patient data
            patient_data["mitigations"] = mitigations
            
            # Return formatted mitigation suggestions
            return f"**Recommendations for {diagnosis_name}:**\n\n{mitigations}"
            
        except Exception as e:
            logger.error(f"Error suggesting mitigations: {str(e)}")
            return "Here are some general recommendations: rest, stay hydrated, and consult with a healthcare professional for personalized advice."
    
    async def detect_emergency(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect if the patient's symptoms indicate a medical emergency.
        """
        symptoms_str = self.get_symptoms_text(patient_data)
        
        # Define critical emergency symptoms
        emergency_indicators = [
            "chest pain", "severe chest pain", "difficulty breathing", "shortness of breath",
            "stroke", "heart attack", "unable to breathe", "coughing blood", "vomiting blood",
            "severe head injury", "loss of consciousness", "unresponsive", "seizure", 
            "severe bleeding", "suicidal", "suicide", "overdose", "poisoning"
        ]
        
        # Check if any symptoms match emergency indicators
        emergency_detected = False
        matching_symptoms = []
        
        for symptom in patient_data.get("symptoms", []):
            if any(indicator in symptom.lower() for indicator in emergency_indicators):
                emergency_detected = True
                matching_symptoms.append(symptom)
        
        if emergency_detected:
            emergency_message = "I've detected potential emergency symptoms: " + ", ".join(matching_symptoms) + ". Please seek immediate medical attention by visiting the nearest emergency room or calling emergency services (911). Don't wait - urgent medical conditions require prompt professional care."
            
            return {
                "is_emergency": True,
                "emergency_symptoms": matching_symptoms,
                "emergency_message": emergency_message
            }
        
        return {"is_emergency": False}
