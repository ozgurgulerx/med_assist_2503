import logging
import time
import json
import re
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def current_utc_timestamp() -> str:
    """
    Simple helper to generate a UTC timestamp string, e.g. 2025-03-10T09:00:00Z
    You can replace with a more robust approach if needed.
    """
    import datetime
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

class DiagnosticEngine:
    """Handles medical diagnosis and symptom analysis."""
    
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

        This is a simplistic approach; for a more robust system, you might store
        each symptom as a dict with onset_date, severity, etc.
        """
        if not symptom:
            logger.warning("Attempted to add empty symptom")
            return
        
        if "symptoms" not in patient_data:
            logger.info("Initializing symptoms list in patient data")
            patient_data["symptoms"] = []
        
        symptoms = patient_data["symptoms"]
        
        # Convert existing symptom strings into text for matching
        existing_keywords = []
        for existing_symptom in symptoms:
            if isinstance(existing_symptom, dict):
                sname = existing_symptom.get("name", "").lower()
            else:
                sname = str(existing_symptom).lower()
            existing_keywords.extend(sname.split())
        
        new_tokens = symptom.lower().split()
        # Basic overlap check
        has_overlap = any(t for t in new_tokens if t in existing_keywords and len(t) > 3)
        
        # If there's overlap with existing symptoms, update with additional info
        if has_overlap and symptoms:
            for idx, existing_symptom in enumerate(symptoms):
                if isinstance(existing_symptom, dict):
                    existing_text = existing_symptom["name"].lower()
                else:
                    existing_text = str(existing_symptom).lower()
                
                if any(token in existing_text for token in new_tokens):
                    # Convert to dict format if it's a string
                    if not isinstance(existing_symptom, dict):
                        symptoms[idx] = {
                            "name": str(existing_symptom),
                            "timestamp": current_utc_timestamp(),
                            "additional_info": symptom
                        }
                        logger.info(f"Updated symptom from string to dict: {symptoms[idx]}")
                    else:
                        # Append to additional info
                        if "additional_info" in existing_symptom:
                            existing_symptom["additional_info"] += f"; {symptom}"
                        else:
                            existing_symptom["additional_info"] = symptom
                        logger.info(f"Updated existing symptom with additional info: {existing_symptom}")
                    return
        
        # If no overlap or empty symptoms list, add as new symptom
        new_symptom = {
            "name": symptom,
            "timestamp": current_utc_timestamp()
        }
        symptoms.append(new_symptom)
        logger.info(f"Added new symptom: {new_symptom}")
    
    def get_symptoms_text(self, patient_data: Dict[str, Any]) -> str:
        """
        Create a short textual summary of the user's symptoms
        by listing each symptom's name. This is used for prompt building.
        """
        symptoms = patient_data.get("symptoms", [])
        if not symptoms:
            logger.info("No symptoms found in patient data")
            return "unknown symptoms"
        
        # For each dict-based symptom, we show the 'name'
        symptom_names = []
        for s in symptoms:
            if isinstance(s, dict):
                symptom_name = s.get("name", "")
                if symptom_name:
                    symptom_names.append(symptom_name)
            else:
                # fallback if it's just a string
                symptom_text = str(s)
                if symptom_text:
                    symptom_names.append(symptom_text)
        
        if not symptom_names:
            logger.warning("Symptoms list exists but no valid symptom names found")
            return "unknown symptoms"
        
        result = ", ".join(symptom_names)
        logger.info(f"Formatted symptoms text: '{result}'")
        return result
    
    async def calculate_diagnosis_confidence(self, symptoms: str) -> Dict[str, Any]:
        """
        Use the LLM to compute a confidence measure (0.0 to 1.0).
        If LLM not available, return a default low confidence.
        """
        if not self.llm_handler.is_available():
            return {
                "confidence": 0.3,
                "reasoning": "Limited confidence - LLM unavailable"
            }
        
        prompt = f"""Analyze these symptoms to assess how confident a medical professional would be in providing a diagnosis:

Symptoms: {symptoms}

Consider:
- Specificity of symptoms
- Duration, severity
- Whether they match multiple conditions or a singular pattern

Respond with a JSON object:
{{
"confidence": 0.7,
"reasoning": "why"
}}
"""
        try:
            response_data = await self.llm_handler.execute_prompt(prompt, use_full_model=False, temperature=0.3)
            response_text = response_data.get("text", "")
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    confidence = float(result.get("confidence", 0.3))
                    confidence = max(0.0, min(1.0, confidence))
                    return {
                        "confidence": confidence,
                        "reasoning": result.get("reasoning", "No reasoning provided")
                    }
                except Exception as e:
                    logger.error(f"Error parsing JSON: {e}")
                    return {"confidence": 0.3, "reasoning": "Error parsing JSON"}
            
            return {"confidence": 0.3, "reasoning": "No JSON found"}
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
        
        # If no follow-up questions asked, do a baseline
        asked_questions = patient_data.get("asked_questions", [])
        question_count = len(asked_questions)
        
        if question_count == 0:
            # baseline
            scount = len(patient_data["symptoms"])
            baseline_conf = min(0.1 * scount, 0.4)
            patient_data["diagnosis"]["confidence"] = baseline_conf
            patient_data["confidence_reasoning"] = f"Initial baseline confidence based on {scount} symptoms"
            logger.info(f"Baseline confidence set to {baseline_conf:.2f} based on {scount} symptoms")
            return
        
        # Otherwise, call the LLM to compute confidence
        logger.info(f"Calculating diagnosis confidence using LLM for symptoms: '{symptoms_str}'")
        conf_data = await self.calculate_diagnosis_confidence(symptoms_str)
        confidence = conf_data.get("confidence", 0.0)
        reasoning = conf_data.get("reasoning", "")
        
        patient_data["diagnosis"]["confidence"] = confidence
        patient_data["confidence_reasoning"] = reasoning
        
        logger.info(f"Diagnosis confidence updated to {confidence:.2f} with reasoning: {reasoning}")
    
    async def generate_followup_question(self, patient_data: Dict[str, Any]) -> str:
        """
        Generate a follow-up question about the user's symptoms.
        The question is appended to asked_questions in 'patient_data'.
        Uses the full model to generate high-value questions that maximize diagnostic value.
        """
        symptoms_str = self.get_symptoms_text(patient_data)
        
        # Create a simple textual representation of asked questions
        asked_list = patient_data.get("asked_questions", [])
        asked_str = ", ".join(q["question"] for q in asked_list if isinstance(q, dict))
        
        logger.info(f"generate_followup_question for: {symptoms_str}")
        
        question_text = "Can you tell me more about your symptoms?"
        if self.llm_handler.is_available():
            try:
                prompt = f"""You are a medical professional gathering information about a patient's symptoms.
Current reported symptoms: {symptoms_str}
Previously asked questions: {asked_str}

Generate ONE specific follow-up question that would provide the highest diagnostic value. Consider:
1. The specific nature and characteristics of the reported symptoms
2. Key differentiating factors that would help narrow down potential diagnoses
3. Important clinical indicators that haven't been asked about yet

Format: Return ONLY the question, without any prefixes or additional text."""
                
                response_data = await self.llm_handler.execute_prompt(prompt, use_full_model=True)
                possible_question = response_data.get("text", "").strip()
                if possible_question:
                    question_text = possible_question
            except Exception as e:
                logger.error(f"Error generating followup question: {e}")
        else:
            logger.info("LLM not available, using fallback question text.")
        
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
        
        # Save back
        patient_data["asked_questions"] = asked_list
        
        return question_text
    
    async def verify_symptoms(self, patient_data: Dict[str, Any]) -> str:
        """
        Verify symptoms using the verifier model (O1).
        
        This method handles two cases:
        1. High confidence (>=0.85): Verify if the diagnosis is correct
        2. Low confidence (<0.85): Explain why symptoms don't fit a known condition 
           and recommend medical consultation
        """
        symptoms_str = self.get_symptoms_text(patient_data)
        diagnosis_data = patient_data.get("diagnosis", {})
        confidence = float(diagnosis_data.get("confidence", 0.0))
        verification_trigger = patient_data.get("verification_info", {}).get("trigger_reason", "unknown")
        
        # Check if verifier model is available
        if not self.llm_handler.is_verifier_model_available():
            logger.warning("Verifier model not available, using fallback verification")
            return f"I understand you're experiencing: {symptoms_str} (Confidence: {confidence:.2f}). Please confirm if this is accurate or if anything is missing."
        
        try:
            # Record which model was used for verification
            patient_data["verification_model"] = "O1/medical-verification"
            
            if verification_trigger == "high_confidence":
                # High confidence case (>=0.85) - verify the diagnosis
                logger.info(f"High confidence verification ({confidence:.2f} >= 0.85), using verifier model")
                
                prompt = f"""We have collected these symptoms:
{symptoms_str}

We have a tentative diagnosis of {diagnosis_data.get('name', 'Unknown')} with confidence {confidence:.2f}.
Please confirm if this diagnosis is correct, or refine it. 
Return a JSON object with "verification": "agree" or "disagree",
optionally "diagnosis_name" for a refined name, 
and "notes" for extra detail.
"""
                resp = await self.llm_handler.execute_prompt(prompt, use_verifier_model=True)
                text = resp.get("text", "")
                
                # Parse JSON
                jmatch = re.search(r'\{.*\}', text, re.DOTALL)
                if jmatch:
                    data = json.loads(jmatch.group(0))
                    verification = data.get("verification", "disagree")
                    if verification.lower() == "agree":
                        # If the verifier agrees, we can finalize
                        diag_name = data.get("diagnosis_name", diagnosis_data.get("name", "ConfirmedDiagnosis"))
                        diagnosis_data["name"] = diag_name
                        # We keep confidence as is
                        patient_data["diagnosis"] = diagnosis_data
                        
                        # Mark this for immediate report generation
                        patient_data["ready_for_report"] = True
                        patient_data["verification_complete"] = True
                        
                        # Return a direct statement that the diagnosis is confirmed
                        return f"I've confirmed your symptoms likely represent: {diag_name}. Based on this assessment, I'll provide you with a medical report."
                    else:
                        # Verifier disagrees -> we ask more
                        return f"The verifier model suggests we need more information. Please clarify any missing or incorrect symptom details."
                else:
                    logger.warning("Could not parse JSON from verifier model response")
                    return f"I need to verify your symptoms further. Can you confirm if these accurately describe your condition: {symptoms_str}?"
            
            else:  # Low confidence case
                # Low confidence case (<0.85) - explain why symptoms don't fit a known condition
                logger.info(f"Low confidence verification ({confidence:.2f} < 0.85), using verifier model for explanation")
                
                prompt = f"""We have collected these symptoms:
{symptoms_str}

After multiple follow-up questions, our diagnostic confidence remains low at {confidence:.2f}.

As a medical expert, please provide:
1. A narrative paragraph explaining why these symptoms may not fit clearly into a known condition with high confidence
2. A list of possible conditions these symptoms could indicate
3. A clear recommendation to consult with a medical professional, including what type of specialist might be appropriate
4. Any urgent warning signs the patient should watch for

Format your response as natural language paragraphs, not as JSON.
"""
                resp = await self.llm_handler.execute_prompt(prompt, use_verifier_model=True)
                explanation = resp.get("text", "")
                
                # Store the explanation in patient data
                if "verification_info" not in patient_data:
                    patient_data["verification_info"] = {}
                patient_data["verification_info"]["low_confidence_explanation"] = explanation
                
                # Add a referral note to the patient data
                patient_data["referral_needed"] = True
                
                return explanation
                
        except Exception as e:
            logger.error(f"Error during verification process: {e}")
            return f"An error occurred while analyzing your symptoms. Let's clarify: Do these symptoms accurately describe your condition: {symptoms_str}?"
    
    async def suggest_mitigations(self, patient_data: Dict[str, Any]) -> str:
        """
        Suggest mitigations based on the (final) diagnosis
        """
        diag_info = patient_data.get("diagnosis", {})
        diag_name = diag_info.get("name", None)
        conf_val = diag_info.get("confidence", 0.0)
        sym_str = self.get_symptoms_text(patient_data)
        
        if not diag_name or diag_name == "unknown":
            return ("I do not have enough to determine a specific diagnosis yet. "
                    "A general recommendation is rest, hydration, and consult a healthcare provider if symptoms worsen.")
        
        if self.llm_handler.is_available():
            try:
                prompt = f"""Diagnosis: {diag_name} (confidence {conf_val:.2f})
Symptoms: {sym_str}

Suggest general care or mitigations, emphasizing this is not a substitute for professional care."""
                rdata = await self.llm_handler.execute_prompt(prompt, use_full_model=True)
                return rdata.get("text", "")
            except Exception as e:
                logger.error(f"Error generating mitigations: {e}")
        
        return ("Here's a general suggestion: get rest, stay hydrated, track any changes in symptoms, and follow up with a healthcare provider if needed.")
    
    async def detect_emergency(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze symptoms to detect potential emergency medical situations that require immediate attention.
        
        Args:
            patient_data: The patient data including symptoms
            
        Returns:
            Dictionary with 'is_emergency' boolean and 'emergency_message' string if it's an emergency
        """
        symptoms_str = self.get_symptoms_text(patient_data)
        if not symptoms_str or symptoms_str == "unknown symptoms":
            return {"is_emergency": False}
        
        logger.info(f"Checking for emergency in symptoms: '{symptoms_str}'")
        
        if not self.llm_handler.is_available():
            # Fallback emergency detection for critical keywords
            emergency_keywords = [
                "can't breathe", "cannot breathe", "difficulty breathing", "severe chest pain",
                "heart attack", "stroke", "seizure", "unconscious", "unresponsive",
                "severe bleeding", "coughing blood", "vomiting blood", "suicidal",
                "overdose", "poisoning", "anaphylaxis", "allergic reaction"
            ]
            
            for keyword in emergency_keywords:
                if keyword in symptoms_str.lower():
                    return {
                        "is_emergency": True,
                        "emergency_message": f"I've detected potential emergency symptoms related to {keyword}. "
                                             f"Please seek immediate medical attention by calling emergency services or going to the nearest emergency room."
                    }
            
            return {"is_emergency": False}
        
        # Use LLM to detect emergency situations
        prompt = f"""Analyze these symptoms to determine if they represent a medical emergency requiring immediate attention:

Symptoms: {symptoms_str}

A medical emergency is a situation that poses an immediate risk to a person's health, life, or long-term function and requires urgent intervention.
Examples include: heart attack, stroke, severe bleeding, difficulty breathing, severe allergic reactions, etc.

Respond with a JSON object:
{{
"is_emergency": true/false,
"reasoning": "brief explanation of why this is or is not an emergency",
"emergency_message": "If this is an emergency, provide a clear message advising the patient to seek immediate medical attention"
}}
"""
        
        try:
            response_data = await self.llm_handler.execute_prompt(prompt, use_full_model=True, temperature=0.1)
            response_text = response_data.get("text", "")
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    is_emergency = result.get("is_emergency", False)
                    
                    if is_emergency:
                        # Store emergency information in patient data
                        if "emergency_info" not in patient_data:
                            patient_data["emergency_info"] = {}
                        
                        patient_data["emergency_info"] = {
                            "detected_at": current_utc_timestamp(),
                            "reasoning": result.get("reasoning", "No reasoning provided"),
                            "symptoms": symptoms_str
                        }
                        
                        logger.warning(f"EMERGENCY DETECTED: {result.get('reasoning', 'No reasoning provided')}")
                        
                        return {
                            "is_emergency": True,
                            "emergency_message": result.get("emergency_message", 
                                                        "MEDICAL EMERGENCY DETECTED. Please seek immediate medical attention "
                                                        "by calling emergency services or going to the nearest emergency room.")
                        }
                    else:
                        logger.info(f"No emergency detected in symptoms: '{symptoms_str}'")
                        return {"is_emergency": False}
                    
                except Exception as e:
                    logger.error(f"Error parsing emergency detection JSON: {e}")
            
            logger.warning("Failed to parse emergency detection response")
            return {"is_emergency": False}
            
        except Exception as e:
            logger.error(f"Error in emergency detection: {str(e)}")
            return {"is_emergency": False}

    async def generate_diagnosis(self, patient_data: Dict[str, Any]) -> str:
        """
        Generate a diagnosis based on the patient's symptoms.
        
        This method analyzes the patient's symptoms and generates a diagnosis with confidence score.
        It also generates differential diagnoses and stores them in the patient data.
        
        Args:
            patient_data: The patient data including symptoms
            
        Returns:
            A string describing the diagnosis and recommendations
        """
        symptoms_str = self.get_symptoms_text(patient_data)
        if not symptoms_str or symptoms_str == "unknown symptoms":
            return "I need more information about your symptoms to provide a diagnosis."
        
        logger.info(f"Generating diagnosis for symptoms: '{symptoms_str}'")
        
        # Ensure diagnosis structure exists
        if "diagnosis" not in patient_data:
            patient_data["diagnosis"] = {"name": None, "confidence": 0.0}
        
        # Get current confidence score
        confidence = patient_data["diagnosis"].get("confidence", 0.0)
        
        if not self.llm_handler.is_available():
            # Fallback if LLM is not available
            return ("Based on the information provided, I cannot make a specific diagnosis. "
                   "Please consult with a healthcare professional for proper evaluation.")
        
        # Use LLM to generate diagnosis
        prompt = f"""Analyze these symptoms and provide a diagnosis with confidence level and differential diagnoses:

Symptoms: {symptoms_str}

Respond with a JSON object:
{{
"most_probable_diagnosis": "name of the most likely diagnosis",
"confidence": 0.7,
"reasoning": "brief explanation of why this diagnosis is likely",
"differential_diagnoses": [
  {{
    "name": "alternative diagnosis 1",
    "confidence": 0.4,
    "key_indicators": "symptoms that suggest this diagnosis"
  }},
  // Add 2-3 more alternatives
]
}}
"""
        
        try:
            response_data = await self.llm_handler.execute_prompt(prompt, use_full_model=True, temperature=0.3)
            response_text = response_data.get("text", "")
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    
                    # Extract the most probable diagnosis
                    most_probable_diagnosis = result.get("most_probable_diagnosis", "Uncertain diagnosis")
                    confidence_value = float(result.get("confidence", 0.5))
                    confidence_value = max(0.0, min(1.0, confidence_value))  # Ensure it's between 0 and 1
                    reasoning = result.get("reasoning", "Insufficient information for detailed reasoning")
                    
                    # Update the diagnosis in patient data
                    patient_data["diagnosis"] = {
                        "name": most_probable_diagnosis,
                        "confidence": confidence_value
                    }
                    
                    # Store reasoning
                    patient_data["confidence_reasoning"] = reasoning
                    
                    # Store differential diagnoses
                    differential_diagnoses = result.get("differential_diagnoses", [])
                    patient_data["differential_diagnoses"] = differential_diagnoses
                    
                    # Log the diagnosis information including the most probable diagnosis
                    logger.info(f"Generated diagnosis: {most_probable_diagnosis} with confidence {confidence_value:.2f}")
                    logger.info(f"Diagnosis confidence is for: {most_probable_diagnosis}")
                    
                    # Generate a response based on confidence level
                    if confidence_value >= 0.7:
                        response = f"Based on your symptoms, you may have {most_probable_diagnosis} (confidence: {confidence_value:.2f})."
                    elif confidence_value >= 0.4:
                        response = f"Your symptoms suggest a possible {most_probable_diagnosis}, but with moderate confidence ({confidence_value:.2f})."
                    else:
                        response = f"Your symptoms don't clearly point to a specific diagnosis (confidence: {confidence_value:.2f})."
                    
                    return response
                    
                except Exception as e:
                    logger.error(f"Error parsing diagnosis JSON: {e}")
                    # If we fail to parse the JSON but have a diagnosis name from before, keep it
                    if patient_data["diagnosis"].get("name") is None:
                        patient_data["diagnosis"]["name"] = "Uncertain diagnosis"
                    logger.info(f"Diagnosis confidence is for: {patient_data['diagnosis'].get('name', 'Uncertain diagnosis')}")
                    return f"I'm having trouble analyzing your symptoms."
            
            logger.warning("Failed to parse diagnosis response")
            # If we fail to get a proper response but have a confidence score, make sure we have a diagnosis name
            if patient_data["diagnosis"].get("name") is None:
                patient_data["diagnosis"]["name"] = "Uncertain diagnosis"
            logger.info(f"Diagnosis confidence is for: {patient_data['diagnosis'].get('name', 'Uncertain diagnosis')}")
            return f"Based on the information provided, I cannot make a specific diagnosis with confidence."
            
        except Exception as e:
            logger.error(f"Error in diagnosis generation: {str(e)}")
            # If we encounter an error but have a confidence score, make sure we have a diagnosis name
            if patient_data["diagnosis"].get("name") is None:
                patient_data["diagnosis"]["name"] = "Uncertain diagnosis"
            logger.info(f"Diagnosis confidence is for: {patient_data['diagnosis'].get('name', 'Uncertain diagnosis')}")
            return f"I encountered an error while analyzing your symptoms."
