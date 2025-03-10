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
            return
        
        symptoms = patient_data.get("symptoms", [])
        
        # Convert existing symptom strings into text for matching
        existing_keywords = []
        for idx, existing_symptom in enumerate(symptoms):
            # If 'existing_symptom' is a dict in your new structure, adapt accordingly
            if isinstance(existing_symptom, dict):
                sname = existing_symptom.get("name", "").lower()
                existing_keywords.extend(sname.split())
            else:
                # Legacy fallback: if it's just a string
                sname = existing_symptom.lower()
                existing_keywords.extend(sname.split())
        
        new_tokens = symptom.lower().split()
        # Basic overlap check
        has_overlap = any(t for t in new_tokens if t in existing_keywords and len(t) > 3)
        
        # If it looks like additional detail about an existing symptom, we can
        # incorporate it. This is a naive approach:
        if has_overlap and symptoms:
            # For brevity, we'll just append text to the first symptom's additional_info
            # Or combine them. In a real system, you'd do a more refined approach.
            # Example: pick the first dictionary in 'symptoms' and update it.
            for sdict in symptoms:
                if isinstance(sdict, dict):
                    # Add to additional_info
                    if "additional_info" not in sdict:
                        sdict["additional_info"] = {}
                    detail_key = f"detail_{int(time.time())}"
                    sdict["additional_info"][detail_key] = symptom
                    logger.info(f"Updated existing symptom details: {sdict}")
                    return
                else:
                    # It's a string, just combine them
                    combined = f"{sdict} ({symptom})"
                    # Replace
                    idx_to_update = symptoms.index(sdict)
                    symptoms[idx_to_update] = combined
                    logger.info(f"Combined existing symptom: {combined}")
                    patient_data["symptoms"] = symptoms
                    return
        
        # Otherwise, treat it as a new symptom
        # In your new structure, you might want to store it as a dict
        new_symptom_entry = {
            "name": symptom,
            "onset_date": "",
            "severity": "",
            "additional_info": {}
        }
        symptoms.append(new_symptom_entry)
        patient_data["symptoms"] = symptoms
        logger.info(f"Added new symptom: {new_symptom_entry}")
    
    def get_symptoms_text(self, patient_data: Dict[str, Any]) -> str:
        """
        Create a short textual summary of the user's symptoms
        by listing each symptom's name. This is used for prompt building.
        """
        symptoms = patient_data.get("symptoms", [])
        if not symptoms:
            return "unknown symptoms"
        
        # For each dict-based symptom, we show the 'name'
        symptom_names = []
        for s in symptoms:
            if isinstance(s, dict):
                symptom_names.append(s.get("name", ""))
            else:
                # fallback if it's just a string
                symptom_names.append(str(s))
        
        if not any(symptom_names):
            return "unknown symptoms"
        return ", ".join(symptom_names)
    
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
        symptoms_str = self.get_symptoms_text(patient_data)
        if not symptoms_str or symptoms_str == "unknown symptoms":
            # No real symptoms => confidence = 0
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
            patient_data["confidence_reasoning"] = "Initial baseline confidence"
            logger.info(f"Baseline confidence set to {baseline_conf:.2f}")
            return
        
        # Otherwise, call the LLM to compute confidence
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
        """
        symptoms_str = self.get_symptoms_text(patient_data)
        
        # Create a simple textual representation of asked questions
        # (In your code, each 'asked_question' is a dict, but for LLM prompt building
        # we can just join them as text if we want.)
        asked_list = patient_data.get("asked_questions", [])
        asked_str = ", ".join(q["question"] for q in asked_list if isinstance(q, dict))
        
        logger.info(f"generate_followup_question for: {symptoms_str}")
        
        question_text = "Can you tell me more about your symptoms?"
        if self.llm_handler.is_available():
            try:
                prompt = f"""I'm a medical assistant collecting more details.
Current symptoms: {symptoms_str}
Already asked: {asked_str}

Suggest one relevant follow-up question to clarify the patient's condition."""
                
                response_data = await self.llm_handler.execute_prompt(prompt, use_full_model=False)
                possible_question = response_data.get("text", "").strip()
                if possible_question:
                    question_text = possible_question
            except Exception as e:
                logger.error(f"Error generating followup question: {e}")
        else:
            logger.info("LLM not available, using fallback question text.")
        
        # Record it in asked_questions with timestamps
        asked_dict = {
            "question": question_text,
            "answer": "",
            "timestamp_asked": current_utc_timestamp(),
            "timestamp_answered": ""
        }
        asked_list.append(asked_dict)
        
        # Save back
        patient_data["asked_questions"] = asked_list
        
        return question_text
    
    async def verify_symptoms(self, patient_data: Dict[str, Any]) -> str:
        """
        If confidence >= 0.85, try a 'verifier model'. If it agrees, finalize the diagnosis.
        Otherwise, produce a question to confirm the listed symptoms.
        """
        symptoms_str = self.get_symptoms_text(patient_data)
        diagnosis_data = patient_data.get("diagnosis", {})
        confidence = float(diagnosis_data.get("confidence", 0.0))
        
        # 0.85 is the new threshold
        use_verifier = confidence >= 0.85 and self.llm_handler.is_verifier_model_available()
        
        if use_verifier:
            logger.info(f"Confidence {confidence:.2f} >= 0.85, using verifier model")
            try:
                # Prompt the verifier to confirm
                prompt = f"""We have collected these symptoms:
{symptoms_str}

We have a tentative diagnosis with confidence {confidence:.2f}.
Please confirm if this is correct, or refine it. 
Return a JSON object with "verification": "agree" or "disagree",
optionally "diagnosis_name" for a refined name, 
and "notes" for extra detail.
"""
                resp = await self.llm_handler.execute_prompt(prompt, use_verifier_model=True)
                text = resp.get("text", "")
                patient_data["verification_model"] = resp.get("model", "verifier_unknown")
                
                # Parse JSON
                jmatch = re.search(r'\{.*\}', text, re.DOTALL)
                if jmatch:
                    data = json.loads(jmatch.group(0))
                    verification = data.get("verification", "disagree")
                    if verification.lower() == "agree":
                        # If the verifier agrees, we can finalize
                        diag_name = data.get("diagnosis_name", "ConfirmedDiagnosis")
                        diagnosis_data["name"] = diag_name
                        # We keep confidence as is
                        patient_data["diagnosis"] = diagnosis_data
                        
                        # Return a direct statement that the diagnosis is confirmed
                        return f"I've confirmed your symptoms likely represent: {diag_name}. We will proceed with that assessment."
                    else:
                        # Verifier disagrees -> we ask more
                        return f"The verifier model suggests we need more info or there's a mismatch. Please clarify any missing or incorrect symptom details."
                else:
                    return f"I tried verifying your symptoms at {confidence:.2f} confidence, but couldn't parse the verifier's response. Let's clarify further."
            except Exception as e:
                logger.error(f"Error verifying with the specialized model: {e}")
                return f"An error occurred while verifying your symptoms. Let's clarify: Do these symptoms accurately describe your condition: {symptoms_str}?"
        else:
            # Normal or fallback verification
            # Just produce a question asking the user to confirm
            return f"I understand you're experiencing: {symptoms_str} (Confidence: {confidence:.2f}). Please confirm if this is accurate or if anything is missing."

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

