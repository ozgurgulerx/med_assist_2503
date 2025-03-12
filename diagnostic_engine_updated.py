import logging
import re
import json
from typing import Dict, List, Any, Tuple, Optional

from llm_handler import LLMHandler
from medical_knowledge_plugin import MedicalKnowledgePlugin

logger = logging.getLogger(__name__)

class DiagnosticEngine:
    """
    Engine for medical diagnosis and symptom analysis.
    
    This class handles the medical diagnosis process, including symptom verification,
    follow-up question generation, and diagnosis confidence calculation.
    """
    
    def __init__(self, llm_handler: LLMHandler, medical_plugin: MedicalKnowledgePlugin):
        """
        Initialize the diagnostic engine.
        
        Args:
            llm_handler: Handler for LLM interactions
            medical_plugin: Plugin for medical knowledge access
        """
        self.llm_handler = llm_handler
        self.medical_plugin = medical_plugin
    
    def get_patient_data(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize the patient data structure if it doesn't exist.
        
        Args:
            user_data: User data dictionary
            
        Returns:
            The patient data structure
        """
        if 'patient_data' not in user_data:
            user_data['patient_data'] = {
                'symptoms': [],
                'diagnosis': {'name': None, 'confidence': 0.0},
                'demographics': {
                    'age': None,
                    'gender': None,
                    'weight': None,  # kg
                    'height': None,  # cm
                },
                'asked_questions': [],
                'answered_questions': [],
                'verification_status': 'not_verified'
            }
        return user_data['patient_data']
    
    def get_symptoms_text(self, patient_data: Dict[str, Any]) -> str:
        """
        Get a formatted string of all symptoms from patient data.
        
        Args:
            patient_data: The patient data
            
        Returns:
            Formatted string of symptoms
        """
        symptoms = patient_data.get('symptoms', [])
        if not symptoms:
            return 'unknown symptoms'
        
        # Format symptoms as a list
        symptoms_text = ', '.join(symptoms)
        return symptoms_text
    
    def add_symptom(self, patient_data: Dict[str, Any], symptom: str) -> None:
        """
        Add a symptom to the patient data if it's not already present.
        
        Args:
            patient_data: The patient data
            symptom: The symptom to add
        """
        if 'symptoms' not in patient_data:
            patient_data['symptoms'] = []
        
        # Normalize the symptom text
        symptom = symptom.strip().lower()
        
        # Check if this or a similar symptom is already in the list
        for existing in patient_data['symptoms']:
            if symptom in existing.lower() or existing.lower() in symptom:
                logger.info(f"Similar symptom already exists: '{existing}' vs '{symptom}'"
