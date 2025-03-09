"""
Enhanced Medical Knowledge Plugin for Semantic Kernel - Fallback Functions Only
"""
import logging
from semantic_kernel.functions import kernel_function

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class MedicalKnowledgePlugin:
    """
    Plugin providing medical knowledge functions.
    Note: These functions should only be used as fallbacks when the LLM service is unavailable.
    Primary medical reasoning is handled directly by the LLM in medical_assistant_bot.py.
    """
    
    @kernel_function
    async def analyze_medical_query(self, query: str, patient_context: str = "") -> str:
        """
        Analyze a medical query to determine the appropriate response or follow-up questions.
        This is a fallback when LLM is unavailable.
        
        Args:
            query: The patient's medical query or description of symptoms
            patient_context: Optional context about the patient's history, demographics, or previously mentioned symptoms
        """
        logger.info(f"Fallback analyze_medical_query: {query}")
        logger.info(f"Patient context: {patient_context}")
        
        # Simple fallback response
        response = """Based on the information you've provided, I recommend discussing your symptoms with a healthcare provider. 
        
Without a full assessment by a medical professional, it's difficult to determine the exact cause of your symptoms.

In the meantime:
- Monitor your symptoms and note any changes
- Rest and stay hydrated
- Avoid activities that worsen your symptoms
- Seek immediate medical attention if symptoms become severe

This is general advice and not a substitute for professional medical care."""
        
        logger.info(f"Fallback Response - analyze_medical_query: {response}")
        return response
    
    @kernel_function
    async def generate_followup_questions(self, current_symptoms: str, medical_history: str = "", previously_asked: str = "") -> str:
        """
        Generate a general follow-up question as a fallback when LLM is unavailable.
        
        Args:
            current_symptoms: Symptoms mentioned so far
            medical_history: Patient's relevant medical history if available
            previously_asked: Questions already asked to avoid repetition
        """
        logger.info(f"Fallback generate_followup_questions - Symptoms: {current_symptoms}")
        
        # Simple general follow-up question
        response = "Could you tell me more about when these symptoms started and if anything makes them better or worse?"
        
        logger.info(f"Fallback Response - generate_followup_questions: {response}")
        return response
    
    @kernel_function
    async def provide_medical_information(self, topic: str, patient_demographics: str = "") -> str:
        """
        Provide general medical information as a fallback when LLM is unavailable.
        
        Args:
            topic: The medical topic being discussed
            patient_demographics: Optional demographic information to personalize the response
        """
        logger.info(f"Fallback provide_medical_information - Topic: {topic}")
        
        response = f"""Here's some general information about health concerns:

Medical symptoms can have many different causes, ranging from minor to serious. It's important to:

1. Track your symptoms (when they occur, severity, duration)
2. Note any patterns or triggers
3. Follow healthy habits like proper nutrition, hydration, and rest
4. Consult with a healthcare provider for proper evaluation

Remember that online information is general and not a substitute for professional medical advice.

For specific information about {topic}, please consult with a healthcare provider."""
        
        logger.info(f"Fallback Response - provide_medical_information: {response}")
        return response