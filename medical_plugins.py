"""
Medical Knowledge Plugin for Semantic Kernel
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