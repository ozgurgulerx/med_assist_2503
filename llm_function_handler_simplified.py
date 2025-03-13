import os
import json
import logging
from typing import Dict, Any, List, Optional

from semantic_kernel import Kernel
from semantic_kernel.functions import KernelFunction
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

logger = logging.getLogger(__name__)

class LLMFunctionHandler:
    """
    Enhanced LLM handler using Semantic Kernel for function calling
    with the medical assistant bot.
    """

    def __init__(self):
        """Initialize the LLM function handler using Semantic Kernel."""
        # Create a Kernel instance
        self.kernel = Kernel()
        
        # Setup chat services
        self.setup_chat_services()
        
        # Register medical functions
        self.register_medical_functions()

    def setup_chat_services(self):
        """Set up Azure OpenAI chat services."""
        try:
            # "mini" model for quick usage
            mini_service = AzureChatCompletion(
                deployment_name="gpt-4o",
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-06-01"
            )

            # "full" model for advanced usage
            full_service = AzureChatCompletion(
                deployment_name="o3-mini",
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2025-01-01-preview"
            )
            
            # "verifier" model for high-confidence verification
            verifier_service = AzureChatCompletion(
                deployment_name="o1",  # O1 model for medical verification
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2025-01-01-preview"
            )

            # Register services with the kernel
            self.kernel.add_service(mini_service, "mini")
            self.kernel.add_service(full_service, "full")
            self.kernel.add_service(verifier_service, "verifier")
            
            logger.info("Successfully registered 'mini', 'full', and 'verifier' chat services.")

        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI services: {str(e)}")
    
    def register_medical_functions(self):
        """Register medical functions to Semantic Kernel for function calling."""
        try:
            # Create a plugin for medical functions
            medical_plugin = self.kernel.add_plugin(
                name="medical",
                description="Medical assistant functions"
            )
            
            # Register diagnosis function
            diagnosis_function = KernelFunction.from_prompt(
                prompt_template=self._get_diagnosis_prompt(),
                description="Generates a diagnosis based on patient symptoms.",
                name="generate_diagnosis",
            )
            medical_plugin.add_function(diagnosis_function)
            
            # Register follow-up question function
            followup_function = KernelFunction.from_prompt(
                prompt_template=self._get_followup_prompt(),
                description="Suggests a follow-up question to gather more health information.",
                name="generate_followup_question"
            )
            medical_plugin.add_function(followup_function)
            
            # Register verification function
            verification_function = KernelFunction.from_prompt(
                prompt_template=self._get_verification_prompt(),
                description="Verifies a high-confidence diagnosis using the 'verifier' model.",
                name="verify_high_confidence_diagnosis"
            )
            medical_plugin.add_function(verification_function)
            
            # Register mitigation function
            mitigation_function = KernelFunction.from_prompt(
                prompt_template=self._get_mitigations_prompt(),
                description="Suggests mitigations and management steps for a diagnosis.",
                name="suggest_mitigations"
            )
            medical_plugin.add_function(mitigation_function)
            
            # Register symptom extraction function
            symptom_function = KernelFunction.from_prompt(
                prompt_template=self._get_symptom_extraction_prompt(),
                description="Extracts symptoms from a user message.",
                name="extract_symptoms"
            )
            medical_plugin.add_function(symptom_function)
            
            # Register intent classification function
            intent_function = KernelFunction.from_prompt(
                prompt_template=self._get_intent_classification_prompt(),
                description="Classifies the intent of a user message.",
                name="classify_intent"
            )
            medical_plugin.add_function(intent_function)
            
            # Register intent extraction function
            intent_extraction_function = KernelFunction.from_prompt(
                prompt_template=self._get_intent_extraction_prompt(),
                description="Extracts multiple intents from a user message.",
                name="extract_intents"
            )
            medical_plugin.add_function(intent_extraction_function)
            
            # Register report generation function
            report_function = KernelFunction.from_prompt(
                prompt_template=self._get_report_prompt(),
                description="Generates a medical report based on diagnosis and symptoms.",
                name="generate_report"
            )
            medical_plugin.add_function(report_function)
            
            logger.info("Successfully registered medical assistant functions.")
        except Exception as e:
            logger.error(f"Error registering medical functions: {str(e)}")
    
    async def generate_diagnosis(self, symptoms: str, service_id: str = "full") -> Dict[str, Any]:
        """
        Generate a diagnosis based on symptoms.
        
        Args:
            symptoms: Patient symptoms as a string
            service_id: The AI service to use
            
        Returns:
            Dictionary with diagnosis information
        """
        try:
            # Verify API key and endpoint are set
            if not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"):
                logger.error("Azure OpenAI credentials not properly configured")
                return {
                    "text": "I'm having trouble connecting to my knowledge base due to missing credentials. Please check your environment configuration.",
                    "error_type": "configuration",
                    "model": "error",
                    "service_id": service_id
                }
                
            kernel_args = KernelArguments(symptoms=symptoms)
            kernel_args.set_ai_service(service_id)
            
            result = await self.kernel.invoke_async(
                function_name="medical-generate_diagnosis",
                arguments=kernel_args
            )
            
            response_text = str(result)
            logger.info(f"Diagnosis response ({service_id}): {response_text[:100]}...")
            
            return {
                "text": response_text,
                "model": service_id,
                "service_id": service_id
            }
        except Exception as e:
            error_message = str(e).lower()
            # Differentiate between connection errors and other errors
            if any(term in error_message for term in ["connection", "timeout", "network", "connect", "unreachable", "server", "request"]):
                logger.error(f"Azure OpenAI connection error: {str(e)}")
                return {
                    "text": "I'm having trouble connecting to my knowledge base. Please try again in a moment.",
                    "error_type": "connection",
                    "model": "error",
                    "service_id": service_id
                }
            else:
                logger.error(f"Error in diagnosis generation: {str(e)}")
                return {
                    "text": f"An unexpected error occurred: {str(e)}",
                    "error_type": "processing",
                    "model": "error",
                    "service_id": service_id
                }
    
    async def generate_followup_question(self, symptoms: str, asked_questions: str, service_id: str = "mini") -> Dict[str, Any]:
        """Generate a follow-up question to gather more information.
        
        Args:
            symptoms: Patient symptoms as a string
            asked_questions: Previously asked questions as a string
            service_id: The AI service to use
            
        Returns:
            Dictionary with follow-up question
        """
        try:
            # Verify API key and endpoint are set
            if not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"):
                logger.error("Azure OpenAI credentials not properly configured")
                return {
                    "text": "I'm having trouble connecting to my knowledge base due to missing credentials.",
                    "error_type": "configuration",
                    "model": "error",
                    "service_id": service_id
                }
                
            kernel_args = KernelArguments(symptoms=symptoms, asked_questions=asked_questions)
            kernel_args.set_ai_service(service_id)
            
            result = await self.kernel.invoke_async(
                function_name="medical-generate_followup_question",
                arguments=kernel_args
            )
            
            response_text = str(result)
            logger.info(f"Follow-up question response ({service_id}): {response_text[:100]}...")
            
            return {
                "text": response_text,
                "model": service_id,
                "service_id": service_id
            }
        except Exception as e:
            error_message = str(e).lower()
            # Differentiate between connection errors and other errors
            if any(term in error_message for term in ["connection", "timeout", "network", "connect", "unreachable", "server", "request"]):
                logger.error(f"Azure OpenAI connection error: {str(e)}")
                return {
                    "text": "I'm having trouble connecting to my knowledge base. Please try again in a moment.",
                    "error_type": "connection",
                    "model": "error", 
                    "service_id": service_id
                }
            else:
                logger.error(f"Error generating follow-up question: {str(e)}")
                return {
                    "text": "Can you tell me more about your symptoms?",
                    "error_type": "processing",
                    "model": "error",
                    "service_id": service_id
                }
    
    async def verify_high_confidence_diagnosis(self, symptoms: str, diagnosis: str, confidence: str, service_id: str = "verifier") -> Dict[str, Any]:
        """Verify a high-confidence diagnosis.
        
        Args:
            symptoms: Patient symptoms as a string
            diagnosis: The diagnosis to verify
            confidence: Confidence level as a string
            service_id: The AI service to use
            
        Returns:
            Dictionary with verification result
        """
        try:
            kernel_args = KernelArguments(
                symptoms=symptoms,
                diagnosis=diagnosis,
                confidence=confidence
            )
            kernel_args.set_ai_service(service_id)
            
            result = await self.kernel.invoke_async(
                function_name="medical-verify_high_confidence_diagnosis",
                arguments=kernel_args
            )
            
            response_text = str(result)
            logger.info(f"Verification response ({service_id}): {response_text[:100]}...")
            
            return {
                "text": response_text,
                "model": service_id,
                "service_id": service_id
            }
        except Exception as e:
            logger.error(f"Error in diagnosis verification: {str(e)}")
            return {
                "text": f"Error in diagnosis verification: {str(e)}",
                "model": "error",
                "service_id": service_id
            }
    
    async def suggest_mitigations(self, diagnosis: str, symptoms: str, service_id: str = "full") -> Dict[str, Any]:
        """Suggest mitigations for a diagnosis.
        
        Args:
            diagnosis: The diagnosis to provide mitigations for
            symptoms: Patient symptoms as a string
            service_id: The AI service to use
            
        Returns:
            Dictionary with mitigation suggestions
        """
        try:
            kernel_args = KernelArguments(
                diagnosis=diagnosis,
                symptoms=symptoms
            )
            kernel_args.set_ai_service(service_id)
            
            result = await self.kernel.invoke_async(
                function_name="medical-suggest_mitigations",
                arguments=kernel_args
            )
            
            response_text = str(result)
            logger.info(f"Mitigation response ({service_id}): {response_text[:100]}...")
            
            return {
                "text": response_text,
                "model": service_id,
                "service_id": service_id
            }
        except Exception as e:
            logger.error(f"Error in mitigation suggestion: {str(e)}")
            return {
                "text": f"Error in mitigation suggestion: {str(e)}",
                "model": "error",
                "service_id": service_id
            }
    
    async def extract_symptoms(self, message: str, service_id: str = "mini") -> Dict[str, Any]:
        """Extract symptoms from a user message.
        
        Args:
            message: The user's message
            service_id: The AI service to use
            
        Returns:
            Dictionary with extracted symptoms
        """
        try:
            kernel_args = KernelArguments(message=message)
            kernel_args.set_ai_service(service_id)
            
            result = await self.kernel.invoke_async(
                function_name="medical-extract_symptoms",
                arguments=kernel_args
            )
            
            response_text = str(result)
            logger.info(f"Symptom extraction response ({service_id}): {response_text[:100]}...")
            
            return {
                "text": response_text,
                "model": service_id,
                "service_id": service_id
            }
        except Exception as e:
            logger.error(f"Error in symptom extraction: {str(e)}")
            return {
                "text": f"Error in symptom extraction: {str(e)}",
                "model": "error",
                "service_id": service_id
            }
    
    async def classify_intent(self, message: str, context: str = "", service_id: str = "mini") -> Dict[str, Any]:
        """Classify the intent of a user message.
        
        Args:
            message: The user's message
            context: Optional context string
            service_id: The AI service to use
            
        Returns:
            Dictionary with intent classification
        """
        try:
            kernel_args = KernelArguments(
                message=message,
                context=context
            )
            kernel_args.set_ai_service(service_id)
            
            result = await self.kernel.invoke_async(
                function_name="medical-classify_intent",
                arguments=kernel_args
            )
            
            response_text = str(result)
            logger.info(f"Intent classification response ({service_id}): {response_text[:100]}...")
            
            return {
                "text": response_text,
                "model": service_id,
                "service_id": service_id
            }
        except Exception as e:
            logger.error(f"Error in intent classification: {str(e)}")
            return {
                "text": f"Error in intent classification: {str(e)}",
                "model": "error",
                "service_id": service_id
            }
    
    async def extract_intents(self, message: str, context: str = "", service_id: str = "mini") -> Dict[str, Any]:
        """Extract multiple intents from a user message.
        
        Args:
            message: The user's message
            context: Optional context string
            service_id: The AI service to use
            
        Returns:
            Dictionary with extracted intents
        """
        try:
            kernel_args = KernelArguments(
                message=message,
                context=context
            )
            kernel_args.set_ai_service(service_id)
            
            result = await self.kernel.invoke_async(
                function_name="medical-extract_intents",
                arguments=kernel_args
            )
            
            response_text = str(result)
            logger.info(f"Intent extraction response ({service_id}): {response_text[:100]}...")
            
            return {
                "text": response_text,
                "model": service_id,
                "service_id": service_id
            }
        except Exception as e:
            logger.error(f"Error in intent extraction: {str(e)}")
            return {
                "text": f"Error in intent extraction: {str(e)}",
                "model": "error",
                "service_id": service_id
            }
    
    async def generate_report(self, symptoms: str, diagnosis: str, confidence: str, verification: str, service_id: str = "full") -> Dict[str, Any]:
        """Generate a medical report.
        
        Args:
            symptoms: Patient symptoms as a string
            diagnosis: The diagnosis
            confidence: Confidence level as a string
            verification: Verification status
            service_id: The AI service to use
            
        Returns:
            Dictionary with generated report
        """
        try:
            kernel_args = KernelArguments(
                symptoms=symptoms,
                diagnosis=diagnosis,
                confidence=confidence,
                verification=verification
            )
            kernel_args.set_ai_service(service_id)
            
            result = await self.kernel.invoke_async(
                function_name="medical-generate_report",
                arguments=kernel_args
            )
            
            response_text = str(result)
            logger.info(f"Report generation response ({service_id}): {response_text[:100]}...")
            
            return {
                "text": response_text,
                "model": service_id,
                "service_id": service_id
            }
        except Exception as e:
            logger.error(f"Error in report generation: {str(e)}")
            return {
                "text": f"Error in report generation: {str(e)}",
                "model": "error",
                "service_id": service_id
            }
    
    # Prompt templates for medical functions
    def _get_diagnosis_prompt(self) -> str:
        return """Analyze the following symptoms and generate a diagnosis with confidence level.\n\n
Patient symptoms: {{$symptoms}}\n\n
Provide a diagnosis in JSON format with the following structure:\n
{{\n
  "diagnosis": {{\n
    "name": "[condition name]",\n
    "confidence": [0.0-1.0 as a float],\n
    "reasoning": "[explanation of diagnosis]"\n
  }},\n
  "differential_diagnoses": [\n
    {{\n
      "name": "[alternative condition 1]",\n
      "confidence": [0.0-1.0 as a float]\n
    }},\n
    {{\n
      "name": "[alternative condition 2]",\n
      "confidence": [0.0-1.0 as a float]\n
    }}\n
  ]\n
}}\n
If you cannot make a diagnosis with the provided symptoms, set the diagnosis name to null and confidence to 0.0."""
    
    def _get_followup_prompt(self) -> str:
        return """You are a medical professional gathering information about a patient's symptoms.\n\n
Current reported symptoms: {{$symptoms}}\n
Previously asked questions: {{$asked_questions}}\n\n
Generate ONE specific follow-up question that would provide the highest diagnostic value. Consider:\n
1. The specific nature and characteristics of the reported symptoms\n
2. Key differentiating factors that would help narrow down potential diagnoses\n
3. Important clinical indicators that haven't been asked about yet\n\n
Format: Return ONLY the question, without any prefixes or additional text."""
    
    def _get_verification_prompt(self) -> str:
        return """We have collected these symptoms:\n{{$symptoms}}\n\n
We have a tentative diagnosis of {{$diagnosis}} with confidence {{$confidence}}.\n
Please confirm if this diagnosis is correct, or refine it.\n 
Return a JSON object with "verification": "agree" or "disagree",\n
optionally "diagnosis_name" for a refined name,\n 
and "notes" for extra detail."""
    
    def _get_mitigations_prompt(self) -> str:
        return """Based on the diagnosis {{$diagnosis}} for a patient with symptoms: {{$symptoms}},\n\n
provide recommendations for managing this condition. Include:\n
1. Immediate self-care steps the patient can take\n
2. When they should seek professional medical care\n
3. Lifestyle modifications that may help\n
4. Any important warning signs to watch for\n\n
Format your response as clear, actionable recommendations in paragraph form."""
    
    def _get_symptom_extraction_prompt(self) -> str:
        return """Extract medical symptoms from the following user message:\n\n
User message: {{$message}}\n\n
Return a JSON array of symptoms. Each symptom should be a string describing a specific health issue mentioned.\n
Only include actual symptoms, not general statements, questions, or non-medical content.\n
If no symptoms are mentioned, return an empty array []."""
    
    def _get_intent_classification_prompt(self) -> str:
        return """You are an intent classifier for a medical assistant bot. 
Classify the user's message into one of these intents:\n
1. symptomReporting - User is reporting symptoms or health concerns\n
2. medicalInquiry - User is asking for medical information or advice\n
3. smallTalk - User is making small talk, casual conversation\n
4. greeting - User is greeting the bot\n
5. farewell - User is saying goodbye\n
6. confirm - User is confirming or agreeing with something\n
7. deny - User is denying or disagreeing with something\n
8. endConversation - User wants to end the conversation\n
9. emergency - User is reporting a medical emergency\n
10. out_of_scope - User's message is not related to health or cannot be processed\n\n
User message: {{$message}}\n
User context: {{$context}}\n\n
Return a JSON object with:\n
\"intent\": the most likely intent from the list above,\n
\"confidence\": a confidence score between 0.0-1.0\n
\"reasoning\": a brief explanation of why this intent was chosen"""
    
    def _get_intent_extraction_prompt(self) -> str:
        return """Extract the medical intents from this user message.\n\n
User message: {{$message}}\n
Previous conversation context: {{$context}}\n\n
Identify all possible intents in the message, focusing on:\n
1. Is the user reporting symptoms? If so, what symptoms?\n
2. Is the user asking for medical information? What specific information?\n
3. Is the user confirming or denying something in response to a previous question?\n
4. Is the user showing signs of a medical emergency?\n\n
Return a JSON object with:\n
\"primary_intent\": the main purpose of the message,\n
\"detected_intents\": an array of all detected intents,\n
\"extracted_entities\": any medical entities mentioned (symptoms, conditions, etc.)"""
    
    def _get_report_prompt(self) -> str:
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
