import os
import json
import logging
from typing import Dict, Any, List, Optional

from semantic_kernel import Kernel
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig
from semantic_kernel.functions.kernel_function import KernelFunction

logger = logging.getLogger(__name__)

class LLMFunctionHandler:
    """
    Enhanced LLM handler that uses Semantic Kernel's function calling capabilities
    for medical assistant tasks, compatible with Semantic Kernel 1.23.1.
    """

    def __init__(self):
        """Initialize the LLM function handler using Semantic Kernel."""
        # Create a Kernel instance
        self.kernel = Kernel()
        
        # Setup AI services
        self.setup_chat_services()
        
        # Register medical assistant functions
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
        """Register semantic functions for medical tasks."""
        try:
            # Register diagnosis generation function
            self._register_function(
                function_name="medical-generate_diagnosis",
                description="Generates a diagnosis based on patient symptoms",
                prompt=self._get_diagnosis_prompt()
            )
            
            # Register follow-up question generation function
            self._register_function(
                function_name="medical-generate_followup_question",
                description="Generates a follow-up question to gather more symptom information",
                prompt=self._get_followup_prompt()
            )
            
            # Register verification function
            self._register_function(
                function_name="medical-verify_high_confidence_diagnosis",
                description="Verifies a high-confidence diagnosis using the O1 model",
                prompt=self._get_verification_prompt()
            )
            
            # Register mitigation function
            self._register_function(
                function_name="medical-suggest_mitigations",
                description="Suggests mitigations and management steps for a diagnosed condition",
                prompt=self._get_mitigations_prompt()
            )

            # Register symptom extraction function
            self._register_function(
                function_name="medical-extract_symptoms",
                description="Extracts symptoms from a user message",
                prompt=self._get_symptom_extraction_prompt()
            )
            
            # Register intent classification function
            self._register_function(
                function_name="intent-classify_intent",
                description="Classifies user messages into predefined intents",
                prompt=self._get_intent_classification_prompt()
            )
            
            # Register intent extraction function
            self._register_function(
                function_name="intent-extract_intents",
                description="Extracts multiple intents and entities from user messages",
                prompt=self._get_intent_extraction_prompt()
            )
            
            # Register report generation function
            self._register_function(
                function_name="conversation-generate_report",
                description="Generates a medical report summarizing the patient interaction",
                prompt=self._get_report_prompt()
            )
            
            logger.info("Successfully registered medical assistant functions")
            
        except Exception as e:
            logger.error(f"Error registering medical functions: {str(e)}")
            
    def _register_function(self, function_name: str, description: str, prompt: str):
        """Register a semantic function with the kernel."""
        # Create the prompt config
        prompt_config = PromptTemplateConfig(
            template=prompt,
            name=function_name,
            description=description
        )
        
        # Create and register the function
        function = self.kernel.create_function_from_prompt(
            prompt=prompt,
            function_name=function_name,
            description=description
        )
        
        # Register the function with the kernel
        self.kernel.add_function(function)

    def _get_diagnosis_prompt(self):
        return """Analyze the following symptoms and generate a diagnosis with confidence level.\n\n
Patient symptoms: {{$symptoms}}\n\n
Provide a diagnosis in JSON format with the following structure:\n
{\n
  "diagnosis": {\n
    "name": "[condition name]",\n
    "confidence": [0.0-1.0 as a float],\n
    "reasoning": "[explanation of diagnosis]"\n
  },\n
  "differential_diagnoses": [\n
    {\n
      "name": "[alternative condition 1]",\n
      "confidence": [0.0-1.0 as a float]\n
    },\n
    {\n
      "name": "[alternative condition 2]",\n
      "confidence": [0.0-1.0 as a float]\n
    }\n
  ]\n
}\n
If you cannot make a diagnosis with the provided symptoms, set the diagnosis name to null and confidence to 0.0."""

    def _get_followup_prompt(self):
        return """You are a medical professional gathering information about a patient's symptoms.\n\n
Current reported symptoms: {{$symptoms}}\n
Previously asked questions: {{$asked_questions}}\n\n
Generate ONE specific follow-up question that would provide the highest diagnostic value. Consider:\n
1. The specific nature and characteristics of the reported symptoms\n
2. Key differentiating factors that would help narrow down potential diagnoses\n
3. Important clinical indicators that haven't been asked about yet\n\n
Format: Return ONLY the question, without any prefixes or additional text."""

    def _get_verification_prompt(self):
        return """We have collected these symptoms:\n{{$symptoms}}\n\n
We have a tentative diagnosis of {{$diagnosis}} with confidence {{$confidence}}.\n
Please confirm if this diagnosis is correct, or refine it.\n 
Return a JSON object with "verification": "agree" or "disagree",\n
optionally "diagnosis_name" for a refined name,\n 
and "notes" for extra detail."""

    def _get_mitigations_prompt(self):
        return """Based on the diagnosis {{$diagnosis}} for a patient with symptoms: {{$symptoms}},\n\n
provide recommendations for managing this condition. Include:\n
1. Immediate self-care steps the patient can take\n
2. When they should seek professional medical care\n
3. Lifestyle modifications that may help\n
4. Any important warning signs to watch for\n\n
Format your response as clear, actionable recommendations in paragraph form."""

    def _get_symptom_extraction_prompt(self):
        return """Extract medical symptoms from the following user message:\n\n
User message: {{$message}}\n\n
Return a JSON array of symptoms. Each symptom should be a string describing a specific health issue mentioned.\n
Only include actual symptoms, not general statements, questions, or non-medical content.\n
If no symptoms are mentioned, return an empty array []."""
    
    def _get_intent_classification_prompt(self):
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
    
    def _get_intent_extraction_prompt(self):
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
    
    async def invoke_function(self, function_name: str, arguments: Dict[str, Any], 
                            service_id: str = "mini") -> Dict[str, Any]:
        """Invoke a registered semantic function with arguments.
        
        Args:
            function_name: The name of the function to invoke (format: 'plugin_name.function_name')
            arguments: Dictionary of arguments to pass to the function
            service_id: The AI service to use for function execution ('mini', 'full', or 'verifier')
            
        Returns:
            Dictionary with function result and metadata
        """
        try:
            # Convert to proper function name format if using plugin.function notation
            sk_function_name = function_name.replace(".", "-")
            
            # Create the kernel arguments
            kernel_args = KernelArguments(**arguments)
            
            # Set the AI service to use
            kernel_args.set_ai_service(service_id)
            
            # Execute the function
            logger.info(f"Invoking function {function_name} with arguments: {arguments}")
            result = await self.kernel.invoke_async(sk_function_name, kernel_args)
            
            # Extract result and return
            response_text = str(result)
            
            logger.info(f"Function {function_name} result: {response_text[:100]}...")
            return {
                "text": response_text,
                "function": function_name,
                "model": service_id,
                "args": arguments
            }
            
        except Exception as e:
            logger.error(f"Error invoking function {function_name}: {str(e)}")
            return {
                "text": f"Error in function execution: {str(e)}",
                "function": function_name,
                "model": "error",
                "args": arguments
            }

    async def execute_chat_prompt(self, prompt: str, service_id: str = "mini", 
                                temperature: float = 0.7) -> Dict[str, Any]:
        """Execute a direct chat prompt without function calling.
        
        Args:
            prompt: The input text
            service_id: The AI service to use ('mini', 'full', or 'verifier')
            temperature: Generation temperature (only applies to some models)
            
        Returns:
            Dictionary with response text and metadata
        """
        try:
            # Set up chat prompt as a simple text completion
            kernel_args = KernelArguments(input=prompt)
            kernel_args.set_ai_service(service_id)
            
            # Configure temperature if applicable
            if temperature != 0.7:
                kernel_args["temperature"] = temperature
            
            # Send prompt directly as a string
            logger.info(f"Executing chat prompt with service {service_id}: {prompt[:100]}...")
            result = await self.kernel.invoke_async(prompt, kernel_args)
            
            response_text = str(result)
            logger.info(f"Chat response ({service_id}): {response_text[:100]}...")
            
            return {
                "text": response_text,
                "model": service_id,
                "service_id": service_id
            }
            
        except Exception as e:
            logger.error(f"Error in chat prompt execution: {str(e)}")
            return {
                "text": f"Error in chat processing: {str(e)}",
                "model": "error",
                "service_id": service_id
            }
