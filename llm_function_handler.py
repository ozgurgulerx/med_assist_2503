import os
import json
import logging
from typing import Dict, Any, List, Optional, Callable

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions.kernel_function import KernelFunction
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.function_result import FunctionResult
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

logger = logging.getLogger(__name__)

class LLMFunctionHandler:
    """
    Enhanced LLM handler that uses Semantic Kernel's function calling capabilities
    for medical assistant tasks.
    """

    def __init__(self):
        """Initialize the LLM function handler using Semantic Kernel."""
        # Create a Kernel instance
        self.kernel = Kernel()
        
        # Configure execution settings
        self.execution_settings = AzureChatPromptExecutionSettings()
        self.execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

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
            self.kernel.add_service(mini_service, service_id="mini")
            self.kernel.add_service(full_service, service_id="full")
            self.kernel.add_service(verifier_service, service_id="verifier")
            
            logger.info("Successfully registered 'mini', 'full', and 'verifier' chat services.")

        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI services: {str(e)}")

    def register_medical_functions(self):
        """Register semantic functions for medical tasks."""
        # Create the medical plugin
        medical_plugin = self.kernel.create_plugin("medical", "Medical assistant functions")
        
        # Define semantic functions
        
        # Function for diagnosis generation
        self.kernel.add_function_to_plugin(
            plugin=medical_plugin,
            function=self.kernel.create_semantic_function(
                prompt="""Analyze the following symptoms and generate a diagnosis with confidence level.\n\n
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
                If you cannot make a diagnosis with the provided symptoms, set the diagnosis name to null and confidence to 0.0.
                """,
                function_name="generate_diagnosis",
                description="Generates a diagnosis based on patient symptoms"
            )
        )
        
        # Function for follow-up question generation
        self.kernel.add_function_to_plugin(
            plugin=medical_plugin,
            function=self.kernel.create_semantic_function(
                prompt="""You are a medical professional gathering information about a patient's symptoms.\n\n
                Current reported symptoms: {{$symptoms}}\n
                Previously asked questions: {{$asked_questions}}\n\n
                Generate ONE specific follow-up question that would provide the highest diagnostic value. Consider:\n
                1. The specific nature and characteristics of the reported symptoms\n
                2. Key differentiating factors that would help narrow down potential diagnoses\n
                3. Important clinical indicators that haven't been asked about yet\n\n
                Format: Return ONLY the question, without any prefixes or additional text.
                """,
                function_name="generate_followup_question",
                description="Generates a follow-up question to gather more symptom information"
            )
        )
        
        # Function for high-confidence verification with O1 model
        self.kernel.add_function_to_plugin(
            plugin=medical_plugin,
            function=self.kernel.create_semantic_function(
                prompt="""We have collected these symptoms:\n{{$symptoms}}\n\n
                We have a tentative diagnosis of {{$diagnosis}} with confidence {{$confidence}}.\n
                Please confirm if this diagnosis is correct, or refine it.\n 
                Return a JSON object with "verification": "agree" or "disagree",\n
                optionally "diagnosis_name" for a refined name,\n 
                and "notes" for extra detail.
                """,
                function_name="verify_high_confidence_diagnosis",
                description="Verifies a high-confidence diagnosis using the O1 model"
            ),
            ai_service_id="verifier"
        )
        
        # Function for low-confidence explanation with O1 model
        self.kernel.add_function_to_plugin(
            plugin=medical_plugin,
            function=self.kernel.create_semantic_function(
                prompt="""We have collected these symptoms:\n{{$symptoms}}\n\n
                After multiple follow-up questions, our diagnostic confidence remains low at {{$confidence}}.\n\n
                As a medical expert, please provide:\n
                1. A narrative paragraph explaining why these symptoms may not fit clearly into a known condition with high confidence\n
                2. A list of possible conditions these symptoms could indicate\n
                3. A clear recommendation to consult with a medical professional, including what type of specialist might be appropriate\n
                4. Any urgent warning signs the patient should watch for\n\n
                Format your response as natural language paragraphs, not as JSON.
                """,
                function_name="explain_low_confidence",
                description="Explains why symptoms don't clearly match a known condition when confidence is low"
            ),
            ai_service_id="verifier"
        )
        
        # Function for suggesting mitigations based on diagnosis
        self.kernel.add_function_to_plugin(
            plugin=medical_plugin,
            function=self.kernel.create_semantic_function(
                prompt="""Based on the diagnosis {{$diagnosis}} for a patient with symptoms: {{$symptoms}},\n\n
                provide recommendations for managing this condition. Include:\n
                1. Immediate self-care steps the patient can take\n
                2. When they should seek professional medical care\n
                3. Lifestyle modifications that may help\n
                4. Any important warning signs to watch for\n\n
                Format your response as clear, actionable recommendations in paragraph form.
                """,
                function_name="suggest_mitigations",
                description="Suggests mitigations and management steps for a diagnosed condition"
            )
        )

        # Function for extracting symptoms from user messages
        self.kernel.add_function_to_plugin(
            plugin=medical_plugin,
            function=self.kernel.create_semantic_function(
                prompt="""Extract medical symptoms from the following user message:\n\n
                User message: {{$message}}\n\n
                Return a JSON array of symptoms. Each symptom should be a string describing a specific health issue mentioned.\n
                Only include actual symptoms, not general statements, questions, or non-medical content.\n
                If no symptoms are mentioned, return an empty array [].
                """,
                function_name="extract_symptoms",
                description="Extracts symptoms from a user message"
            )
        )
        
        logger.info("Successfully registered medical assistant functions")

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
            # Prepare function parts (plugin.function)
            plugin_name, func_name = function_name.split('.') if '.' in function_name else ("medical", function_name)
            
            # Get the function from the kernel
            function = self.kernel.get_plugin(plugin_name).get_function(func_name)
            
            # Create kernel arguments from dictionary
            kernel_args = KernelArguments(**arguments)
            
            # Set the AI service to use
            kernel_args.ai_service_id = service_id
            
            # Execute the function
            logger.info(f"Invoking function {function_name} with arguments: {arguments}")
            result: FunctionResult = await self.kernel.invoke(function, kernel_args)
            
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
            # Get the service
            service = self.kernel.get_service(service_id)
            if not service:
                return {
                    "text": f"No service available with ID: {service_id}",
                    "model": "None",
                    "service_id": service_id
                }
            
            # Configure execution settings
            settings = AzureChatPromptExecutionSettings()
            
            # Only set temperature for models that support it
            if service_id == "mini":
                settings.temperature = temperature
                
            # Create a temporary chat history
            chat_history = ChatHistory()
            chat_history.add_user_message(prompt)
            
            # Get response
            logger.info(f"Executing chat prompt with service {service_id}: {prompt[:100]}...")
            result = await service.get_chat_message_content(
                chat_history=chat_history,
                settings=settings,
                kernel=self.kernel
            )
            
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
