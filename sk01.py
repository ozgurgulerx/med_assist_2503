import os
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.azure_openai import AzureOpenAIChatCompletionClient
from semantic_kernel.memory import InMemoryChatMemory

# Load environment variables from .env
load_dotenv()

# Extract credentials from environment variables
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

# Create a minimal Azure OpenAI client using the provided credentials
azure_client = AzureOpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_version="2024-06-01",
    azure_endpoint=azure_endpoint,
    api_key=api_key
)

# Initialize the Semantic Kernel
kernel = Kernel()

# Add the Azure OpenAI chat completion service to the kernel
kernel.add_chat_completion_service("azure", azure_client)

# Set up an in-memory chat history (memory)
chat_memory = InMemoryChatMemory()
kernel.memory = chat_memory

# A simple interactive chat loop with memory support
print("Semantic Kernel Chatbot with Memory. Type 'exit' to quit.")
while True:
    user_input = input("User > ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Add the user's message to memory
    kernel.memory.add_message("user", user_input)
    
    # Generate a response using the chat completion service and the current conversation history
    response = kernel.chat_completion("azure", memory=kernel.memory)
    
    print("Assistant >", response)
    
    # Add the assistant's response to memory
    kernel.memory.add_message("assistant", response)
