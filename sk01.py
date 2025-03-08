import os
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureOpenAIChatCompletionClient
from semantic_kernel.memory import VolatileMemoryStore
from semantic_kernel.core_plugins import TimePlugin, MathPlugin
from semantic_kernel.planning import ActionPlanner
from semantic_kernel.prompt_template import PrompTemplateConfig, PromptTemplate

# Load environment variables from .env
load_dotenv()

# Create a minimal Azure OpenAI client
azure_client = AzureOpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_version="2024-06-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

# Initialize the semantic kernel
kernel = Kernel()
kernel.add_chat_service("azure_chat_completion", azure_client)

# Add some core plugins
kernel.import_plugin(TimePlugin(), plugin_name="time")
kernel.import_plugin(MathPlugin(), plugin_name="math")

# Create a memory store for conversation history
memory_store = VolatileMemoryStore()
kernel.register_memory_store(memory_store=memory_store)

# Create a chat memory plugin for context management
class ChatMemoryPlugin:
    def __init__(self):
        self.conversation_history = []
    
    def add_message(self, role, content):
        self.conversation_history.append({"role": role, "content": content})
        return "Message added to history"
    
    def get_history(self):
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_history])
    
    def clear_history(self):
        self.conversation_history.clear()
        return "Conversation history cleared"

# Register the chat memory plugin
chat_plugin = ChatMemoryPlugin()
kernel.import_plugin_from_object(chat_plugin, plugin_name="chat")

# Define a basic prompt template for chat
chat_prompt_config = PrompTemplateConfig(
    template="You are a helpful assistant.\n\nConversation history:\n{{$history}}\n\nUser: {{$input}}\nAssistant:",
    input_variables=["history", "input"]
)

chat_function = kernel.create_function_from_prompt(
    plugin_name="dialog",
    function_name="chat",
    prompt_template=PromptTemplate(
        template=chat_prompt_config.template,
        prompt_config=chat_prompt_config
    )
)

# Create a simple dialog manager
class DialogManager:
    def __init__(self, kernel, chat_function, chat_plugin):
        self.kernel = kernel
        self.chat_function = chat_function
        self.chat_plugin = chat_plugin
        self.planner = ActionPlanner(kernel)
    
    async def process_message(self, user_message):
        # Add user message to history
        self.chat_plugin.add_message("user", user_message)
        
        # Get conversation history
        history = self.chat_plugin.get_history()
        
        # Process the message using the chat function
        response = await self.kernel.invoke(
            self.chat_function,
            {"history": history, "input": user_message}
        )
        
        # Add assistant response to history
        self.chat_plugin.add_message("assistant", str(response))
        
        return str(response)
    
    def clear_conversation(self):
        return self.chat_plugin.clear_history()

# Example usage
dialog_manager = DialogManager(kernel, chat_function, chat_plugin)

# Main loop for interactive chat
async def main():
    print("Bot: Hello! I'm a bot powered by Semantic Kernel. Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Bot: Goodbye!")
            break
        
        response = await dialog_manager.process_message(user_input)
        print(f"Bot: {response}")

# Run the interactive chat
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())