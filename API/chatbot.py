# Import necessary libraries
import os
import requests
from typing import List, Dict
from dotenv import load_dotenv

# Define the VirtualMechanicBot class
class VirtualMechanicBot:
    def __init__(self):
        """Initialize the Virtual Mechanic chatbot with Groq API integration."""
        # Load environment variables (kept for compatibility, though unused)
        load_dotenv()

        # Hardcoded API key (replace with your actual key)
        self.api_key = "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"
        if not self.api_key:
            print("Warning: No Groq API key provided.")

        # Groq API endpoint
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

        # Default model - llama3 or mixtral are good choices
        self.model = "llama3-70b-8192"

        # Conversation history
        self.conversation_history = []

        # System prompt that defines the chatbot's behavior
        self.system_prompt = """
        You are a Virtual Mechanic Assistant, an expert in automotive repair, maintenance, and diagnostics.

        Your capabilities include:
        1. Diagnosing vehicle problems based on symptoms
        2. Providing step-by-step repair instructions for common issues
        3. Estimating repair costs for various problems including dent repairs
        4. Recommending appropriate tools and parts for repairs
        5. Suggesting nearby repair shops (if location is provided)
        6. Explaining maintenance schedules and best practices
        7. Providing emergency roadside assistance instructions
        8. Helping with dent assessment and repair options

        When providing cost estimates for repairs or dent fixes, give realistic ranges based on:
        - Vehicle make, model, and year
        - Part costs and labor hours
        - Geographic location (if provided)
        - Severity of the issue

        For dent repairs specifically, consider:
        - Dent size, depth, and location
        - Whether paint is damaged
        - Accessibility of the dent
        - Best repair method (PDR, traditional, or panel replacement)

        Always prioritize safety in your recommendations. If an issue is potentially dangerous,
        advise the user to seek professional help rather than attempting DIY repairs.

        Respond in a helpful, knowledgeable manner while being concise and practical.
        """

    def add_message_to_history(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})

    def get_response(self, user_message: str) -> str:
        """
        Get a response from the Groq API based on the conversation history.

        Args:
            user_message: The user's message

        Returns:
            The assistant's response
        """
        # Add user message to history
        self.add_message_to_history("user", user_message)

        # Prepare messages for API call
        messages = [{"role": "system", "content": self.system_prompt}] + self.conversation_history

        try:
            # Call Groq API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 800
            }

            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Extract assistant's response
            assistant_response = response.json()["choices"][0]["message"]["content"]

            # Add assistant response to history
            self.add_message_to_history("assistant", assistant_response)

            return assistant_response

        except requests.exceptions.RequestException as e:
            error_message = f"Error communicating with Groq API: {str(e)}"
            print(error_message)
            return error_message

        except (KeyError, IndexError) as e:
            error_message = f"Error parsing Groq API response: {str(e)}"
            print(error_message)
            return error_message

# Define the CLI interface
def cli_interface():
    """Command-line interface for the chatbot."""
    bot = VirtualMechanicBot()
    print("Virtual Mechanic Assistant (Type 'exit' to quit)")
    print("Bot: Hello! I'm your Virtual Mechanic Assistant. How can I help you with your vehicle today?")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Bot: Thank you for using Virtual Mechanic Assistant. Drive safely!")
            break

        response = bot.get_response(user_input)
        print(f"Bot: {response}")

# Run the CLI interface
cli_interface()