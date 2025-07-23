import os
from dotenv import load_dotenv

# Get the absolute path to the project root
project_root = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(project_root, 'config.env')

print(f"Project root: {project_root}")
print(f"Looking for config file at: {config_path}")
print(f"Config file exists: {os.path.exists(config_path)}")

# Load environment variables
load_dotenv(dotenv_path=config_path)

# Check if the API key is loaded
api_key = os.getenv('GOOGLE_API_KEY')
print(f"\nAPI Key loaded: {'Yes' if api_key else 'No'}")
if api_key:
    print(f"API Key length: {len(api_key)} characters")
    print(f"Starts with: {api_key[:5]}...")
else:
    print("\nMake sure your config.env file contains a line like:")
    print("GOOGLE_API_KEY=your_actual_api_key_here")
    print("\nAnd that the file is in the project root directory.")
