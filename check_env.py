import os

print("Environment Variables:")
for key, value in os.environ.items():
    if 'GOOGLE' in key or 'API' in key:
        print(f"{key}: {'*' * 8}{value[-4:] if value else 'None'}")  # Show only last 4 chars of sensitive data
