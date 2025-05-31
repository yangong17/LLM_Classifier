from dotenv import load_dotenv
import os
import openai

def check_api_key():
    # Load the API key from .env
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ No API key found in .env file")
        return
        
    print(f"Testing API key: {api_key[:6]}...{api_key[-4:]}")
    
    try:
        # Initialize the client
        client = openai.OpenAI(api_key=api_key)
        
        # Try to list models (this is a lightweight API call)
        client.models.list()
        
        print("✅ API key is valid!")
        
    except openai.APIConnectionError as e:
        print("❌ Connection Error: Check your internet connection")
        print(f"Error details: {str(e)}")
    except Exception as e:
        print("❌ Invalid API key or other error")
        print(f"Error details: {str(e)}")

if __name__ == '__main__':
    check_api_key() 