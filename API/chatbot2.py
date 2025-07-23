import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables from config.env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.env'))

# Configure Gemini API
API_KEY = os.getenv('GOOGLE_API_KEY')
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=API_KEY)

# Path to FAISS index
FAISS_INDEX_PATH = "faiss_index"

# Function to load FAISS index
def load_faiss_index():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        return vector_store
    except Exception as e:
        return f"Error loading FAISS index: {str(e)}"

# Function to query Gemini with RAG
def query_gemini_rag(question, vector_store):
    try:
        # Retrieve relevant chunks
        docs = vector_store.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Define prompt for Gemini with strict formatting requirements
        prompt = f"""
You are a vehicular repair assistant for DIY users who may be stuck in remote locations.

IMPORTANT FORMATTING RULES:
1. Start your response with a main heading using # (e.g., "# Emergency Tire Repair Guide")
2. List ALL steps as bullet points using • symbol
3. Each bullet point should be on a new line
4. Keep steps concise and clear
5. Include safety warnings where necessary

Example format:
# Emergency Tire Repair Guide
• First, ensure your car is parked safely
• Locate your spare tire and jack
• Loosen the lug nuts before jacking up the car
• Use the jack to lift the car
• Remove the flat tire and install the spare
• Tighten the lug nuts in a star pattern
• Lower the car and double-check the lug nuts

*Car Repair Guide Content*:
{context}

*User Query*: {question}
"""
        # Initialize Gemini model
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Generate response
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Ensure proper formatting
        if not response_text.startswith('#'):
            response_text = f"# Car Repair Guide\n{response_text}"
        
        # Convert any numbered lists or other formats to bullet points
        lines = response_text.split('\n')
        formatted_lines = []
        for line in lines:
            if line.strip() and not line.startswith('#'):
                if line.strip()[0].isdigit() or line.strip().startswith('-') or line.strip().startswith('*'):
                    line = '• ' + line.lstrip('0123456789.-* ')
                elif not line.strip().startswith('•'):
                    line = '• ' + line.strip()
            formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    except Exception as e:
        return f"Error querying Gemini: {str(e)}"

# Main function
def main():
    print("DIY Car Repair Chatbot with RAG")
    print("Loading FAISS index...")
    
    # Load FAISS index
    vector_store = load_faiss_index()
    if isinstance(vector_store, str):
        print(vector_store)
        return
    
    print("\nEnter your car issue (e.g., 'My car tire is punctured, and I'm stuck in the middle of nowhere.')")
    print("Type 'exit' to quit.")
    
    while True:
        question = input("\nYour query: ")
        if question.lower() == "exit":
            print("Exiting chatbot.")
            break
        if not question.strip():
            print("Please enter a valid query.")
            continue
        
        print("\nResponse:")
        response = query_gemini_rag(question, vector_store)
        print(response)

# Run the chatbot
if __name__ == "__main__":
    main()
