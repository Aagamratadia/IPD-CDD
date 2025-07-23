import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Path to your car repair guide PDF in Colab
PDF_PATH = "Teen Driver Car Maintencance Guide.pdf"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

# Function to create FAISS index
def create_faiss_index(text):
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Use HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create FAISS index
    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=[{"source": f"chunk_{i}"} for i in range(len(chunks))]
    )
    
    # Save FAISS index
    vector_store.save_local("faiss_index")
    print("FAISS index created and saved as 'faiss_index'.")
    return vector_store

# Main function
def main():
    print("Creating FAISS index from car repair guide PDF...")
    
    # Extract text from PDF
    pdf_text = extract_text_from_pdf(PDF_PATH)
    if "Error" in pdf_text:
        print(pdf_text)
        return
    
    # Create and save FAISS index
    create_faiss_index(pdf_text)

# Run the script
if __name__ == "__main__":
    main()

