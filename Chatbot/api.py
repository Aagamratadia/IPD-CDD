from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from API.chatbot import VirtualMechanicBot

app = FastAPI(
    title="Virtual Mechanic API",
    description="API for the Virtual Mechanic chatbot that provides automotive repair and maintenance assistance",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files
app.mount("/static", StaticFiles(directory="Chatbot/static"), name="static")

# Initialize the bot
bot = VirtualMechanicBot()

class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """
    Send a message to the Virtual Mechanic bot and get a response.
    
    Args:
        message: The user's message
        
    Returns:
        The bot's response
    """
    try:
        response = bot.get_response(message.message)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """
    Root endpoint that returns basic API information.
    """
    return {
        "name": "Virtual Mechanic API",
        "version": "1.0.0",
        "description": "API for automotive repair and maintenance assistance",
        "endpoints": {
            "/chat": "POST - Send a message to the Virtual Mechanic bot",
            "/": "GET - API information"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6000) 