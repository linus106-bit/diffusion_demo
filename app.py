from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Any

# Import local model generator
try:
    from generate import generate_local_response
    LOCAL_MODEL_AVAILABLE = True
    print("✅ Local model generator imported successfully")
except ImportError as e:
    LOCAL_MODEL_AVAILABLE = False
    print(f"❌ Local model not available: {e}")
    print("Install transformers and torch to use local models: pip install transformers torch")

# Import LLaDA
try:
    from llada import generate_llada_response
    LLADA_AVAILABLE = True
    print("✅ LLaDA imported successfully")
except ImportError as e:
    LLADA_AVAILABLE = False
    print(f"❌ LLaDA not available: {e}")
    print("Install transformers and torch to use LLaDA: pip install transformers torch")

# Import OpenAI with error handling
try:
    import openai
except ImportError as e:
    print(f"Failed to import openai: {e}")
    openai = None

app = FastAPI(title="AI Chat Demo", description="OpenAI GPT Powered Chat Interface")

# Setup logging for OpenAI messages
def setup_message_logging():
    """Setup logging to file for OpenAI messages"""
    log_filename = f"openai_messages_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Create a specific logger for OpenAI messages
    logger = logging.getLogger('openai_messages')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create simple formatter for from,content format
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    return logger

# Initialize message logger
message_logger = setup_message_logging()

def log_openai_messages(messages, request_type="CHAT"):
    """Log messages in from,content format and raw JSON to file"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message_logger.info(f"=== {request_type} REQUEST - {timestamp} ===")
    
    # Log in from,content format
    for msg in messages:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        # Replace newlines with \\n for single-line logging
        content_clean = content.replace('\n', '\\n').replace('\r', '\\r')
        message_logger.info(f"{role},{content_clean}")
    
    # Log raw JSON format
    message_logger.info("--- RAW JSON ---")
    message_logger.info(json.dumps(messages, ensure_ascii=False, indent=2))
    
    message_logger.info("=" * 50)

# Configure OpenAI client to use real OpenAI API
# Get your API key from https://platform.openai.com/api-keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Required for OpenAI API
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set!")
    print("Please set your OpenAI API key: export OPENAI_API_KEY=your_api_key_here")

# Initialize OpenAI client for real OpenAI API
def create_openai_client():
    """Create OpenAI client for real OpenAI API"""
    if openai is None:
        print("OpenAI library not available")
        return None
    
    if not OPENAI_API_KEY:
        print("OpenAI API key not provided")
        return None
    
    print(f"Creating OpenAI client...")
    print(f"OpenAI version: {getattr(openai, '__version__', 'unknown')}")
    
    try:
        # Try modern client first
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        print("✅ Successfully created modern OpenAI client")
        return client
    except Exception as e:
        # If modern client fails, use legacy configuration (more reliable)
        print(f"Modern client failed, using legacy configuration...")
        try:
            openai.api_key = OPENAI_API_KEY
            print("✅ Successfully configured legacy OpenAI client")
            return openai
        except Exception as e2:
            print(f"❌ All client initialization methods failed: {str(e2)}")
            return None

# Initialize client at startup
client = create_openai_client()
if client is None:
    print("Warning: Could not initialize OpenAI client. API calls will fail.")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Pydantic models for request validation
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "gpt-3.5-turbo"

class ChatResponse(BaseModel):
    success: bool
    message: str = None
    timestamp: str = None
    error: str = None

class ModelsResponse(BaseModel):
    success: bool
    models: List[str]
    error: str = None

@app.get("/", response_class=HTMLResponse)
async def openai_chat(request: Request):
    return templates.TemplateResponse("openai_chat.html", {
        "request": request,
        "active_tab": "openai"
    })

@app.get("/local", response_class=HTMLResponse)
async def local_chat(request: Request):
    return templates.TemplateResponse("local_chat.html", {
        "request": request,
        "active_tab": "local"
    })

@app.get("/llada", response_class=HTMLResponse)
async def llada_chat(request: Request):
    return templates.TemplateResponse("llada_chat.html", {
        "request": request,
        "active_tab": "llada"
    })

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    if client is None:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    
    try:
        # Convert Pydantic models to dict format for OpenAI client
        messages = [{"role": msg.role, "content": msg.content} for msg in chat_request.messages]
        
        # Log messages to file
        log_openai_messages(messages, "CHAT")
        
        # Make request to OpenAI API
        if hasattr(client, 'chat'):
            # New client object
            response = client.chat.completions.create(
                model=chat_request.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                stream=False
            )
        else:
            # Legacy module-level API
            response = openai.ChatCompletion.create(
                model=chat_request.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                stream=False
            )
        
        # Extract response content
        assistant_message = response.choices[0].message.content
        
        return ChatResponse(
            success=True,
            message=assistant_message,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/local", response_model=ChatResponse)
async def chat_local(chat_request: ChatRequest):
    """Chat endpoint for local Transformers model"""
    if not LOCAL_MODEL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Local model not available. Install transformers and torch.")
    
    try:
        # Convert Pydantic models to dict format for local model
        messages = [{"role": msg.role, "content": msg.content} for msg in chat_request.messages]
        
        # Log messages to file
        log_openai_messages(messages, "LOCAL_MODEL")
        
        # Generate response using local model
        assistant_message = generate_local_response(
            messages=messages,
            max_new_tokens=1000,
            temperature=0.7
        )
        
        return ChatResponse(
            success=True,
            message=assistant_message,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        print(f"Local model error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Local model error: {str(e)}")

@app.post("/chat/llada", response_model=ChatResponse)
async def chat_llada(chat_request: ChatRequest):
    """Chat endpoint for LLaDA (LLM as Diffusion Autoencoders)"""
    if not LLADA_AVAILABLE:
        raise HTTPException(status_code=503, detail="LLaDA not available. Install transformers and torch.")
    
    try:
        # Convert Pydantic models to dict format for LLaDA
        messages = [{"role": msg.role, "content": msg.content} for msg in chat_request.messages]
        
        # Log messages to file
        log_openai_messages(messages, "LLADA")
        
        # Generate response using LLaDA diffusion model
        # Note: chat_request.model contains the model selection, but LLaDA uses a fixed model
        assistant_message = generate_llada_response(
            messages=messages,
            max_new_tokens=128,  # LLaDA works better with shorter generations
            temperature=0.0,     # LLaDA uses temperature differently (for Gumbel noise)
            model_name="GSAI-ML/LLaDA-8B-Instruct"  # Fixed LLaDA model
        )
        
        return ChatResponse(
            success=True,
            message=assistant_message,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        print(f"LLaDA error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLaDA error: {str(e)}")

@app.post("/chat/stream")
async def chat_stream(chat_request: ChatRequest):
    """Streaming endpoint for real-time responses"""
    if client is None:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    
    try:
        # Convert Pydantic models to dict format for OpenAI client
        messages = [{"role": msg.role, "content": msg.content} for msg in chat_request.messages]
        
        # Log messages to file
        log_openai_messages(messages, "CHAT_STREAM")
        
        async def generate():
            try:
                # Make streaming request to OpenAI API
                if hasattr(client, 'chat'):
                    # New client object
                    response = client.chat.completions.create(
                        model=chat_request.model,
                        messages=messages,
                        max_tokens=1000,
                        temperature=0.7,
                        stream=True
                    )
                else:
                    # Legacy module-level API
                    response = openai.ChatCompletion.create(
                        model=chat_request.model,
                        messages=messages,
                        max_tokens=1000,
                        temperature=0.7,
                        stream=True
                    )
                
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        yield f"data: {json.dumps({'content': content})}\n\n"
                
                yield f"data: {json.dumps({'done': True})}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(generate(), media_type="text/plain")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", response_model=ModelsResponse)
async def get_models():
    """Get available models from OpenAI API"""
    if client is None:
        return ModelsResponse(
            success=False,
            error="OpenAI client not initialized",
            models=['gpt-3.5-turbo']  # Fallback
        )
    
    try:
        if hasattr(client, 'models'):
            # New client object
            models = client.models.list()
            model_list = [model.id for model in models.data]
        else:
            # Legacy module-level API
            models = openai.Model.list()
            model_list = [model.id for model in models.data]
        
        return ModelsResponse(
            success=True,
            models=model_list
        )
    except Exception as e:
        return ModelsResponse(
            success=False,
            error=str(e),
            models=['gpt-3.5-turbo']  # Fallback
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
