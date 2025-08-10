"""
API routes for DiffuChatGPT
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import json
from typing import List, Dict, Any

from .models import (
    ChatRequest, ChatResponse, ModelsResponse, 
    ModelInfoResponse, HealthResponse
)
from ..models.base import ModelManager
from ..utils.logger import app_logger, log_chat_messages
from .. import __version__

# Create router
router = APIRouter()

# Setup templates
templates = Jinja2Templates(directory="src/diffusion_demo/templates")

# Global model manager
model_manager = ModelManager()


@router.get("/", response_class=HTMLResponse)
async def openai_chat(request: Request):
    """OpenAI chat interface"""
    return templates.TemplateResponse("openai_chat.html", {
        "request": request,
        "active_tab": "openai"
    })


@router.get("/autoregressive", response_class=HTMLResponse)
async def autoregressive_chat(request: Request):
    """Autoregressive model chat interface"""
    return templates.TemplateResponse("autoregressive_chat.html", {
        "request": request,
        "active_tab": "autoregressive"
    })


@router.get("/llada", response_class=HTMLResponse)
async def llada_chat(request: Request):
    """LLaDA chat interface"""
    return templates.TemplateResponse("llada_chat.html", {
        "request": request,
        "active_tab": "llada"
    })


@router.post("/chat", response_model=ChatResponse)
async def chat_openai(chat_request: ChatRequest):
    """Chat endpoint for OpenAI models"""
    openai_model = model_manager.get_model("openai")
    if not openai_model or not openai_model.is_available():
        raise HTTPException(status_code=503, detail="OpenAI model not available")
    
    try:
        # Convert Pydantic models to dict format
        messages = [{"role": msg.role, "content": msg.content} for msg in chat_request.messages]
        
        # Log messages
        log_chat_messages(messages, "OPENAI")
        
        # Generate response
        response = openai_model.generate_response(
            messages=messages,
            max_tokens=chat_request.max_tokens,
            temperature=chat_request.temperature
        )
        
        return ChatResponse(
            success=True,
            message=response
        )
        
    except Exception as e:
        app_logger.error(f"OpenAI chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/autoregressive", response_model=ChatResponse)
async def chat_autoregressive(chat_request: ChatRequest):
    """Chat endpoint for autoregressive models"""
    autoregressive_model = model_manager.get_model("autoregressive")
    if not autoregressive_model or not autoregressive_model.is_available():
        raise HTTPException(status_code=503, detail="Autoregressive model not available")
    
    try:
        # Convert Pydantic models to dict format
        messages = [{"role": msg.role, "content": msg.content} for msg in chat_request.messages]
        
        # Log messages
        log_chat_messages(messages, "AUTOREGRESSIVE_MODEL")
        
        # Generate response
        response = autoregressive_model.generate_response(
            messages=messages,
            max_tokens=chat_request.max_tokens,
            temperature=chat_request.temperature
        )
        
        return ChatResponse(
            success=True,
            message=response
        )
        
    except Exception as e:
        app_logger.error(f"Autoregressive model chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/llada", response_model=ChatResponse)
async def chat_llada(chat_request: ChatRequest):
    """Chat endpoint for LLaDA models"""
    llada_model = model_manager.get_model("llada")
    if not llada_model or not llada_model.is_available():
        raise HTTPException(status_code=503, detail="LLaDA model not available")
    
    try:
        # Convert Pydantic models to dict format
        messages = [{"role": msg.role, "content": msg.content} for msg in chat_request.messages]
        
        # Log messages
        log_chat_messages(messages, "LLADA")
        
        # Generate response
        response = llada_model.generate_response(
            messages=messages,
            max_tokens=chat_request.max_tokens,
            temperature=chat_request.temperature
        )
        
        return ChatResponse(
            success=True,
            message=response
        )
        
    except Exception as e:
        app_logger.error(f"LLaDA chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=ModelsResponse)
async def get_models():
    """Get available models from OpenAI API"""
    openai_model = model_manager.get_model("openai")
    if not openai_model or not openai_model.is_available():
        return ModelsResponse(
            success=False,
            error="OpenAI model not available",
            models=['gpt-3.5-turbo']  # Fallback
        )
    
    try:
        models = openai_model.get_available_models()
        return ModelsResponse(
            success=True,
            models=models
        )
    except Exception as e:
        app_logger.error(f"Failed to fetch models: {e}")
        return ModelsResponse(
            success=False,
            error=str(e),
            models=['gpt-3.5-turbo']  # Fallback
        )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_status = {}
    
    for model_name in ["openai", "autoregressive", "llada"]:
        model = model_manager.get_model(model_name)
        models_status[model_name] = {
            "available": model.is_available() if model else False,
            "info": model.get_model_info() if model else {"status": "not_loaded"}
        }
    
    return HealthResponse(
        status="healthy",
        version=__version__,
        models=models_status
    )


@router.get("/model-info/{model_name}", response_model=ModelInfoResponse)
async def get_model_info(model_name: str):
    """Get information about a specific model"""
    model = model_manager.get_model(model_name)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    return ModelInfoResponse(
        success=True,
        model_info=model.get_model_info()
    )
