"""
API models for DiffuChatGPT
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class Message(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request model"""
    messages: List[Message] = Field(..., description="Conversation history")
    model: str = Field("gpt-3.5-turbo", description="Model to use for generation")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, description="Generation temperature")


class ChatResponse(BaseModel):
    """Chat response model"""
    success: bool = Field(..., description="Whether the request was successful")
    message: Optional[str] = Field(None, description="Generated response")
    error: Optional[str] = Field(None, description="Error message if failed")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ModelsResponse(BaseModel):
    """Models list response model"""
    success: bool = Field(..., description="Whether the request was successful")
    models: List[str] = Field(..., description="List of available models")
    error: Optional[str] = Field(None, description="Error message if failed")


class ModelInfoResponse(BaseModel):
    """Model information response model"""
    success: bool = Field(..., description="Whether the request was successful")
    model_info: dict = Field(..., description="Model information")
    error: Optional[str] = Field(None, description="Error message if failed")


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    models: dict = Field(..., description="Model availability status")
