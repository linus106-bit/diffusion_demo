"""
OpenAI model implementation for DiffuChatGPT
"""

import warnings
from typing import List, Dict, Any, Optional

from .base import BaseModel
from ..config import config
from ..utils.logger import app_logger

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class OpenAIModel(BaseModel):
    """OpenAI API model implementation"""
    
    def __init__(self, model_name: str = None, api_key: str = None, base_url: str = None):
        super().__init__(model_name or config.openai.default_model)
        self.api_key = api_key or config.openai.api_key
        self.base_url = base_url or config.openai.base_url
        self.client = None
        
        if self.api_key:
            self.load()
    
    def load(self) -> bool:
        """Load OpenAI client"""
        try:
            import openai
            
            # Try modern client first
            try:
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                app_logger.info("✅ OpenAI client loaded (modern)")
            except Exception as e:
                app_logger.warning(f"Modern client failed: {e}")
                # Fallback to legacy
                openai.api_key = self.api_key
                if self.base_url:
                    openai.api_base = self.base_url
                self.client = openai
                app_logger.info("✅ OpenAI client loaded (legacy)")
            
            self.is_loaded = True
            return True
            
        except ImportError:
            app_logger.error("❌ OpenAI library not installed")
            return False
        except Exception as e:
            app_logger.error(f"❌ Failed to load OpenAI client: {e}")
            return False
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """Generate response using OpenAI API"""
        if not self.is_loaded:
            return "Error: OpenAI client not loaded"
        
        try:
            max_tokens = max_tokens or config.openai.max_tokens
            temperature = temperature or config.openai.temperature
            
            # Convert messages to OpenAI format
            openai_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
            
            app_logger.info(f"🎯 OpenAI generating with {max_tokens} tokens, temp={temperature}")
            
            # Make API call
            if hasattr(self.client, 'chat'):
                # Modern client
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=openai_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False
                )
                result = response.choices[0].message.content
            else:
                # Legacy client
                response = self.client.ChatCompletion.create(
                    model=self.model_name,
                    messages=openai_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False
                )
                result = response.choices[0].message.content
            
            app_logger.info(f"🎯 OpenAI Response: {result[:100]}{'...' if len(result) > 100 else ''}")
            return result
            
        except Exception as e:
            error_msg = f"OpenAI generation failed: {str(e)}"
            app_logger.error(f"❌ {error_msg}")
            return error_msg
    
    def get_available_models(self) -> List[str]:
        """Get list of available OpenAI models"""
        if not self.is_loaded:
            return [self.model_name]
        
        try:
            if hasattr(self.client, 'models'):
                # Modern client
                models = self.client.models.list()
                return [model.id for model in models.data]
            else:
                # Legacy client
                models = self.client.Model.list()
                return [model.id for model in models.data]
        except Exception as e:
            app_logger.error(f"❌ Failed to fetch OpenAI models: {e}")
            return [self.model_name]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "status": "loaded" if self.is_loaded else "not_loaded",
            "model_name": self.model_name,
            "model_type": "OpenAI API",
            "available_models": self.get_available_models(),
            "api_key_set": bool(self.api_key),
            "base_url": self.base_url
        }
