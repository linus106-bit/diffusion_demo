"""
Base model interface for DiffuChatGPT
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json
from ..utils.logger import app_logger


class BaseModel(ABC):
    """Abstract base class for all AI models"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.is_loaded = False
        self.device = "cpu"
    
    @abstractmethod
    def load(self) -> bool:
        """Load the model"""
        pass
    
    def log_messages_before_generation(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> None:
        """Log messages and parameters before generation"""
        try:
            # Log the model type and parameters
            app_logger.info(f"🎯 [{self.__class__.__name__}] Starting generation")
            app_logger.info(f"📊 Model: {self.model_name}")
            app_logger.info(f"⚙️ Parameters: max_tokens={max_tokens}, temperature={temperature}")
            if kwargs:
                app_logger.info(f"🔧 Additional kwargs: {kwargs}")
            
            # Log the messages in a readable format
            app_logger.info(f"💬 Input messages ({len(messages)} messages):")
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                # Truncate long content for readability
                content_preview = content[:200] + "..." if len(content) > 200 else content
                app_logger.info(f"  {i+1}. [{role.upper()}]: {content_preview}")
            
            # Log the full messages as JSON for debugging
            app_logger.debug(f"📋 Full messages JSON: {json.dumps(messages, ensure_ascii=False, indent=2)}")
            
        except Exception as e:
            app_logger.error(f"❌ Error logging messages: {e}")
    
    @abstractmethod
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate response from messages"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass
    
    def is_available(self) -> bool:
        """Check if model is available"""
        return self.is_loaded
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.model_name})"
    
    def __repr__(self) -> str:
        return self.__str__()


class ModelManager:
    """Manager for multiple model instances"""
    
    def __init__(self):
        self.models: Dict[str, BaseModel] = {}
    
    def register_model(self, name: str, model: BaseModel) -> None:
        """Register a model"""
        self.models[name] = model
    
    def get_model(self, name: str) -> Optional[BaseModel]:
        """Get a model by name"""
        return self.models.get(name)
    
    def list_models(self) -> List[str]:
        """List all available models"""
        return list(self.models.keys())
    
    def get_available_models(self) -> List[str]:
        """Get list of loaded models"""
        return [name for name, model in self.models.items() if model.is_available()]
