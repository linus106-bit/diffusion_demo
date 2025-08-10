"""
Base model interface for DiffuChatGPT
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


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
