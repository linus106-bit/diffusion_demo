"""
Configuration settings for DiffuChatGPT
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class OpenAIConfig:
    """OpenAI API configuration"""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    default_model: str = "gpt-3.5-turbo"
    max_tokens: int = 1000
    temperature: float = 0.7


@dataclass
class AutoregressiveModelConfig:
    """Autoregressive model configuration"""
    default_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    max_tokens: int = 1000
    temperature: float = 0.7
    device: Optional[str] = None  # Auto-detect if None


@dataclass
class LLaDAConfig:
    """LLaDA model configuration"""
    default_model: str = "GSAI-ML/LLaDA-8B-Instruct"
    max_tokens: int = 128
    temperature: float = 0.0
    steps: int = 128
    block_length: int = 32
    cfg_scale: float = 0.0
    remasking: str = 'low_confidence'


@dataclass
class ServerConfig:
    """Server configuration"""
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    reload: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "logs/diffusion_demo.log"
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.openai = OpenAIConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            default_model=os.getenv("OPENAI_DEFAULT_MODEL", "gpt-3.5-turbo"),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "1000")),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        )
        
        self.autoregressive_model = AutoregressiveModelConfig(
            default_model=os.getenv("AUTOREGRESSIVE_MODEL_NAME", "HuggingFaceTB/SmolLM2-135M-Instruct"),
            max_tokens=int(os.getenv("AUTOREGRESSIVE_MAX_TOKENS", "1000")),
            temperature=float(os.getenv("AUTOREGRESSIVE_TEMPERATURE", "0.7")),
            device=os.getenv("AUTOREGRESSIVE_DEVICE")
        )
        
        self.llada = LLaDAConfig(
            default_model=os.getenv("LLADA_MODEL_NAME", "GSAI-ML/LLaDA-8B-Instruct"),
            max_tokens=int(os.getenv("LLADA_MAX_TOKENS", "128")),
            temperature=float(os.getenv("LLADA_TEMPERATURE", "0.0")),
            steps=int(os.getenv("LLADA_STEPS", "128")),
            block_length=int(os.getenv("LLADA_BLOCK_LENGTH", "32")),
            cfg_scale=float(os.getenv("LLADA_CFG_SCALE", "0.0")),
            remasking=os.getenv("LLADA_REMASKING", "low_confidence")
        )
        
        self.server = ServerConfig(
            host=os.getenv("SERVER_HOST", "0.0.0.0"),
            port=int(os.getenv("SERVER_PORT", "8080")),
            debug=os.getenv("SERVER_DEBUG", "false").lower() == "true",
            reload=os.getenv("SERVER_RELOAD", "true").lower() == "true"
        )
        
        self.logging = LoggingConfig(
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            log_file=os.getenv("LOG_FILE", "logs/diffusion_demo.log"),
            max_log_size=int(os.getenv("LOG_MAX_SIZE", str(10 * 1024 * 1024))),
            backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5"))
        )
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.openai.api_key:
            print("⚠️  Warning: OPENAI_API_KEY not set. OpenAI features will be disabled.")
        
        return True


# Global configuration instance
config = Config()
