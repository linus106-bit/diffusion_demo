"""
Logging utilities for DiffuChatGPT
"""

import logging
import logging.handlers
import os
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

from ..config import config


def setup_logger(name: str = "diffusion_demo") -> logging.Logger:
    """Setup application logger"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.logging.log_level))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(config.logging.log_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    log_path = Path(config.logging.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=config.logging.max_log_size,
        backupCount=config.logging.backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def setup_message_logger() -> logging.Logger:
    """Setup specialized logger for chat messages"""
    logger = logging.getLogger('chat_messages')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create daily log file
    log_filename = f"logs/chat_messages_{datetime.now().strftime('%Y%m%d')}.log"
    log_path = Path(log_filename)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Simple formatter for message logging
    formatter = logging.Formatter('%(message)s')
    
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def log_chat_messages(messages: List[Dict[str, str]], model_type: str) -> None:
    """Log chat messages in a structured format"""
    logger = logging.getLogger('chat_messages')
    
    # Log in role,content format
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "").replace('\n', '\\n')
        logger.info(f"{role},{content}")
    
    # Log raw JSON for debugging
    import json
    logger.info(f"RAW_JSON: {json.dumps(messages, ensure_ascii=False)}")
    logger.info(f"MODEL_TYPE: {model_type}")
    logger.info("-" * 80)


# Global logger instances
app_logger = setup_logger()
message_logger = setup_message_logger()
