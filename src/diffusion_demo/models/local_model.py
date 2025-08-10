"""
Autoregressive model implementation for DiffuChatGPT
"""

import warnings
from typing import List, Dict, Any, Optional

from .base import BaseModel
from ..config import config
from ..utils.logger import app_logger

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class AutoregressiveModel(BaseModel):
    """Autoregressive Transformers model implementation"""
    
    def __init__(self, model_name: str = None, device: str = None):
        super().__init__(model_name or config.autoregressive_model.default_model)
        self.device = device or config.autoregressive_model.device
        self.model = None
        self.tokenizer = None
        self.supports_chat_template = False
        
        self.load()
    
    def load(self) -> bool:
        """Load the autoregressive model"""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Auto-detect device if not specified
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            app_logger.info(f"🚀 Loading autoregressive model: {self.model_name}")
            app_logger.info(f"🔧 Device: {self.device}")
            
            # Load tokenizer
            app_logger.info("📥 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Set pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Check if model supports chat template
            self.supports_chat_template = hasattr(self.tokenizer, 'apply_chat_template')
            
            # Load model
            app_logger.info("📥 Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            
            self.model.eval()
            
            app_logger.info("✅ Autoregressive model loaded successfully!")
            app_logger.info(f"📊 Device: {self.device}")
            app_logger.info(f"🎯 Chat template support: {'✅' if self.supports_chat_template else '❌'}")
            
            self.is_loaded = True
            return True
            
        except ImportError:
            app_logger.error("❌ Transformers library not installed")
            return False
        except Exception as e:
            app_logger.error(f"❌ Failed to load autoregressive model: {e}")
            # Try fallback to smaller model
            if "SmolLM2" in self.model_name:
                app_logger.info("🔄 Trying fallback to SmolLM (v1)...")
                self.model_name = "HuggingFaceTB/SmolLM-135M"
                return self.load()
            else:
                return False
    
    def _format_messages_with_template(self, messages: List[Dict[str, str]]) -> str:
        """Format messages using the model's chat template"""
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            app_logger.warning(f"Chat template failed: {e}")
            return self._format_messages_simple(messages)
    
    def _format_messages_simple(self, messages: List[Dict[str, str]]) -> str:
        """Simple fallback message formatting"""
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"Human: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        
        prompt += "Assistant:"
        return prompt
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """Generate response using autoregressive model"""
        if not self.is_loaded:
            return "Error: Autoregressive model not loaded"
        
        try:
            import torch
            
            max_tokens = max_tokens or config.autoregressive_model.max_tokens
            temperature = temperature or config.autoregressive_model.temperature
            
            # Format messages using chat template if available
            if self.supports_chat_template:
                text = self._format_messages_with_template(messages)
            else:
                text = self._format_messages_simple(messages)
            
            # Tokenize input
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            app_logger.info(f"🎯 Autoregressive model generating with {max_tokens} tokens, temp={temperature}")
            
            # Generate response with advanced parameters
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    top_p=0.9,  # Nucleus sampling
                    top_k=50,   # Top-k sampling
                    repetition_penalty=1.1,  # Reduce repetition
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            # Decode only the generated part (excluding input)
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            
            # Clean up the response
            response = response.strip()
            
            # Handle empty responses
            if not response:
                response = "I'm not sure how to respond to that. Could you please rephrase your question?"
            
            app_logger.info(f"🎯 Autoregressive model response: {response[:100]}{'...' if len(response) > 100 else ''}")
            return response
            
        except Exception as e:
            error_msg = f"Autoregressive model generation failed: {str(e)}"
            app_logger.error(f"❌ {error_msg}")
            return error_msg
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.model:
            return {"status": "not_loaded"}
        
        try:
            import torch
            num_params = sum(p.numel() for p in self.model.parameters())
            return {
                "status": "loaded",
                "model_name": self.model_name,
                "device": self.device,
                "parameters": f"~{num_params / 1e6:.1f}M",
                "chat_template": self.supports_chat_template,
                "torch_dtype": str(self.model.dtype),
                "memory_usage": f"{torch.cuda.memory_allocated() / 1e9:.2f}GB" if self.device == "cuda" else "N/A"
            }
        except Exception as e:
            return {"status": "loaded", "error": str(e)}
