#!/usr/bin/env python3
"""
Enhanced local model generator for DiffuChatGPT
Merged implementation from generate.py and smollm2.py with advanced features
"""

import torch
import warnings
from typing import List, Dict, Any, Optional

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class LocalModelGenerator:
    """Enhanced local model generator with multiple model support and chat templates"""
    
    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct", device: Optional[str] = None):
        """
        Initialize with SmolLM2 Instruct model (better than SmolLM)
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required. Install with: pip install transformers torch")
        
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self.tokenizer = None
        self.supports_chat_template = False
        
        print(f"🚀 Loading local model: {model_name}")
        print(f"🔧 Device: {self.device}")
        
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer with enhanced error handling"""
        try:
            print(f"📥 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Set pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Check if model supports chat template
            self.supports_chat_template = hasattr(self.tokenizer, 'apply_chat_template')
            
            print(f"📥 Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            
            self.model.eval()
            
            print(f"✅ Model loaded successfully!")
            print(f"📊 Device: {self.device}")
            print(f"🎯 Chat template support: {'✅' if self.supports_chat_template else '❌'}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            # Try fallback to smaller model
            if "SmolLM2" in self.model_name:
                print("🔄 Trying fallback to SmolLM (v1)...")
                self.model_name = "HuggingFaceTB/SmolLM-135M"
                self._load_model()
            else:
                raise e
    
    def _format_messages_with_template(self, messages: List[Dict[str, str]]) -> str:
        """Format messages using the model's chat template"""
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            print(f"⚠️ Chat template failed: {e}")
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
        max_new_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """
        Generate response from messages using advanced techniques
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature (0.0 = deterministic)
            
        Returns:
            Generated text string
        """
        try:
            # Format messages using chat template if available
            if self.supports_chat_template:
                text = self._format_messages_with_template(messages)
            else:
                text = self._format_messages_simple(messages)
            
            # Tokenize input
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            # Generate response with advanced parameters
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
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
                return "I'm not sure how to respond to that. Could you please rephrase your question?"
            
            return response
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.model:
            return {"status": "not_loaded"}
        
        try:
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

# Global model instance (lazy loading)
_model_generator = None

def get_model_generator(model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct") -> LocalModelGenerator:
    """
    Get or create the global model generator
    
    Args:
        model_name: Model to load if creating new instance
        
    Returns:
        LocalModelGenerator instance
    """
    global _model_generator
    if _model_generator is None:
        _model_generator = LocalModelGenerator(model_name)
    return _model_generator

def generate_local_response(
    messages: List[Dict[str, str]],
    max_new_tokens: int = 1000,
    temperature: float = 0.7
) -> str:
    """
    Convenience function to generate response using the global model
    
    Args:
        messages: Conversation history
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated response text
    """
    try:
        generator = get_model_generator()
        response = generator.generate_response(messages, max_new_tokens, temperature)
        
        print(f"🎯 Local model response: {response[:100]}{'...' if len(response) > 100 else ''}")
        return response
        
    except Exception as e:
        error_msg = f"Local model error: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg

# Test function for standalone usage
if __name__ == "__main__":
    print("🧪 Testing Enhanced Local Model Generator...")
    
    try:
        # Test with a conversation
        test_messages = [
            {"role": "user", "content": "Give me a brief explanation of gravity in simple terms."}
        ]
        
        print(f"\n💬 Test Input: {test_messages[0]['content']}")
        
        # Generate response
        response = generate_local_response(test_messages, max_new_tokens=200, temperature=0.7)
        
        print(f"🤖 Generated response:")
        print(f"   {response}")
        
        # Get model info
        generator = get_model_generator()
        info = generator.get_model_info()
        
        print(f"\n📊 Model Information:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        print(f"\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
