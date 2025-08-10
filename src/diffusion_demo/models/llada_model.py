"""
LLaDA model implementation for DiffuChatGPT
"""

import warnings
from typing import List, Dict, Any, Optional

from .base import BaseModel
from ..config import config
from ..utils.logger import app_logger

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class LLaDAModel(BaseModel):
    """LLaDA (LLM as Diffusion Autoencoders) model implementation"""
    
    def __init__(self, model_name: str = None):
        super().__init__(model_name or config.llada.default_model)
        self.model = None
        self.tokenizer = None
        
        self.load()
    
    def load(self) -> bool:
        """Load the LLaDA model"""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            app_logger.info(f"🚀 Loading LLaDA model: {self.model_name}")
            app_logger.info(f"🔧 Device: {self.device}")
            
            # Load model with appropriate dtype
            if self.device == "cuda":
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16
                ).to(self.device).eval()
            else:
                # For CPU, use float32
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                ).to(self.device).eval()
            
            # Load tokenizer
            app_logger.info("📥 Loading LLaDA tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            app_logger.info("✅ LLaDA loaded successfully!")
            
            self.is_loaded = True
            return True
            
        except ImportError:
            app_logger.error("❌ Transformers library not installed")
            return False
        except Exception as e:
            app_logger.error(f"❌ Failed to load LLaDA: {e}")
            # Fallback to a smaller model if the main one fails
            if "8B" in self.model_name:
                app_logger.info("🔄 Trying smaller LLaDA model...")
                try:
                    fallback_model = "GSAI-ML/LLaDA-1.3B-Instruct"  # Assuming there's a smaller version
                    self.model_name = fallback_model
                    return self.load()
                except:
                    app_logger.error(f"Failed to load both main and fallback LLaDA models: {e}")
                    return False
            else:
                return False
    
    def format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages for LLaDA using the chat template"""
        # Convert to the format expected by the chat template
        formatted_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                # Add system messages as user messages with a prefix
                formatted_messages.append({
                    "role": "user", 
                    "content": f"System: {content}"
                })
            elif role == "user":
                formatted_messages.append({
                    "role": "user",
                    "content": content
                })
            elif role == "assistant":
                formatted_messages.append({
                    "role": "assistant",
                    "content": content
                })
        
        # If the last message is from assistant, add a user prompt to continue
        if formatted_messages and formatted_messages[-1]["role"] == "assistant":
            # For remask scenarios, we want to continue the assistant's response
            last_content = formatted_messages[-1]["content"]
            if "<|mask|>" in last_content:
                # This is a remask request - format it properly
                formatted_messages[-1] = {
                    "role": "user",
                    "content": f"Please complete this response: {last_content}"
                }
        
        # Apply chat template
        try:
            prompt = self.tokenizer.apply_chat_template(
                formatted_messages, 
                add_generation_prompt=True, 
                tokenize=False
            )
        except Exception as e:
            app_logger.warning(f"Chat template failed, using simple format: {e}")
            # Fallback to simple formatting
            prompt = ""
            for msg in formatted_messages:
                if msg["role"] == "user":
                    prompt += f"Human: {msg['content']}\n"
                else:
                    prompt += f"Assistant: {msg['content']}\n"
            prompt += "Assistant: "
        
        return prompt
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = None,
        temperature: float = None,
        steps: int = None,
        block_length: int = None,
        cfg_scale: float = None,
        remasking: str = None,
        **kwargs
    ) -> str:
        """Generate response using LLaDA diffusion model"""
        if not self.is_loaded:
            return "Error: LLaDA model not loaded"
        
        try:
            import torch
            
            # Use config defaults if not provided
            max_tokens = max_tokens or config.llada.max_tokens
            temperature = temperature or config.llada.temperature
            steps = steps or config.llada.steps
            block_length = block_length or config.llada.block_length
            cfg_scale = cfg_scale or config.llada.cfg_scale
            remasking = remasking or config.llada.remasking
            
            # Log messages before generation
            self.log_messages_before_generation(messages, max_tokens, temperature, **kwargs)
            
            # Format the conversation
            prompt = self.format_chat_messages(messages)
            
            # Tokenize input
            input_ids = self.tokenizer(prompt)['input_ids']
            input_ids = torch.tensor(input_ids).to(self.device).unsqueeze(0)
            
            # Generate using the original generate function from llada.py
            with torch.no_grad():
                output = self._generate_diffusion(
                    model=self.model,
                    prompt=input_ids,
                    steps=steps,
                    gen_length=max_tokens,
                    block_length=min(block_length, max_tokens),
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    remasking=remasking,
                    mask_id=126336  # Default mask token ID
                )
            
            # Decode the generated part (everything after the input)
            generated_tokens = output[:, input_ids.shape[1]:]
            response = self.tokenizer.batch_decode(
                generated_tokens, 
                skip_special_tokens=True
            )[0].strip()
            
            app_logger.info(f"🎯 LLaDA Response: {response[:100]}{'...' if len(response) > 100 else ''}")
            return response
            
        except Exception as e:
            error_msg = f"LLaDA generation failed: {str(e)}"
            app_logger.error(f"❌ {error_msg}")
            return error_msg
    
    def _generate_diffusion(self, model, prompt, steps=128, gen_length=128, block_length=128, 
                           temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336):
        """Original LLaDA diffusion generation function"""
        import torch
        import numpy as np
        import torch.nn.functional as F
        
        def add_gumbel_noise(logits, temperature):
            if temperature == 0:
                return logits
            logits = logits.to(torch.float64)
            noise = torch.rand_like(logits, dtype=torch.float64)
            gumbel_noise = (- torch.log(noise)) ** temperature
            return logits.exp() / gumbel_noise
        
        def get_num_transfer_tokens(mask_index, steps):
            mask_num = mask_index.sum(dim=1, keepdim=True)
            base = mask_num // steps
            remainder = mask_num % steps
            num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
            for i in range(mask_num.size(0)):
                num_transfer_tokens[i, :remainder[i]] += 1
            return num_transfer_tokens
        
        x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
        x[:, :prompt.shape[1]] = prompt.clone()
        
        prompt_index = (x != mask_id)
        
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        
        assert steps % num_blocks == 0
        steps = steps // num_blocks
        
        for num_block in range(num_blocks):
            block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            for i in range(steps):
                mask_index = (x == mask_id)
                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits
                
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                
                if remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)
                
                x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
                
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)
                
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]
        
        return x
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.model:
            return {"status": "not_loaded"}
        
        import torch
        return {
            "status": "loaded",
            "model_name": self.model_name,
            "device": self.device,
            "model_type": "LLaDA (Diffusion-based)",
            "torch_dtype": str(self.model.dtype),
            "memory_usage": f"{torch.cuda.memory_allocated() / 1e9:.2f}GB" if self.device == "cuda" else "N/A"
        }
