#!/usr/bin/env python3
"""
LLaDA (LLM as Diffusion Autoencoders) - Enhanced Implementation
Original diffusion generation functions with integrated chat interface support
"""

import torch
import numpy as np
import torch.nn.functional as F
import warnings
from typing import List, Dict, Any

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from transformers import AutoTokenizer, AutoModel
    LLADA_AVAILABLE = True
except ImportError as e:
    LLADA_AVAILABLE = False
    print(f"❌ LLaDA dependencies not available: {e}")


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
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
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
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


class LLaDAWrapper:
    """
    Chat interface wrapper for LLaDA (LLM as Diffusion Autoencoders)
    Integrates the original LLaDA diffusion functions with chat functionality
    """
    
    def __init__(self, model_name: str = "GSAI-ML/LLaDA-8B-Instruct"):
        """
        Initialize LLaDA wrapper
        
        Args:
            model_name: HuggingFace model name for LLaDA
        """
        if not LLADA_AVAILABLE:
            raise ImportError("LLaDA dependencies not available. Install transformers and torch.")
        
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
        print(f"🚀 Initializing LLaDA...")
        print(f"📍 Model: {model_name}")
        print(f"🔧 Device: {self.device}")
        
        self._load_model()
    
    def _load_model(self):
        """Load the LLaDA model and tokenizer"""
        try:
            print(f"📥 Loading LLaDA model...")
            
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
            
            print(f"📥 Loading LLaDA tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            print(f"✅ LLaDA loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load LLaDA: {e}")
            # Fallback to a smaller model if the main one fails
            if "8B" in self.model_name:
                print("🔄 Trying smaller LLaDA model...")
                try:
                    fallback_model = "GSAI-ML/LLaDA-1.3B-Instruct"  # Assuming there's a smaller version
                    self.model_name = fallback_model
                    self._load_model()
                except:
                    raise Exception(f"Failed to load both main and fallback LLaDA models: {e}")
            else:
                raise e
    
    def format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format chat messages for LLaDA using the chat template
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Formatted prompt string
        """
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
            print(f"⚠️ Chat template failed, using simple format: {e}")
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
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        steps: int = 128,
        block_length: int = 32,
        cfg_scale: float = 0.0,
        remasking: str = 'low_confidence'
    ) -> str:
        """
        Generate response using LLaDA diffusion model
        
        Args:
            messages: Conversation history
            max_new_tokens: Maximum tokens to generate (gen_length)
            temperature: Sampling temperature for Gumbel noise
            steps: Diffusion sampling steps
            block_length: Block length for generation
            cfg_scale: Classifier-free guidance scale
            remasking: Remasking strategy ('low_confidence' or 'random')
            
        Returns:
            Generated response text
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("LLaDA model not loaded")
        
        try:
            # Format the conversation
            prompt = self.format_chat_messages(messages)
            
            # Tokenize input
            input_ids = self.tokenizer(prompt)['input_ids']
            input_ids = torch.tensor(input_ids).to(self.device).unsqueeze(0)
            
            print(f"🎯 LLaDA generating with {steps} steps, {max_new_tokens} tokens...")
            
            # Generate using the original generate function
            with torch.no_grad():
                output = generate(
                    model=self.model,
                    prompt=input_ids,
                    steps=steps,
                    gen_length=max_new_tokens,
                    block_length=min(block_length, max_new_tokens),
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
            
            print(f"🎯 LLaDA Response: {response[:100]}{'...' if len(response) > 100 else ''}")
            return response
            
        except Exception as e:
            error_msg = f"LLaDA generation failed: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded LLaDA model"""
        if not self.model:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_name": self.model_name,
            "device": self.device,
            "model_type": "LLaDA (Diffusion-based)",
            "torch_dtype": str(self.model.dtype),
            "memory_usage": f"{torch.cuda.memory_allocated() / 1e9:.2f}GB" if self.device == "cuda" else "N/A"
        }

# Global model instance for efficiency
_llada_wrapper = None

def get_llada_wrapper(model_name: str = "GSAI-ML/LLaDA-8B-Instruct") -> LLaDAWrapper:
    """
    Get or create the global LLaDA wrapper instance
    
    Args:
        model_name: Model to load
        
    Returns:
        LLaDAWrapper instance
    """
    global _llada_wrapper
    
    if _llada_wrapper is None:
        _llada_wrapper = LLaDAWrapper(model_name)
    
    return _llada_wrapper

def generate_llada_response(
    messages: List[Dict[str, str]],
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    model_name: str = "GSAI-ML/LLaDA-8B-Instruct"
) -> str:
    """
    Convenience function for generating responses with LLaDA
    
    Args:
        messages: Conversation history
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        model_name: Model to use
        
    Returns:
        Generated response
    """
    try:
        wrapper = get_llada_wrapper(model_name)
        response = wrapper.generate_response(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        return response
        
    except Exception as e:
        error_msg = f"LLaDA generation failed: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg

def main():
    """Original standalone test function"""
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])

def test_chat_interface():
    """Test the new chat interface"""
    print("🧪 Testing LLaDA Chat Interface...")
    
    try:
        # Test conversation
        test_messages = [
            {"role": "user", "content": "Hello! Can you help me with a math problem?"},
        ]
        
        # Generate response
        response = generate_llada_response(
            messages=test_messages,
            max_new_tokens=64,
            temperature=0.0
        )
        
        print(f"\n💬 Test Conversation:")
        print(f"👤 User: {test_messages[0]['content']}")
        print(f"🧠 LLaDA: {response}")
        
        # Get model info
        wrapper = get_llada_wrapper()
        info = wrapper.get_model_info()
        
        print(f"\n📊 Model Info:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        print(f"\n✅ LLaDA chat interface test completed successfully!")
        
    except Exception as e:
        print(f"❌ LLaDA chat interface test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        test_chat_interface()
    else:
        main()