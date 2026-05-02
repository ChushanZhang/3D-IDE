try:
    from .model import LlavaLlamaForCausalLM
except ImportError as e:
    print(f"Warning: Could not import LlavaLlamaForCausalLM: {e}")
    LlavaLlamaForCausalLM = None
