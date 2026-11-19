"""LLM wrapper: a free instruction-tuned Hugging Face model run locally (NVIDIA GPU, Apple Silicon Mac or CPU)
with native tool calling.

The model's chat template accepts `tools=[...]` and makes the model answer with a <tool_call> block:
JSON inside the tag for Qwen2.5, XML (<function=...><parameter=...>) for Qwen3.5.
agent.parse_tool_calls() parses both.
"""
import re

_THINK = re.compile(r"<think>.*?(</think>|$)", re.S)   # a reasoning part, if the model writes one


class HFChatModel:
    """Loads the model once, then generate() turns a chat (messages + tool schemas) into the model's raw text."""

    def __init__(self, model_id="Qwen/Qwen3.5-4B", load_in_4bit=False, max_new_tokens=400, device=None):
        import torch                                       # imported here: the offline tests do not need torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if device is None:                                 # pick the best device available
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()   # Apple Silicon (M1-M4) GPU
                      else "cpu")
        self.device = device
        dtype = torch.float32 if device == "cpu" else torch.float16
        kwargs = {}
        if load_in_4bit:                                   # bitsandbytes: NVIDIA GPU only
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
            kwargs["device_map"] = "auto"                  # a 4-bit model is placed while it loads
        else:
            kwargs["dtype"] = dtype                        # fp16 on a GPU, fp32 on a CPU
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        except ValueError:                                 # multimodal models such as Qwen3.5 (text use only)
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        if not load_in_4bit:                               # also works on a ZeroGPU Space (emulated CUDA)
            self.model.to(self.device)
        self.model.eval()
        self.model_id = model_id

    def generate(self, messages, tools):
        """Greedy (repeatable) generation. Returns only the new text, without special tokens or <think>."""
        # enable_thinking=False: Qwen3/3.5 answer directly (no long <think> part); older templates ignore it
        prompt = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        return _THINK.sub("", text.replace("<|im_end|>", "").replace("<|endoftext|>", "")).strip()
