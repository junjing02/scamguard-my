"""Run ScamGuard MY on your own computer.

    python app.py                     # then open http://127.0.0.1:7860
    SCAMGUARD_MODEL=Qwen/Qwen2.5-3B-Instruct python app.py   # the first model: smaller and faster, but skips tools
"""
import os

import torch

from scamguard.app import build_ui
from scamguard.llm import HFChatModel

# Qwen3.5-4B needs ~10 GB on a GPU but ~18 GB on a CPU, so a CPU-only computer gets the 2B model (slow, but it fits).
GPU_MODEL = "Qwen/Qwen3.5-4B"
CPU_MODEL = "Qwen/Qwen3.5-2B"

if __name__ == "__main__":
    has_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    llm = HFChatModel(os.environ.get("SCAMGUARD_MODEL", GPU_MODEL if has_gpu else CPU_MODEL))
    print(f"Loaded {llm.model_id} on {llm.device}")
    build_ui(llm).launch()
