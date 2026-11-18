# ScamGuard MY

An LLM tool-calling agent that checks suspicious Malaysian scam messages (SMS, WhatsApp, email, links) and knows its limits.

**Model:** `Qwen/Qwen3.5-4B` on a GPU or Apple Silicon (free, Apache 2.0, native tool calling, thinking mode off).
On a CPU-only computer, `app.py` uses `Qwen/Qwen3.5-2B`, because the 4B model needs about 18 GB of memory on a CPU.
This works but is very slow (several minutes per message). Set the `SCAMGUARD_MODEL` variable to choose another model.
The first model, `Qwen/Qwen2.5-3B-Instruct`, often skipped tools, so the default changed. No training or fine-tuning.

## Run on your own computer
Python 3.10 or newer. An Apple Silicon Mac or an NVIDIA GPU is recommended.
```
pip install -r requirements.txt
python tests_offline.py      # code checks with a mock LLM, no download
python app.py                # downloads the model once (~9 GB), then open http://127.0.0.1:7860
```

## Files
| File | Purpose |
|---|---|
| `app.py` | Start file: loads the model and launches the interface |
| `scamguard/tools.py` | 6 tool functions over a small JSON dataset (URLs analysed as text, never visited) |
| `scamguard/validation.py` | Tool schemas shown to the LLM + allow list and argument validation |
| `scamguard/guardrails.py` | Python guardrails: blocked requests, injection flag, PII redaction, output check, procedure check |
| `scamguard/agent.py` | Agent loop: the LLM proposes calls, Python validates and runs them |
| `scamguard/llm.py` | Hugging Face `transformers` wrapper |
| `scamguard/app.py` | Gradio interface that shows every function call |
| `scamguard/session.py` | Session memory and the human Confirm / Cancel step for reports |
| `scamguard/testing.py` | `run_case()`: runs one message and prints the input, every call and the response |
| `scamguard/data/` | Hand-written dataset: scam patterns, scam types, official organisations, URL flags |
| `tests_offline.py` | Offline code checks with a scripted mock LLM |
