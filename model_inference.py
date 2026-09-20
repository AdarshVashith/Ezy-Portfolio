"""
Step 3: Fine-Tuned Model Inference & Concept Engine -- Hybrid Architecture
--------------------------------------------------------------------------
Design:
    - CONCEPT / EXPLANATORY questions -> Fine-tuned model / Concept Engine
    - LIVE MARKET DATA (price, news, volatility regime, buy/sell statistical snapshot)
      -> Deterministic SQLite RAG system (hallucination-proof)
"""

import os
import re
import json

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "finance_model_lora"
DATASET_FILE = "finance_training_data.jsonl"

_model = None
_tokenizer = None
_knowledge_cache = None


def load_knowledge_cache():
    global _knowledge_cache
    if _knowledge_cache is not None:
        return _knowledge_cache

    _knowledge_cache = []
    if os.path.exists(DATASET_FILE):
        try:
            with open(DATASET_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        _knowledge_cache.append(json.loads(line))
        except Exception:
            pass
    return _knowledge_cache


def is_concept_question(message: str) -> bool:
    """
    Router logic: Checks if user query is seeking concept explanations,
    methodology definitions, or statistical theory.
    """
    concept_keywords = [
        "what is", "explain", "how does", "why does", "kya hota hai",
        "kya hai", "samjhao", "matlab kya", "define", "definition",
        "p-value", "garch", "kelly", "hmm", "alpha decay", "sharpe",
        "drawdown", "monte carlo", "geometric brownian", "martingale",
        "survivorship bias", "backtest artifact", "methodology"
    ]
    message_lower = message.lower()
    return any(re.search(r'\b' + re.escape(kw) + r'\b', message_lower) if ' ' in kw else kw in message_lower for kw in concept_keywords)


def load_finetuned_model():
    """Attempts to load fine-tuned model and adapter if weights exist."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    if not os.path.exists(ADAPTER_PATH):
        return None, None

    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        from peft import PeftModel  # type: ignore

        _tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        _model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        _model.eval()
        return _model, _tokenizer
    except Exception as e:
        print(f"Fine-tuned model loading deferred: {e}")
        return None, None


def find_best_knowledge_match(message: str):
    """Fallback semantic keyword matcher over curated quant dataset."""
    cache = load_knowledge_cache()
    if not cache:
        return None

    msg_lower = message.lower()
    best_match = None
    max_matches = 0

    for item in cache:
        instr = item.get("instruction", "").lower()
        words = set(re.findall(r'\w+', instr))
        matches = sum(1 for w in words if len(w) > 3 and w in msg_lower)
        if matches > max_matches:
            max_matches = matches
            best_match = item

    if best_match and max_matches >= 1:
        return best_match.get("output")

    return None


def generate_concept_answer(message: str, max_new_tokens: int = 250) -> str:
    """
    Generates fluent, finance-specialized explanation using fine-tuned weights
    or curated domain dataset engine.
    """
    model, tokenizer = load_finetuned_model()

    if model is not None and tokenizer is not None:
        try:
            import torch  # type: ignore
            messages = [
                {"role": "system", "content": "You are a specialized quant finance and trading systems assistant, trained on financial concepts and this project's empirical research findings."},
                {"role": "user", "content": message}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return response.strip()
        except Exception as e:
            print(f"Model generation fallback: {e}")

    # Fallback to curated quant knowledge base
    matched_answer = find_best_knowledge_match(message)
    if matched_answer:
        return matched_answer

    return (
        "I am trained on this project's quantitative finance curriculum. "
        "You can ask me to explain concepts such as GARCH volatility modeling, "
        "the Kelly Criterion, Hidden Markov Models (HMM), Alpha Decay, RSI, "
        "Monte Carlo simulations, p-value validation, or our empirical findings."
    )
