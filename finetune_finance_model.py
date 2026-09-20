"""
Step 2: LoRA Fine-Tuning -- Finance-Specialized Small Language Model
--------------------------------------------------------------------
LoRA (Low-Rank Adaptation): poore model (jisme lakhon/crores parameters
hain) ko retrain nahi karte -- sirf ek CHHOTA "adapter" (kuch lakh
parameters) train karte hain jo base model ke upar "sit" karta hai.
Isse training FAST hoti hai aur consumer hardware (ya free Google Colab
GPU) pe bhi chal jaati hai.

HARDWARE REQUIREMENT: Ek GPU chahiye hoga (free Google Colab T4 GPU
kaafi hai). Agar local GPU nahi hai, ye script Google Colab mein chalao
(colab.research.google.com, Runtime > Change runtime type > GPU).

Chalane ka tarika:
    python3 finetune_finance_model.py
"""

import os
import json

try:
    import torch  # type: ignore
    from datasets import Dataset  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, TaskType  # type: ignore
    HAS_DEEP_LEARNING_LIBS = True
except ImportError:
    HAS_DEEP_LEARNING_LIBS = False

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_FILE = "finance_training_data.jsonl"
OUTPUT_DIR = "finance_model_lora"


def load_training_data(filename):
    examples = []
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Training data file {filename} not found. Run build_finetune_dataset.py first.")
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def format_as_chat(example, tokenizer):
    """Instruction/output ko model ke chat-format mein convert karta hai"""
    messages = [
        {"role": "system", "content": "You are a specialized quant finance and trading systems assistant, trained on financial concepts and this project's empirical research findings."},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}


def main():
    if not HAS_DEEP_LEARNING_LIBS:
        print("[Notice] Deep learning packages (torch, transformers, peft) are not installed in the local environment.")
        print("Option 2 (Google Colab): Open and run 'Colab_Finance_FineTuning.ipynb' in Google Colab on a free T4 GPU.")
        print("To install locally: pip install torch transformers datasets peft accelerate")
        return

    print(f"Loading base model: {BASE_MODEL}")
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()
    print(f"Device support: CUDA={has_cuda}, MPS (Apple Silicon)={has_mps}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if has_cuda else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto" if has_cuda else None
    )

    # LoRA Configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading and formatting training data...")
    raw_examples = load_training_data(DATASET_FILE)
    formatted = [format_as_chat(ex, tokenizer) for ex in raw_examples]
    dataset = Dataset.from_list(formatted)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="epoch",
        fp16=has_cuda,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    print("Starting fine-tuning...")
    trainer.train()

    print(f"Saving fine-tuned adapter to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("\nFine-tuning complete. Adapter weights saved successfully.")


if __name__ == "__main__":
    main()
