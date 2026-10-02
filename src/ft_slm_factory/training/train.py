from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

import torch
import torch.backends.python_native as python_native

# GTX 1080 (Pascal): Triton n'a pas de kernels utilisables ici et tente
# de recompiler un driver (besoin de python3.11-dev). On force ATen.
python_native.triton.enabled = False


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SELECTED_LABELS = [
    "card_arrival",
    "card_payment_not_recognised",
    "cash_withdrawal_not_recognised",
    "cash_withdrawal_charge",
]


def main():
    print("Loading Banking77...")

    dataset = load_dataset("PolyAI/banking77")

    label_names = dataset["train"].features["label"].names

    def keep_selected(row):
        return label_names[row["label"]] in SELECTED_LABELS

    train_dataset = dataset["train"].filter(keep_selected)

    def format_example(row):
        label_name = label_names[row["label"]]

        text = (
            "Classify this banking request.\n"
            f"Request: { row['text'] }\n"
            f"Category: { label_name }"
        )

        return {"text": text}

    train_dataset = train_dataset.map(format_example)

    print(train_dataset.select(range(3)))

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
    )

    torch_dtype=torch.float16

    lora_config = LoraConfig(
        r=8, # limité à 8 en raison de faible mémoire
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        task_type="CAUSAL_LM",
    )

    training_config = SFTConfig(
        output_dir="artifacts/models/qwen-banking",
        num_train_epochs=1,
        per_device_train_batch_size=1, # Ma carte graphique est trop faible donc je mets 1
        gradient_accumulation_steps=8,
        learning_rate=2e-4, # 0.0002 d"itération
        max_length=128, # taille maximale de texte attendue
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=lora_config,
        args=training_config,
        processing_class=tokenizer,
    )

    trainer.train()

    print(f"Training samples: {len(train_dataset)}")
    print(f"GPU available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    trainer.save_model("artifacts/models/qwen-banking")

    print("Training finished.")


if __name__ == "__main__":
    main()
