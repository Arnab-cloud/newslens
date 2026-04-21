import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from newslens.core.config import settings


class NewsLensFinetuner:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _prepare_dataset(self, train_path: str, val_path: str):
        dataset = load_dataset(
            "csv", data_files={"train": train_path, "validation": val_path}
        )

        def preprocess(example):
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes news articles.",
                },
                {
                    "role": "user",
                    "content": f"Summarize the following news article:\n\n{example['article']}",
                },
                {"role": "assistant", "content": example["summary"]},
            ]
            full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            tokenized = self.tokenizer(full_text, max_length=2048, truncation=True)

            prompt_only = self.tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )
            prompt_len = len(self.tokenizer(prompt_only)["input_ids"])

            labels = list(tokenized["input_ids"])
            tokenized["labels"] = [-100] * prompt_len + labels[prompt_len:]
            return tokenized

        return dataset.map(preprocess, remove_columns=["article", "summary"])

    def train(self, train_csv: str, val_csv: str, output_dir: str = "./adapter_output"):
        # 1. Load Model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16
        )

        # 2. Configure LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # 3. Prepare Data
        tokenized_data = self._prepare_dataset(train_csv, val_csv)

        # 4. Training Args
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            save_strategy="epoch",
            fp16=True,
            report_to="none",
        )

        # 5. Trainer
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["validation"],
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

        trainer.train()
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"✅ Fine-tuning complete. Adapter saved to {output_dir}")
