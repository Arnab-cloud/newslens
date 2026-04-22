from newslens.core.config import settings


class InferenceEngine:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "torch is not installed. Please install it with: pip install 'newslens[local]'"
        )

    def __init__(self, adapter_path: str | None = None):
        self.device = settings.device if torch.cuda.is_available() else "cpu"
        self.adapter_path = adapter_path or settings.adapter_path

        print(f"🔹 Loading model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(settings.base_model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            settings.base_model_name,
            torch_dtype=torch.float16
            if settings.load_in_half_precision
            else torch.float32,
            device_map=self.device,
        )

        if self.adapter_path:
            print(f"🔹 Attaching adapters from {self.adapter_path}...")
            self.model = PeftModel.from_pretrained(model, self.adapter_path)
        else:
            self.model = model

        self.model.eval()
        print("✅ Engine initialized.")

    def generate(self, prompt: str, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        gen_config = {
            "max_new_tokens": kwargs.get("max_tokens", settings.default_max_tokens),
            "temperature": kwargs.get("temperature", settings.default_temperature),
            "top_p": kwargs.get("top_p", settings.default_top_p),
            "do_sample": True if kwargs.get("temperature", 0.7) > 0 else False,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        with torch.no_grad():
            output_tokens = self.model.generate(**inputs, **gen_config)

        new_tokens = output_tokens[0][inputs.input_ids.shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
