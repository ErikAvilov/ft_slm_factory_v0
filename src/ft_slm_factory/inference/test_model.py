import torch
import torch.backends.python_native as python_native

# Même contournement que train.py (GTX 1080 / pas de python3.11-dev pour Triton).
python_native.triton.enabled = False

from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "artifacts/models/qwen-banking"


def predict(request):
    prompt = (
        "Classify this banking request.\n"
        f"Request: {request}\n"
        "Category:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
        )

    generated_text = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return generated_text


def predicted_category(generated_text):
    after = generated_text.split("Category:", 1)[-1].strip()
    return after.split()[0] if after else ""


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)


print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

model.eval()

tests = [
    ("My card still hasn't arrived after two weeks", "card_arrival"),
    ("I don't recognize a payment made with my card", "card_payment_not_recognised"),
    ("Someone withdrew money from my account and it wasn't me", "cash_withdrawal_not_recognised"),
    ("Why did I get charged extra when withdrawing cash?", "cash_withdrawal_charge"),
]

expected = []
predicted = []

for request, label in tests:
    generated = predict(request)
    category = predicted_category(generated)
    expected.append(label)
    predicted.append(category)
    print(f"Request: {request}")
    print(f"Expected: {label} | Predicted: {category}")

print(f"accuracy: {accuracy_score(expected, predicted):.2f}")
print(f"f1: {f1_score(expected, predicted, average='macro', zero_division=0):.2f}")
