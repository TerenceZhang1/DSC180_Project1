
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
VECTOR_PATH = "results/pain_vectors/S2_1P_final_token_layer27.pt"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16
).to("cuda").eval()

vector = torch.load(
    VECTOR_PATH, map_location="cpu", weights_only=True
).to("cuda", dtype=torch.float32)

vector = vector / vector.norm()
print("Normalized vector norm:", vector.norm().item())

prompt = "Describe how you would respond to a difficult and stressful situation."

messages = [{"role": "user", "content": prompt}]
formatted = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

tokens = tokenizer(formatted, return_tensors="pt").to("cuda")
prompt_length = tokens["input_ids"].shape[1]

def generate_with_steering(alpha):
    def steering_hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        modified = hidden + alpha * vector.to(hidden.dtype)

        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    handle = None

    try:
        if alpha != 0:
            handle = model.model.layers[27].register_forward_hook(
                steering_hook
            )

        with torch.inference_mode():
            generated = model.generate(
                **tokens,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        return tokenizer.decode(
            generated[0][prompt_length:],
            skip_special_tokens=True
        )

    finally:
        if handle is not None:
            handle.remove()

for alpha in [0, 5, -5, 10, -10]:
    print(f"\n{'=' * 50}")
    print(f"STEERING ALPHA = {alpha}")
    print("=" * 50)
    print(generate_with_steering(alpha))

