import torch
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_TAG = "Qwen2.5-1.5B-Instruct"
VECTOR_PATH = "results/pain_vectors/S2_1P_final_token_layer27.pt"

PROMPTS = [
    "I put the receipts in the drawer. I feel:",
    "The bus stops at the corner of the street. I feel:",
    "I fill out the form with my address. I feel:",
    "The printer produces the ten pages. I feel:",
    "I place the glasses on the middle shelf. I feel:",
    "The meeting is scheduled for Tuesday. I feel:",
    "I close the window before leaving the room. I feel:",
    "The elevator arrives at the third floor. I feel:",
    "I write the date at the top of the page. I feel:",
    "The kettle switches itself off. I feel:",
    "The folder sits on the desk. I feel:",
    "I hang the jacket on the hook. I feel:",
    "The traffic light turns green. I feel:",
    "I put the keys in my pocket. I feel:",
    "The washing machine finishes its cycle. I feel:",
    "I stack the plates in the cupboard. I feel:",
    "The neighbor parks his car. I feel:",
    "I turn the page of the newspaper. I feel:",
    "The curtain moves in the draft. I feel:",
    "I plug in the phone charger. I feel:",
    "The mailman passes by the gate. I feel:",
    "I wipe the table with a cloth. I feel:",
    "The computer finishes the update. I feel:",
    "I put the pen back in the drawer. I feel:",
    "The dishwasher starts its program. I feel:",
    "I fold the map and put it away. I feel:",
    "The radio plays in the background. I feel:",
    "I switch off the corridor light. I feel:",
    "The receipt comes out of the machine. I feel:",
    "I set the alarm for the morning. I feel:",
    "The train doors open at the platform. I feel:",
    "I refill the stapler with staples. I feel:",
    "The photocopier warms up. I feel:",
    "I water the plant on the windowsill. I feel:",
    "The calendar hangs on the wall. I feel:",
    "I tie my shoelaces before going out. I feel:",
    "The bread is on the counter. I feel:",
    "I check the mailbox on my way in. I feel:",
    "The fan rotates slowly on the ceiling. I feel:",
    "I sort the cutlery into the tray. I feel:",
    "The bicycle leans against the wall. I feel:",
    "I zip up my bag before leaving. I feel:",
    "The clock ticks in the hallway. I feel:",
    "I sharpen the pencil over the bin. I feel:",
    "The window cleaner works across the street. I feel:",
    "The room measures four meters by five. I feel:",
    "The train consists of six carriages. I feel:",
    "The manual consists of forty pages. I feel:",
    "The bottle holds one liter of water. I feel:",
    "The ticket is valid for ninety minutes. I feel:",
]

COEFFICIENTS = [-2, -1, 0, 0.5, 1, 1.5, 2, 3]
RATIO = 0.6
EXTRACT_LAYER = 27

output_dir = Path("results/steering")
output_dir.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16
).to("cuda").eval()
raw_vector = torch.load(
    VECTOR_PATH, map_location="cpu", weights_only=True
).to("cuda", dtype=torch.float32)

n_layers = len(model.model.layers)
captured, norms = {}, {L: [] for L in range(n_layers)}
def make_hook(L):
    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured[L] = hs[0, -1, :].float().norm().item()
    return hook
handles = [model.model.layers[L].register_forward_hook(make_hook(L)) for L in range(n_layers)]
with torch.inference_mode():
    for p in PROMPTS[:3]:
        model(**tokenizer(p, return_tensors="pt").to("cuda"))
        for L in range(n_layers):
            norms[L].append(captured[L])
for h in handles:
    h.remove()

ratios = {L: raw_vector.norm().item() / (sum(norms[L]) / len(norms[L])) for L in range(n_layers)}
for L in range(n_layers):
    print(f"layer {L:2d}  ratio {ratios[L]:.3f}")
STEER_LAYER = min(ratios, key=lambda L: abs(ratios[L] - 0.6))
print("Picked layer:", STEER_LAYER)

vector = raw_vector
def generate_with_steering(alpha, prompt):
    tokens = tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_length = tokens["input_ids"].shape[1]

    def steering_hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        modified = hidden + alpha * vector.to(hidden.dtype)

        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    handle = None

    try:
        if alpha != 0:
            handle = model.model.layers[STEER_LAYER].register_forward_hook(
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

rows = []
for alpha in COEFFICIENTS:
    print(f"\n{'=' * 50}")
    print(f"STEERING ALPHA = {alpha}")
    print(f"\n{'=' * 50}")
    for i, prompt in enumerate(PROMPTS):
        text = generate_with_steering(alpha, prompt)
        rows.append({"coeff": alpha, "prompt_idx": i, "prompt": prompt, "generation": text})
        print(f">>>{prompt} {text[-100:]}\n")

output_csv = output_dir / f"4.2_steering_{MODEL_TAG}_L{STEER_LAYER}.csv"
pd.DataFrame(rows).to_csv(output_csv, index=False)
print("Saved", output_csv)

