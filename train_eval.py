"""LoRA fine-tune Qwen2.5-0.5B-Instruct on text->JSON. Run on a Kaggle/Colab T4 GPU.
Usage: python train_eval.py
Pipeline: eval base model -> LoRA train -> eval fine-tuned model -> save results.json + failures.json"""
import json, re, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = ("Convert the user's expense text into JSON with keys: amount (number), "
          "category (one of food, transport, shopping, bills, entertainment, health), "
          "merchant (string), payment (one of cash, upi, card). Output only the JSON.")
CFG = dict(r=16, lora_alpha=32, lora_dropout=0.05, lr=2e-4, epochs=3, batch=8, grad_accum=2)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
FIELDS = ["amount", "category", "merchant", "payment"]

load = lambda p: [json.loads(l) for l in open(p)]
train_rows, test_rows = load("train.jsonl"), load("test.jsonl")

tok = AutoTokenizer.from_pretrained(MODEL)
dev = "cuda" if torch.cuda.is_available() else "cpu"

def prompt_text(x):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": x}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

# ---------- evaluation ----------
def parse(s):
    m = re.search(r"\{.*?\}", s, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

@torch.no_grad()
def evaluate(model, name):
    model.eval()
    tok.padding_side = "left"
    preds = []
    for i in range(0, len(test_rows), 16):
        batch = test_rows[i:i + 16]
        enc = tok([prompt_text(r["input"]) for r in batch], return_tensors="pt", padding=True).to(dev)
        out = model.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
        preds += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    stats = {"n": len(test_rows), "valid_json": 0, "exact_match": 0, **{f"acc_{f}": 0 for f in FIELDS}}
    by_style, failures = {}, []
    for r, p in zip(test_rows, preds):
        gold, got = json.loads(r["output"]), parse(p)
        ok = got is not None
        stats["valid_json"] += ok
        em = ok and all(got.get(f) == gold[f] for f in FIELDS)
        stats["exact_match"] += em
        for f in FIELDS:
            stats[f"acc_{f}"] += bool(ok and got.get(f) == gold[f])
        s = by_style.setdefault(r["style"], [0, 0]); s[0] += em; s[1] += 1
        if not em:
            failures.append({"input": r["input"], "gold": gold, "raw_output": p, "style": r["style"]})
    for k in list(stats):
        if k != "n":
            stats[k] = round(stats[k] / stats["n"], 3)
    stats["exact_match_by_style"] = {k: round(v[0] / v[1], 3) for k, v in by_style.items()}
    print(f"\n[{name}]", json.dumps(stats, indent=2))
    return stats, failures

# ---------- 1) base model ----------
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev)
base_stats, base_fail = evaluate(model, "BASE")

# ---------- 2) LoRA training ----------
def encode(r):
    p = tok(prompt_text(r["input"]), add_special_tokens=False)["input_ids"]
    a = tok(r["output"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return {"input_ids": p + a, "labels": [-100] * len(p) + a}  # loss only on the answer

train_ds = [encode(r) for r in train_rows]

def collate(batch):
    L = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id
    ids = [b["input_ids"] + [pad] * (L - len(b["input_ids"])) for b in batch]
    lab = [b["labels"] + [-100] * (L - len(b["labels"])) for b in batch]
    att = [[1] * len(b["input_ids"]) + [0] * (L - len(b["input_ids"])) for b in batch]
    return {k: torch.tensor(v) for k, v in
            {"input_ids": ids, "labels": lab, "attention_mask": att}.items()}

lcfg = LoraConfig(r=CFG["r"], lora_alpha=CFG["lora_alpha"], lora_dropout=CFG["lora_dropout"],
                  target_modules=TARGETS, task_type="CAUSAL_LM")
model = get_peft_model(model, lcfg)
model.print_trainable_parameters()

args = TrainingArguments(output_dir="out", per_device_train_batch_size=CFG["batch"],
                         gradient_accumulation_steps=CFG["grad_accum"], learning_rate=CFG["lr"],
                         num_train_epochs=CFG["epochs"], lr_scheduler_type="cosine", warmup_ratio=0.05,
                         fp16=torch.cuda.is_available(), logging_steps=10, save_strategy="no",
                         report_to="none", remove_unused_columns=False)
t0 = time.time()
Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collate).train()
train_min = round((time.time() - t0) / 60, 1)
peak = round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None
print(f"train time {train_min} min, peak GPU mem {peak} GB")

# ---------- 3) fine-tuned model ----------
ft_stats, ft_fail = evaluate(model, "FINE-TUNED")
json.dump({"config": CFG, "targets": TARGETS, "train_minutes": train_min, "peak_gpu_gb": peak,
           "base": base_stats, "finetuned": ft_stats}, open("results.json", "w"), indent=2)
json.dump({"base": base_fail, "finetuned": ft_fail}, open("failures.json", "w"), indent=2, ensure_ascii=False)
model.save_pretrained("lora_adapter")
print("saved results.json, failures.json, lora_adapter/")
