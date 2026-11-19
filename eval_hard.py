"""Stress test: 25 hand-written hard cases. Run after train_eval.py (needs ./lora_adapter).
Reports exact match for base and fine-tuned, and saves every failure to hard_results.json."""
import json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = ("Convert the user's expense text into JSON with keys: amount (number), "
          "category (one of food, transport, shopping, bills, entertainment, health), "
          "merchant (string), payment (one of cash, upi, card). Output only the JSON.")
F = ["amount", "category", "merchant", "payment"]
def g(a, c, m, p): return dict(amount=a, category=c, merchant=m, payment=p)

CASES = [
 ("Paid KFC 450 by GPay", g(450, "food", "KFC", "upi")),
 ("Spent Rs 1,200 at Ajio using my card", g(1200, "shopping", "Ajio", "card")),
 ("Uber ride cost 2.5k, paid in cash", g(2500, "transport", "Uber", "cash")),
 ("Paid two thousand rupees to Airtel through UPI", g(2000, "bills", "Airtel", "upi")),
 ("Swiggy order ₹349.50 on credit card", g(349.5, "food", "Swiggy", "card")),
 ("IndiGo ticket 5800 rupees paid via debit card", g(5800, "transport", "IndiGo", "card")),
 ("Zee5 subscription 299 paid with PhonePe", g(299, "entertainment", "Zee5", "upi")),
 ("Medplus 180 cash", g(180, "health", "Medplus", "cash")),
 ("Bought shoes on Myntra for 1k via UPI", g(1000, "shopping", "Myntra", "upi")),
 ("Rs 75.5 to Metro, cash", g(75.5, "transport", "Metro", "cash")),
 ("upi payment of 3,499 to Croma", g(3499, "shopping", "Croma", "upi")),
 ("Apollo Pharmacy bill 1250 paid by card, 20% discount applied", g(1250, "health", "Apollo Pharmacy", "card")),
 ("Spent ₹5000 on Decathlon (credit card) last Tuesday at 6pm", g(5000, "shopping", "Decathlon", "card")),
 ("bhai Domino's ko 399 diye GPay se", g(399, "food", "Domino's", "upi")),
 ("Vi recharge 239 through PhonePe", g(239, "bills", "Vi", "upi")),
 ("Paid 12.75 to Barista in cash", g(12.75, "food", "Barista", "cash")),
 ("Tata Power electricity bill Rs 1840 debit card", g(1840, "bills", "Tata Power", "card")),
 ("PVR tickets cost ₹760 paid via UPI", g(760, "entertainment", "PVR", "upi")),
 ("Practo consultation 500 rupees cash", g(500, "health", "Practo", "cash")),
 ("Amazon: 99999 (card)", g(99999, "shopping", "Amazon", "card")),
 ("Steam wallet top-up 1.5k with GPay", g(1500, "entertainment", "Steam", "upi")),
 ("Spent 0.5k on Haldiram's in cash", g(500, "food", "Haldiram's", "cash")),
 ("Paid ola 210 via UPI", g(210, "transport", "Ola", "upi")),
 ("Cult.fit membership 3 thousand 500, debit card", g(3500, "health", "Cult.fit", "card")),
 ("Jio 999 phonepe", g(999, "bills", "Jio", "upi")),
]

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"

def prompt(x):
    return tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": x}],
                                   tokenize=False, add_generation_prompt=True)
def parse(s):
    m = re.search(r"\{.*?\}", s, re.S)
    try: return json.loads(m.group(0)) if m else None
    except Exception: return None

@torch.no_grad()
def run(model, name):
    model.eval(); preds = []
    for i in range(0, len(CASES), 8):
        b = CASES[i:i + 8]
        enc = tok([prompt(x) for x, _ in b], return_tensors="pt", padding=True).to(dev)
        out = model.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
        preds += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    em, fails = 0, []
    for (x, gold), p in zip(CASES, preds):
        got = parse(p); ok = got is not None and all(got.get(f) == gold[f] for f in F)
        em += ok
        if not ok: fails.append({"input": x, "gold": gold, "raw_output": p})
    print(f"[{name}] exact match {em}/{len(CASES)} = {em/len(CASES):.2f}")
    return {"exact_match": round(em / len(CASES), 3), "n": len(CASES), "failures": fails}

base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev)
res = {"base": run(base, "BASE")}
ft = PeftModel.from_pretrained(base, "lora_adapter")
res["finetuned"] = run(ft, "FINE-TUNED")
for f in res["finetuned"]["failures"]:
    print("\nINPUT :", f["input"], "\nGOLD  :", f["gold"], "\nOUTPUT:", f["raw_output"])
json.dump(res, open("hard_results.json", "w"), indent=2, ensure_ascii=False)
