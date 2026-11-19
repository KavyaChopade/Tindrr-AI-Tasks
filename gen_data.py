"""Generates a synthetic dataset: plain-English expense text -> fixed JSON.
Run: python gen_data.py   -> writes train.jsonl (800) and test.jsonl (100)
Test = 50 'seen-style' phrasings + 50 'unseen-style' phrasings (new wording, Hinglish, '2k' shorthand)
so we can measure generalisation, not just memorisation."""
import json, random
random.seed(42)

MERCHANTS = {
    "food": ["Zomato", "Swiggy", "Domino's", "Barista", "Haldiram's"],
    "transport": ["Uber", "Ola", "Rapido", "IRCTC", "Metro"],
    "shopping": ["Amazon", "Flipkart", "Myntra", "Decathlon", "Croma"],
    "bills": ["Airtel", "Jio", "Tata Power", "Netflix", "Vi"],
    "entertainment": ["BookMyShow", "Spotify", "PVR", "Steam", "Hotstar"],
    "health": ["Apollo Pharmacy", "PharmEasy", "Practo", "Cult.fit", "1mg"],
}
PAY = {"cash": ["cash", "in cash"], "upi": ["UPI", "GPay", "PhonePe"],
       "card": ["credit card", "debit card", "my card"]}

def rand_amount():
    if random.random() < 0.6:
        return random.randrange(50, 5001, 10)
    return round(random.uniform(20, 3000), 2)

def fmt_amt(a):
    s = str(a)
    return random.choice([f"₹{s}", f"Rs {s}", f"{s} rupees", f"Rs.{s}"])

SEEN = [
    "Spent {amt} at {m} using {p}",
    "Paid {amt} to {m} via {p}",
    "I bought something from {m} for {amt}, paid by {p}",
    "{m}: {amt} ({p})",
    "Please log an expense of {amt} on {m}, payment method {p}",
    "Just paid {m} {amt} with {p}",
]
UNSEEN = [
    "{p} se {m} pe {amt} kharch kiye",
    "{m} bill came to {amt} and I cleared it using {p}",
    "Note down - {p} payment, {amt}, merchant {m}",
    "gave {m} {amt} yesterday, through {p}",
    "Settled my {m} dues: {amt}. Mode: {p}",
]

def make(templates, allow_k=False):
    cat = random.choice(list(MERCHANTS))
    m = random.choice(MERCHANTS[cat])
    pay = random.choice(list(PAY))
    p = random.choice(PAY[pay])
    a = rand_amount()
    amt_text = fmt_amt(a)
    if allow_k and isinstance(a, int) and a % 1000 == 0 and random.random() < 0.8:
        amt_text = f"{a // 1000}k"
    text = random.choice(templates).format(amt=amt_text, m=m, p=p)
    out = {"amount": a, "category": cat, "merchant": m, "payment": pay}
    return {"input": text, "output": json.dumps(out, separators=(",", ":"), ensure_ascii=False)}

def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

train = [make(SEEN) for _ in range(800)]
test_seen = [dict(make(SEEN), style="seen") for _ in range(50)]
# unseen set: force some round-thousand amounts so '2k' shorthand appears
test_unseen = []
while len(test_unseen) < 50:
    r = make(UNSEEN, allow_k=True)
    test_unseen.append(dict(r, style="unseen"))

# leakage check: no test input may appear in train
train_inputs = {r["input"] for r in train}
test = [r for r in test_seen + test_unseen if r["input"] not in train_inputs]
write("train.jsonl", train)
write("test.jsonl", test)
print(len(train), "train;", len(test), "test")
