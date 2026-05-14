import pandas as pd, numpy as np, json, re, time, sys, os, warnings
warnings.filterwarnings('ignore')
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLE = 15

print(f"Loading {MODEL_ID} on {DEVICE}...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)
mdl.eval()
print("Model loaded.")

df = pd.read_csv("data/simulated_daily.csv")
stations = sorted(df['station'].unique())[:N_SAMPLE]

system_msg = (
    "You are a transportation data analyst. Identify the cause(s) of daily ridership change. "
    "Use these rules:\n"
    "- rain (precip > 1.0mm): rain usually REDUCES ridership (negative delta)\n"
    "- heat (temp > 28C): heat REDUCES ridership (fewer people go out in extreme heat)\n"
    "- wind (wind > 10m/s): strong wind REDUCES ridership\n"
    "- weekend (is_weekend=1): weekend often REDUCES ridership (commuters stay home)\n"
    "- neighbor_spillover (|neighbor_delta| > 50): large activity at nearby stations causes spillover\n"
    "- none: no cause applies (choose this when all values are near-normal and delta is small)\n\n"
    "Output ONLY a JSON array (no code fences, no extra text). "
    "Each item: {\"date\":\"YYYY-MM-DD\",\"cause_weather\":[\"label1\",\"label2\"],\"note\":\"reason\"}. "
    "Select all labels that apply. Use [\"none\"] if no cause applies."
)

def build_prompt(station_df):
    rows = []
    for _, row in station_df.iterrows():
        rows.append(
            f"{row['date']} | "
            f"delta={row['delta']:+.0f} | "
            f"precip={row['precip_mm']:.1f}mm | "
            f"temp={row['temp_c']:.1f}C | "
            f"wind={row['wind_ms']:.1f}m/s | "
            f"weekend={int(row['is_weekend'])} | "
            f"neighbor_delta={row['neighbor_delta']:+.0f}"
        )
    table_str = "\n".join(rows)
    return (
        f"Station: {station_df.iloc[0]['station']}\n\n"
        f"Daily data (7 days):\n"
        f"date | delta(ridership) | precip | temp | wind | weekend | neighbor_delta\n"
        f"{table_str}\n\n"
        "For each date, apply the thresholds to identify all causes. "
        "Return a JSON array of exactly 7 objects (one per date, in order)."
    )

def extract_labels(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [item.get("cause_weather", []) for item in data]
        elif isinstance(data, dict):
            return [data.get("cause_weather", [])]
    except json.JSONDecodeError:
        pass
    m = re.findall(r'"cause_weather"\s*:\s*(\[[^\]]+?\])', text)
    if m:
        try:
            return [json.loads(x) for x in m]
        except:
            pass
    return None

results = []
hit_scores = []
neighbor_acc_list = []
skip = 0
GT_COLS = ['gt_rain', 'gt_hot', 'gt_wind', 'gt_weekend', 'gt_neighbor']
LABELS = ['rain', 'heat', 'wind', 'weekend', 'neighbor_spillover']
per_label_stats = {lbl: {'tp': 0, 'fp': 0, 'fn': 0} for lbl in LABELS}

for st in stations:
    st_df = df[df['station'] == st].sort_values('date').reset_index(drop=True)
    prompt_text = build_prompt(st_df)
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_text},
    ]
    input_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(input_text, return_tensors="pt", truncation=True, max_length=4096).to(DEVICE)
    with torch.no_grad():
        outputs = mdl.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    response = tok.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    sys.stdout.write(f"\r{st}: done ")
    sys.stdout.flush()

    parsed = extract_labels(response)
    if parsed is None or len(parsed) != len(st_df):
        sys.stdout.write(f"[parse fail] ")
        sys.stdout.flush()
        skip += 1
        continue

    for i in range(len(st_df)):
        row = st_df.iloc[i]
        pred_labels = set(parsed[i]) if isinstance(parsed[i], list) else set()
        gt_labels = set()
        for col, label in zip(GT_COLS, LABELS):
            if row[col]:
                gt_labels.add(label)
        if not gt_labels:
            gt_labels.add("none")
        pred_labels = {l for l in pred_labels if l != "none"}

        hit = 1 if (gt_labels - {"none"}) & pred_labels else 0
        hit_scores.append(hit)

        for lbl in LABELS:
            in_pred = lbl in pred_labels
            in_gt = lbl in gt_labels
            if in_pred and in_gt:
                per_label_stats[lbl]['tp'] += 1
            elif in_pred and not in_gt:
                per_label_stats[lbl]['fp'] += 1
            elif not in_pred and in_gt:
                per_label_stats[lbl]['fn'] += 1

        if "neighbor_spillover" in gt_labels:
            neighbor_acc_list.append(1 if "neighbor_spillover" in pred_labels else 0)

        results.append({
            'station': st,
            'date': row['date'],
            'gt': list(gt_labels),
            'pred': list(pred_labels),
            'hit': hit,
        })
    pd.DataFrame(results).to_csv("data/llm_results_partial.csv", index=False)

hit_rate = np.mean(hit_scores) if hit_scores else 0
per_label_f1 = {}
for lbl in LABELS:
    s = per_label_stats[lbl]
    tp, fp, fn = s['tp'], s['fp'], s['fn']
    if tp + fp + fn > 0:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_label_f1[lbl] = f1
    else:
        per_label_f1[lbl] = 0
macro_f1 = np.mean(list(per_label_f1.values())) if per_label_f1 else 0
neighbor_acc = np.mean(neighbor_acc_list) if neighbor_acc_list else 0

print(f"\n\n===== LLM Evaluation Results (Qwen2.5-7B-Instruct) =====")
print(f"Stations: {len(stations)}, Samples: {len(hit_scores)}, Skipped: {skip}")
print(f"Hit Rate: {hit_rate:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
for lbl, f1 in per_label_f1.items():
    print(f"  F1({lbl}): {f1:.4f}")
print(f"Neighbor Acc: {neighbor_acc:.4f}")

np.savez("data/llm_results.npz",
         hit_rate=hit_rate, macro_f1=macro_f1, neighbor_acc=neighbor_acc,
         n_samples=len(hit_scores), n_skipped=skip)
print(f"Saved to data/llm_results.npz")
