import json

records = [json.loads(l) for l in open("output/dapo/training_log.jsonl") if l.strip()]
top = sorted(records, key=lambda r: r["mean_reward"], reverse=True)[:10]
print("Top 10 checkpoints by mean_reward:")
print(f"{'Iter':>5}  {'Reward':>7}  {'Format':>7}  {'Loss':>8}  {'Filtered':>8}")
for r in top:
    print(f"{r['iteration']:>5}  {r['mean_reward']:>7.4f}  {r['mean_format_reward']:>6.1%}  {r['loss']:>8.4f}  {r.get('n_filtered',0):>8}")
