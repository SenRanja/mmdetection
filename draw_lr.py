import json
import matplotlib.pyplot as plt

# ----------- Configuration -----------------
log_file = r"C:\Users\shenyanjian\Downloads\new\output (1)\output\20251114_122202\vis_data\scalars.json"   # Replace with your log file
metrics_to_plot = [
    "loss",
    "loss_cls",
    "loss_bbox",
    "loss_rpn_cls",
    "loss_rpn_bbox",
    "lr",
    "acc",
    "coco/bbox_mAP_50",
    "coco/bbox_mAP_75",
]
# ---------------------------------

# Read line-by-line JSON
logs = []
with open(log_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            logs.append(json.loads(line))
        except:
            # Ignore non-JSON lines
            continue

print(f"Loaded {len(logs)} lines of logs.")

# Extract all keys
all_keys = set()
for entry in logs:
    all_keys.update(entry.keys())

print("Available metrics in file:")
print(all_keys)

# Automatic plotting functions
def plot_metric(metric_name, logs):
    xs, ys = [], []
    for entry in logs:
        if metric_name in entry:
            # epoch takes precedence; if it doesn't exist, use iter.
            x = entry.get("epoch", entry.get("iter", None))
            if x is not None:
                xs.append(x)
                ys.append(entry[metric_name])

    if len(xs) == 0:
        print(f"[Skip] Metric '{metric_name}' not found.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker='o')
    plt.title(metric_name)
    plt.xlabel("epoch")
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{metric_name}.png")
    plt.close()
    print(f"[Saved] {metric_name}.png")

# Draw all specified indicators
for m in metrics_to_plot:
    plot_metric(m, logs)
