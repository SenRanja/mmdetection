import json
import matplotlib.pyplot as plt

# ====== Configuration: Your scalars.json file path ======
json_file = r"C:\Users\shenyanjian\Downloads\new\output (1)\output\20251114_122202\vis_data\scalars.json"  # Replace with your log file

# ====== Read JSON line by line ======
logs = []
with open(json_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            logs.append(json.loads(line))
        except:
            pass

print(f"Loaded {len(logs)} JSON entries.")

# ====== Automatically find all keys in coco/bbox_mAP* ======
map_keys = set()
for entry in logs:
    for k in entry.keys():
        if "coco/bbox_mAP" in k:
            map_keys.add(k)

map_keys = sorted(map_keys)
print("Detected mAP keys:", map_keys)

# ====== X-axis (epoch first, step second) ======
x_vals = []
for entry in logs:
    if "epoch" in entry:
        x_vals.append(entry["epoch"])
    elif "step" in entry:
        x_vals.append(entry["step"])
    else:
        x_vals.append(len(x_vals))  # fallback

# ====== Plot each mAP curve individually ======
for key in map_keys:
    y_vals = [entry.get(key, None) for entry in logs]

    plt.figure(figsize=(8,5))
    plt.plot(x_vals, y_vals, marker='o')
    plt.title(key)
    plt.xlabel("Epoch / Step")
    plt.ylabel(key)
    plt.grid(True)
    plt.tight_layout()
    out_name = key.replace("/", "_") + ".png"
    plt.savefig(out_name)
    plt.close()
    print(f"Saved {out_name}")

# ====== Plot all mAP curves on one graph ======
plt.figure(figsize=(10,6))
for key in map_keys:
    y_vals = [entry.get(key, None) for entry in logs]
    plt.plot(x_vals, y_vals, marker='o', label=key)

plt.title("COCO mAP Metrics")
plt.xlabel("Epoch / Step")
plt.ylabel("mAP Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("coco_mAP_all.png")
plt.close()
print("Saved coco_mAP_all.png")
