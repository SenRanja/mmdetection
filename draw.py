import json
import matplotlib.pyplot as plt
import pandas as pd

# ====== 读取 scalars.json ======
log_file = r"C:\Users\shenyanjian\Downloads\mmdetection-dataclened\output (1)\output\20251113_095928\vis_data\scalars.json"   # 修改为你的路径

records = []
with open(log_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            records.append(json.loads(line.strip()))
        except:
            pass

df = pd.DataFrame(records)

# ====== 选择要画的指标 ======
metrics = ["loss", "acc", "loss_cls", "loss_rpn_cls", "loss_rpn_bbox", "loss_bbox", "lr"]

# 过滤掉不存在的字段
metrics = [m for m in metrics if m in df.columns]

# ====== 绘图 ======
plt.figure(figsize=(12, 6))

for m in metrics:
    plt.plot(df["step"], df[m], label=m)

plt.xlabel("Step / Iter")
plt.ylabel("Value")
plt.title("Training Scalars Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
