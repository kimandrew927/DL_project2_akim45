"""Generate training_curve.png from training_log.json."""
import json
import matplotlib.pyplot as plt

with open("training_log.json") as f:
    log = json.load(f)

epochs     = [h["epoch"]      for h in log["history"]]
train_loss = [h["train_loss"] for h in log["history"]]
val_loss   = [h["val_loss"]   for h in log["history"]]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(epochs, train_loss, marker="o", label="Train loss")
ax.plot(epochs, val_loss,   marker="s", label="Val loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-Entropy Loss")
ax.set_title("TinyGPT Training Curve")
ax.set_xticks(epochs)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("training_curve.png", dpi=150)
print("Saved training_curve.png")
