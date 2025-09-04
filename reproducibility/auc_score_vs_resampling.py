import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/onur/Code/recovar/reproducibility/resampling_vs_score.csv")

# Plot
plt.figure(figsize=(8,5))
plt.plot(df["resample_eq_ratio"], df["roc_auc"], marker="o", linestyle="-", linewidth=2, markersize=6)

# Labels & title
plt.xlabel("Earthquake waveforms / Total waveforms", fontsize=12)
plt.ylabel("ROC AUC", fontsize=12)
plt.title("ROC AUC vs Resample Equalization Ratio", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.7)

plt.savefig("instance_auc_score_vs_resampling.png")