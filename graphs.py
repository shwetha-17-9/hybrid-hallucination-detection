import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

# Histogram of hallucination scores
plt.figure()

plt.hist(df["Final Score"], bins=20)

plt.title("Distribution of Hallucination Scores")
plt.xlabel("Hallucination Score")
plt.ylabel("Frequency")

plt.savefig("score_distribution.png")

thresholds = [0.60, 0.65, 0.70]

rates = [
    (df["Final Score"] > 0.60).mean(),
    (df["Final Score"] > 0.65).mean(),
    (df["Final Score"] > 0.70).mean()
]

plt.figure()

plt.bar(["0.60","0.65","0.70"], rates)

plt.title("Detection Rate vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("Detection Rate")

plt.savefig("threshold_comparison.png")

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

plt.figure()

plt.boxplot(df["Final Score"])

plt.title("Hallucination Score Distribution (Box Plot)")
plt.ylabel("Hallucination Score")

plt.savefig("hallucination_boxplot.png")