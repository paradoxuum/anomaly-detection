import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd


def main():
    data_path = Path("data/TravelTime_451.csv")
    data = pd.read_csv(data_path, parse_dates=["timestamp"])

    analysis_path = Path("analysis/forecasting")
    analysis_path.mkdir(parents=True, exist_ok=True)

    # Plot data
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x="timestamp", y="value")
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.savefig(analysis_path / "plot.png")


if __name__ == "__main__":
    main()
