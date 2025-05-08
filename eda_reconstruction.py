import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA


def read_ecg_data(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        data = []
        for line in lines:
            # Split the line into values and convert to float
            values = list(map(float, line.strip().split()))
            data.append(values)
    return pd.DataFrame(data)


def visualize_ecg_data(folder: Path, X, y):
    folder.mkdir(parents=True, exist_ok=True)

    # Plot class distribution
    class_distribution = y.value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_distribution.index, y=class_distribution.values)
    plt.title("Class Distribution in ECG Data")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(folder / "class_distribution.png")

    # Plot each class separately
    plt.figure(figsize=(20, 10))

    if len(y.unique()) <= 4:
        width, height = 1, len(y.unique())
    else:
        width, height = 2, len(y.unique()) // 2 + len(y.unique()) % 2

    for i, label in enumerate(y.unique()):
        plt.subplot(width, height, i + 1)
        plt.title(f"Class {label}")
        plt.plot(X[y == label].T, color="blue", alpha=0.1)
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.grid()
    plt.tight_layout()
    plt.savefig(folder / "class_separation.png")

    # Plot scatterplot using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="Set2")
    plt.legend(title="Class")
    plt.tight_layout()
    plt.savefig(folder / "scatterplot.png")


def main():
    ecg_data_path = Path("data/reconstruction")
    test_data = pd.read_csv(ecg_data_path / "ecg5000_test.txt", sep="\s+", header=None)
    train_data = pd.read_csv(
        ecg_data_path / "ecg5000_train.txt", sep="\s+", header=None
    )
    data = pd.concat([train_data, test_data], axis=0)
    print("Data shape:", data.shape)

    analysis_folder = Path("analysis/reconstruction")
    visualize_ecg_data(
        analysis_folder / "before-preprocess",
        data.iloc[:, 1:],
        data.iloc[:, 0],
    )

    # Combine classes
    data[0] = data[0].map(
        {
            1: 1,  # Normal
            2: 0,  # Anomaly
            3: 0,
            4: 0,
            5: 0,
        }
    )
    visualize_ecg_data(
        analysis_folder / "after-preprocess",
        data.iloc[:, 1:],
        data.iloc[:, 0],
    )

    # Write the processed data to a new CSV file
    data.to_csv(ecg_data_path / "ecg.csv", index=False, header=False)


if __name__ == "__main__":
    main()
