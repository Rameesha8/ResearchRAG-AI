import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data/processed_text/arxiv_documents.jsonl"
REPORTS_DIR = PROJECT_ROOT / "reports"
EDA_DIR = REPORTS_DIR / "eda_exports"
SUMMARY_JSON_PATH = EDA_DIR / "eda_summary.json"
CATEGORY_CSV_PATH = EDA_DIR / "top_categories.csv"
TITLE_PLOT_PATH = EDA_DIR / "title_length_distribution.png"
ABSTRACT_PLOT_PATH = EDA_DIR / "abstract_length_distribution.png"
CATEGORY_PLOT_PATH = EDA_DIR / "top_category_distribution.png"


def load_dataframe() -> pd.DataFrame:
    """Load processed arXiv documents into a dataframe for EDA."""
    records = []
    with DATA_PATH.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                records.append(json.loads(line))

    dataframe = pd.DataFrame(records)
    dataframe["primary_category"] = dataframe["categories"].fillna("").str.split().str[0]
    dataframe["title_length"] = dataframe["title"].fillna("").str.len()
    dataframe["abstract_length"] = dataframe["abstract"].fillna("").str.len()
    return dataframe


def save_category_plot(df: pd.DataFrame) -> None:
    top_categories = df["primary_category"].value_counts().head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_categories.values, y=top_categories.index, hue=top_categories.index, palette="crest", legend=False)
    plt.title("Top 10 Primary arXiv Categories")
    plt.xlabel("Document Count")
    plt.ylabel("Primary Category")
    plt.tight_layout()
    plt.savefig(CATEGORY_PLOT_PATH, dpi=200)
    plt.close()
    top_categories.rename_axis("category").reset_index(name="count").to_csv(CATEGORY_CSV_PATH, index=False)


def save_length_plots(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    sns.histplot(df["title_length"], bins=30, kde=True, color="#1f77b4")
    plt.title("Distribution of Title Lengths")
    plt.xlabel("Title Length (characters)")
    plt.tight_layout()
    plt.savefig(TITLE_PLOT_PATH, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.histplot(df["abstract_length"], bins=40, kde=True, color="#d95f02")
    plt.title("Distribution of Abstract Lengths")
    plt.xlabel("Abstract Length (characters)")
    plt.tight_layout()
    plt.savefig(ABSTRACT_PLOT_PATH, dpi=200)
    plt.close()


def save_summary(df: pd.DataFrame) -> None:
    summary = {
        "documents": int(len(df)),
        "unique_primary_categories": int(df["primary_category"].nunique()),
        "avg_title_length": float(round(df["title_length"].mean(), 2)),
        "avg_abstract_length": float(round(df["abstract_length"].mean(), 2)),
        "top_categories": Counter(df["primary_category"]).most_common(10),
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def export_eda_assets() -> None:
    """Generate EDA summary files and charts for reports and slides."""
    if not DATA_PATH.exists():
        print(f"Processed data not found: {DATA_PATH}")
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    dataframe = load_dataframe()
    save_category_plot(dataframe)
    save_length_plots(dataframe)
    save_summary(dataframe)
    print(f"Saved EDA assets to {EDA_DIR}")


if __name__ == "__main__":
    export_eda_assets()
