import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal
from pathlib import Path

FIGURES_PATH = Path.cwd().parent / "reports" / "figures"
FIGURES_PATH.mkdir(parents=True, exist_ok=True)


def drop_cols_above_threshold(df: pd.DataFrame, threshold=0.9, verbose=0) -> pd.DataFrame:
    """
    Drops columns from the DataFrame that have a percentage of missing values above the specified threshold.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    threshold (float): The threshold for dropping columns (between 0 and 1).
    verbose (int): Verbosity level. If greater than 0, prints the names of dropped columns.

    Returns:
    pd.DataFrame: The DataFrame with columns dropped.
    """
    if threshold < 0 or threshold > 1:
        raise ValueError("Threshold must be between 0 and 1.")
    missing_percent = df.isnull().mean()
    cols_to_drop = missing_percent[missing_percent > threshold].index
    df_dropped = df.drop(columns=cols_to_drop)
    if verbose > 0:
        print(f"Dropped columns: {list(cols_to_drop)}")
    return df_dropped


def join_dfs_on_key(dfs: list[pd.DataFrame], key: str,
                    how: Literal["left", "right", "inner", "outer", "cross"] = 'inner', **kwargs) -> pd.DataFrame:
    """
    Joins multiple DataFrames on a specified key using the specified join method.

    Parameters:
    dfs (list): List of DataFrames to join.
    key (str): The column name to join on.
    how (str): Type of join to perform ('left', 'right', 'inner', 'outer', 'cross').
    **kwargs: Additional keyword arguments to pass to the merge function.

    Returns:
    pd.DataFrame: The joined DataFrame.
    """
    if not dfs:
        raise ValueError("The list of DataFrames is empty.")

    joined_df = dfs[0]
    for df in dfs[1:]:
        joined_df = joined_df.merge(df, on=key, how=how, **kwargs)

    return joined_df


def null_value_chart1(df: pd.DataFrame) -> None:
    """
    Plots a heatmap showing the locations of missing values in the DataFrame.
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    """

    name = df.__class__.__name__

    if (FIGURES_PATH / f"{name}_null_value_chart.png").exists():
        img = plt.imread(FIGURES_PATH / f"{name}_null_value_chart.png")
        plt.figure(figsize=(20, 9))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        return

    if sum(df.isna().sum()) > 0:
        plt.figure(figsize=(20, 9))
        sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap="Oranges")
        plt.title(f"Missing values in {name} dataset")
        plt.savefig(FIGURES_PATH / f"{name}_null_value_chart.png")
        plt.show()
    else:
        print(f"There are no missing values in the {name} dataset")


def null_value_chart(df: pd.DataFrame, name: str = "dataset", subdir: str = "", save: bool = True) -> None:
    """
    Plots a heatmap showing the locations of missing values in the DataFrame.
    Args:
        df: pd.DataFrame: The input DataFrame.
        name: str: The name of the dataset for labeling and saving the figure.
        save: bool: Whether to save the figure to disk.

    """
    directory = FIGURES_PATH / subdir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}_null_value_chart.png"

    if path.exists() and save:
        img = plt.imread(path)
        plt.figure(figsize=(20, 9))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        return

    if df.isna().any().any():
        plt.figure(figsize=(20, 9))
        sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap="Oranges")
        plt.title(f"Missing values in {name}")
        if save:
            plt.savefig(path, bbox_inches="tight")
        plt.show()
    else:
        print(f"There are no missing values in the {name}")
