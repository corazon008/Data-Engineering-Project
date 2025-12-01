import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal, List
from pathlib import Path

FIGURES_PATH = Path.cwd().parent / "reports" / "figures"
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

class DataExplorer:
    def __init__(self) -> None:
        """
        Loads multiple datasets related to home credit default risk into pandas DataFrames.
        The datasets are stored as attributes of the class instance.
        """

        self.application_test = pd.read_csv("../home-credit-default-risk/application_test.csv")
        self.application_train = pd.read_csv("../home-credit-default-risk/application_train.csv")
        self.bureau = pd.read_csv("../home-credit-default-risk/bureau.csv")
        self.bureau_balance = pd.read_csv("../home-credit-default-risk/bureau_balance.csv")
        self.credit_card_balance = pd.read_csv("../home-credit-default-risk/credit_card_balance.csv")
        self.installments_payments = pd.read_csv("../home-credit-default-risk/installments_payments.csv")
        self.POS_CASH_balance = pd.read_csv("../home-credit-default-risk/POS_CASH_balance.csv")
        self.previous_application = pd.read_csv("../home-credit-default-risk/previous_application.csv")

        self.datasets: List[str] = [
            "application_test",
            "application_train",
            "bureau",
            "bureau_balance",
            "credit_card_balance",
            "installments_payments",
            "POS_CASH_balance",
            "previous_application"
        ]

    def __iter__(self):
        for dataset in self.datasets:
            yield getattr(self, dataset)

    def __getitem__(self, item):
        return getattr(self, self.datasets[item])

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def items(self):
        for dataset in self.datasets:
            yield dataset, getattr(self, dataset)

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

def null_cols_above_threshold(df: pd.DataFrame, threshold=0.7) -> None:
    """
    Prints the names of columns in the DataFrame that have a percentage of missing values above the specified threshold.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    threshold (float): The threshold for identifying columns (between 0 and 1).
    """

    for col in df.columns:
        missing_percentage = df[col].isna().mean()
        if missing_percentage > threshold:
            print(f"  {col}: {missing_percentage:.2f}% missing")


def join_dfs_on_key(left: pd.DataFrame, right: pd.DataFrame, key: str,
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
    if left.empty or right.empty:
        raise ValueError("Input DataFrames must not be empty.")

    joined_df = left.merge(right, on=key, how=how, **kwargs)
    return joined_df


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
