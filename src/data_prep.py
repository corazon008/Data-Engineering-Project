import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal, List, Generator
from pathlib import Path
import numpy as np

FIGURES_PATH = Path.cwd().parent / "reports" / "figures"
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

class DataExplorer:
    def __init__(self) -> None:
        """
        Loads multiple datasets related to home credit default risk into pandas DataFrames.
        The datasets are stored as attributes of the class instance.
        """

        self.application_test = pd.read_csv("../home-credit-default-risk/application_test.csv")
        self.application_test = reduce_mem_usage(self.application_test)
        self.application_train = pd.read_csv("../home-credit-default-risk/application_train.csv")
        self.application_train = reduce_mem_usage(self.application_train)
        self.bureau = pd.read_csv("../home-credit-default-risk/bureau.csv")
        self.bureau = reduce_mem_usage(self.bureau)
        self.bureau_balance = pd.read_csv("../home-credit-default-risk/bureau_balance.csv")
        self.bureau_balance = reduce_mem_usage(self.bureau_balance)
        self.credit_card_balance = pd.read_csv("../home-credit-default-risk/credit_card_balance.csv")
        self.credit_card_balance = reduce_mem_usage(self.credit_card_balance)
        self.installments_payments = pd.read_csv("../home-credit-default-risk/installments_payments.csv")
        self.installments_payments = reduce_mem_usage(self.installments_payments)
        self.POS_CASH_balance = pd.read_csv("../home-credit-default-risk/POS_CASH_balance.csv")
        self.POS_CASH_balance = reduce_mem_usage(self.POS_CASH_balance)
        self.previous_application = pd.read_csv("../home-credit-default-risk/previous_application.csv")
        self.previous_application = reduce_mem_usage(self.previous_application)

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

        for dataset in self:
            dataset.drop_duplicates()

    def __iter__(self)-> Generator[pd.DataFrame]:
        """

        Returns:
        Generator that yields each dataset DataFrame.

        """
        for dataset in self.datasets:
            yield getattr(self, dataset)

    def __getitem__(self, item):
        return getattr(self, self.datasets[item])

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def items(self):
        """

        Returns:
        Generator that yields tuples of dataset names and their corresponding DataFrames.

        """
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

def reduce_mem_usage(df):
    """
    Itère sur toutes les colonnes d'un DataFrame et réduit la précision
    des types numériques (int et float) pour diminuer la consommation de mémoire.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Usage mémoire initial du DataFrame: {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        # Traiter uniquement les colonnes numériques
        if col_type != object and col_type != str and col_type != bool:
            c_min = df[col].min()
            c_max = df[col].max()

            # --- Conversion des entiers (Integers) ---
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)

            # --- Conversion des décimaux (Floats) ---
            else:
                # La majorité de vos colonnes d'agrégats (mean, var, proportions) sont ici
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    # Conversion principale : float64 -> float32
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64) # Garder float64 si la précision est nécessaire

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Usage mémoire final du DataFrame: {end_mem:.2f} MB")
    print(f"Mémoire réduite de {(start_mem - end_mem) / start_mem * 100:.1f} %")

    return df