import joblib
import pandas as pd
from pathlib import Path



def load_csv_df(
    df_path: str | Path,
) -> pd.DataFrame:
    """Load a csv file into a dataframe with error handling.

    Firstly finding file by df_path argument (pathlib.Path).
    Then trying read this dataframe.

    1. if dataframe found:
        - printing success message
        - printing dataframe stats (shape and memory usage)
        - return dataframe
    2. otherwise:
        - printing failure message
        - raises error

    function required arguments:
    1. df_path (str / pathlib.Path) file path to dataframe:

    function returns:
    1. df (pd.DataFrame) founded dataframe by df_path:
    """

    df_path = Path(df_path)

    if not df_path.is_file():
        raise FileNotFoundError(f"File not found: {df_path}")

    try:
        df = pd.read_csv(df_path)
        print(f"Successfully loaded dataframe: {df_path}")

    except Exception as e:
        print(f"Failed to load dataframe: {df_path}")
        raise e

    df_memory_usage = df.memory_usage(deep=True).sum() / 1024 ** 2

    print(f"Dataframe has: {df.shape[0]} rows / {df.shape[1]} columns")
    print(f"Dataframe size: {df_memory_usage:.3f}MB")

    return df



def save_processed_artifacts(
    scaled_df: pd.DataFrame,
    pca_df: pd.DataFrame,
    scaler_obj,
    pca_obj,
    data_output_path: str | Path,
    model_output_path: str | Path,
) -> None:
    """Save processed dataframes and transformer objects with error handling.

    Firstly creating output directories for data and models by data_output_path
    and model_output_path arguments (pathlib.Path).
    Then trying to save dataframes to CSV files and model objects to pickle files.

    1. if saving succeeds:
        - exports scaled_df and pca_df to CSV files
        - dumps scaler_obj and pca_obj to pickle files
        - printing success messages with resolved paths
    2. otherwise:
        - printing failure messages
        - raises error

    function required arguments:
    1. scaled_df (pd.DataFrame) scaled feature dataframe
    2. pca_df (pd.DataFrame) PCA transformed feature dataframe
    3. scaler_obj (object) fitted scaler transformer
    4. pca_obj (object) fitted PCA transformer
    5. data_output_path (str / pathlib.Path) directory path for output CSV files
    6. model_output_path (str / pathlib.Path) directory path for model files

    function returns nothing.
    """

    try:
        data_dir = Path(data_output_path)
        model_dir = Path(model_output_path)

        data_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        scaled_df.to_csv(data_dir / "scaled_df.csv", index=False)
        pca_df.to_csv(data_dir / "pca_df.csv", index=False)

        joblib.dump(scaler_obj, model_dir / "scaler.pkl")
        joblib.dump(pca_obj, model_dir / "pca.pkl")

        print(f"Data saved to: {data_dir.resolve()}")
        print(f"Models saved to: {model_dir.resolve()}")

    except Exception as e:
        print("Failed to save artifacts.")
        raise e