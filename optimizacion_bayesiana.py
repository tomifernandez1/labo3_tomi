from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
import time
import gc
import warnings
import os
import shutil
import pickle
import os
from glob import glob
import pandas as pd
import numpy as np
import lightgbm as lgb
import tempfile
import tqdm
import datetime
import warnings
import optuna
import io
from contextlib import redirect_stdout
from IPython.display import display, HTML
import holidays
from google.cloud import storage
from urllib.parse import urlparse
import multiprocessing
import logging
import shutil
import traceback
from optuna.exceptions import TrialPruned
import json

warnings.filterwarnings("ignore", message="DataFrame is highly fragmented*")

def fallback_latest_notebook():
    notebooks = glob("*.ipynb")
    if not notebooks:
        return None
    notebooks = sorted(notebooks, key=os.path.getmtime, reverse=True)
    return notebooks[0]



warnings.filterwarnings('ignore', category=FutureWarning)

class PipelineStep(ABC):
    """
    Abstract base class for pipeline steps.
    Each step in the pipeline must inherit from this class and implement the execute method.
    """
    def __init__(self, name: Optional[str] = None):
        """
        Initialize a pipeline step.

        Args:
            name (str): Name of the step for identification and logging purposes.
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def execute(self, pipeline: "Pipeline") -> None:
        """
        Execute the pipeline step.

        Args:
            pipeline (Pipeline): The pipeline instance that contains this step.
        """
        pass

    def save_artifact(self, pipeline: "Pipeline", artifact_name: str, artifact: Any) -> None:
        """
        Save an artifact produced by this step to the pipeline.

        Args:
            pipeline (Pipeline): The pipeline instance.
            artifact_name (str): Name to identify the artifact.
            artifact (Any): The artifact to save.
        """
        pipeline.save_artifact(artifact_name, artifact)


class Pipeline:
    """
    Main pipeline class that manages the execution of steps and storage of artifacts.
    """
    DEFAULT_BUCKET_NAME = "labo3_2025_tomifernandez"
    def __init__(self, steps: Optional[List] = None, use_gcs: bool = False, bucket_name: Optional[str] = None,experiment_name: Optional[str] = None):
        """Initialize the pipeline."""
        self.steps: List[PipelineStep] = steps if steps is not None else []
        self.artifacts: Dict[str, Any] = {}
        self.df: Optional[pd.DataFrame] = None        
        self.use_gcs = use_gcs
        self.bucket_name = bucket_name or self.DEFAULT_BUCKET_NAME
        self.storage_client = storage.Client() if self.use_gcs else None      
        self.last_step = None
                
        if self.use_gcs and not self.bucket_name:
            raise ValueError("Bucket name must be provided or set as DEFAULT_BUCKET_NAME.")
        
        #Configuracion del logging:
        self.experiment_name = experiment_name

        #Nombre del archivo de log
        self.log_filename = f"{self.experiment_name}_log.txt"

        #Configurar logger
        self.logger = logging.getLogger(f"PipelineLogger_{self.experiment_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # Evita duplicados si se reutiliza logger

        #Si no hay handlers previos, agregamos uno
        if not self.logger.handlers:
            file_handler = logging.FileHandler(self.log_filename)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
    def add_step(self, step: PipelineStep, position: Optional[int] = None) -> None:
        """
        Add a new step to the pipeline.

        Args:
            step (PipelineStep): The step to add.
            position (Optional[int]): Position where to insert the step. If None, appends to the end.
        """
        if position is not None:
            self.steps.insert(position, step)
        else:
            self.steps.append(step)

    def save_artifact(self, artifact_name: str, artifact: Any) -> None:
        """
        Save artifact to Google Cloud Storage if use_gcs is True,
        otherwise save it locally to ./home/tomifernandezlabo3/gcs-bucket.
        """
        if self.use_gcs:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(f"artifacts/{artifact_name}.pkl")

            # Serializa en memoria sin tocar el disco
            buffer = io.BytesIO()
            pickle.dump(artifact, buffer)
            buffer.seek(0)

            # Subida directa al bucket
            blob.upload_from_file(buffer, content_type='application/octet-stream')
            self.artifacts[artifact_name] = f"gs://{self.bucket_name}/artifacts/{artifact_name}.pkl"
        else:
            # Guardado local en /tmp
            local_dir = "/tmp"
            os.makedirs(local_dir, exist_ok=True)

            artifact_path = os.path.join(local_dir, f"{artifact_name}.pkl")
            with open(artifact_path, "wb") as f:
                pickle.dump(artifact, f)

            self.artifacts[artifact_name] = artifact_path

    def get_artifact(self, artifact_name: str) -> Any:
        """
        Retrieve a stored artifact from GCS or local storage.

        Args:
            artifact_name (str): Name of the artifact to retrieve.

        Returns:
            Any: The requested artifact.
        """
        artifact_path = self.artifacts.get(artifact_name)
        if artifact_path is None:
            raise ValueError(f"Artifact '{artifact_name}' not found.")

        # Caso GCS
        if isinstance(artifact_path, str) and artifact_path.startswith("gs://"):
            try:
                parsed = urlparse(artifact_path)
                bucket_name = parsed.netloc
                blob_path = parsed.path.lstrip('/')

                client = storage.Client()
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_path)

                buffer = io.BytesIO()
                blob.download_to_file(buffer)
                buffer.seek(0)

                return pickle.load(buffer)

            except Exception as e:
                raise RuntimeError(f"Failed to load artifact '{artifact_name}' from GCS: {e}")

        # Caso local
        else:
            artifact_path_local = os.path.join("/tmp", f"{artifact_name}.pkl")
            if not os.path.exists(artifact_path_local):
                raise FileNotFoundError(f"Local artifact file not found at {artifact_path_local}")
            try:
                with open(artifact_path_local, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load local artifact '{artifact_name}': {e}")
    
    def del_artifact(self, artifact_name: str, soft=True) -> None:
        """
        Delete a stored artifact and free memory.

        Args:
            artifact_name (str): Name of the artifact to delete.
        """
        del self.artifacts[artifact_name]
        if not soft:
            # Force garbage collection if not soft delete
            gc.collect()
    
    def before_step_callback(self) -> None:
        """
        Set a callback to be called before each step execution.
        """
        pass
        
    def after_step_callback(self) -> None:
        """
        Set a callback to be called after each step execution.
        """
        pass
        
    def after_last_step_callback(self) -> None:
        """
        Set a callback to be called after the last step execution.
        """
        pass
    
    def run(self, verbose: bool = True, last_step_callback: Callable = None) -> None:
        """
        Execute all steps in sequence and log execution time.
        Logs errors if any step fails.
        """
        for step in self.steps:
            if verbose:
                print(f"Executing step: {step.name}")
            self.logger.info(f"Executing step: {step.name}")    

            start_time = time.time()
            try:
                self.before_step_callback()
                step.execute(self)
                self.after_step_callback()
                end_time = time.time()
            except Exception as e:
                self.logger.error(f"Error in step '{step.name}': {e}", exc_info=True)
                raise  #corta el script

            if verbose:
                print(f"Step {step.name} completed in {end_time - start_time:.2f} seconds")
            self.logger.info(f"Step {step.name} completed in {end_time - start_time:.2f} seconds")
            self.last_step = step

            if step == self.steps[-1]:
                self.after_last_step_callback()
                if last_step_callback:
                    last_step_callback(self)

    def clear(self, collect_garbage: bool = False) -> None:
        """
        Clean up all artifacts and free memory.
        """
        if collect_garbage:
            del self.artifacts
            gc.collect()
        self.artifacts = {}
        self.last_step = None
        
class ReduceMemoryUsageStep(PipelineStep):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
                   
    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df
        initial_mem_usage = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                c_min = df[col].min()
                c_max = df[col].max()
                if pd.api.types.is_float_dtype(df[col]):
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                elif pd.api.types.is_integer_dtype(df[col]):
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
        
        final_mem_usage = df.memory_usage().sum() / 1024**2
        print('--- Memory usage before: {:.2f} MB'.format(initial_mem_usage))
        print('--- Memory usage after: {:.2f} MB'.format(final_mem_usage))
        print('--- Decreased memory usage by {:.1f}%\n'.format(100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage))
        pipeline.df = df 

class LoadDataFrameStep(PipelineStep):
    """
    Example step that loads a DataFrame.
    """
    def __init__(self, path: str, name: Optional[str] = None):
        super().__init__(name)
        self.path = path

    def execute(self, pipeline: Pipeline) -> None:
        df = pd.read_parquet(self.path, engine="pyarrow")
        df = df.drop(columns=["periodo"])
        pipeline.df = df
        
class LoadDataFrameFromPickleStep(PipelineStep):
    """
    Carga un DataFrame desde un archivo .pkl y lo guarda en pipeline.df.
    Si encuentra una columna que contenga 'target' en el nombre, la renombra a 'target'.
    """
    def __init__(self, path: str, name: Optional[str] = None):
        super().__init__(name)
        self.path = path

    def execute(self, pipeline: Pipeline) -> None:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"No se encontró el archivo: {self.path}")
        
        df = pd.read_pickle(self.path)

        # Renombrar columnas que contengan 'target' en el nombre a 'target'
        target_cols = [col for col in df.columns if 'target' in col.lower()]
        if target_cols:
            # Solo mantener la primera que se encuentra
            print(f"Renombrando columna '{target_cols[0]}' a 'target'")
            df.rename(columns={target_cols[0]: 'target'}, inplace=True)

        pipeline.df = df
        print(f"DataFrame cargado desde: {self.path} (shape: {df.shape})")
        
class WeightedSubsampleSeriesStep(PipelineStep):
    """
    Submuestrea series (customer_id, product_id) ponderando por el promedio de tn.
    Se priorizan combinaciones con mayor promedio en la selección aleatoria."""
    
    def __init__(
        self,
        tn_col: str = "tn",
        sample_fraction: float = 0.25,
        random_state: int = 42,
        name: Optional[str] = None
    ):
        super().__init__(name)
        self.tn_col = tn_col
        self.sample_fraction = sample_fraction
        self.random_state = random_state

    def execute(self, pipeline: "Pipeline") -> None:
        df = pipeline.df
        # 1. calculo agregado x serie
        series_avg_tn = (
            df.groupby(["customer_id", "product_id"])[self.tn_col]
            .sum()
            .reset_index(name="sum_tn")
        )
        # 2. Normalizar pesos
        series_avg_tn["sampling_weight"] = series_avg_tn["sum_tn"]
        # 3. Muestrear series con probabilidad proporcional al promedio
        sampled_series = series_avg_tn.sample(
            frac=self.sample_fraction,
            weights="sampling_weight",
            random_state=self.random_state
        )
        # 4. Filtrar dataset original
        df_filtered = df.merge(
            sampled_series[["customer_id", "product_id"]],
            on=["customer_id", "product_id"],
            how="inner")
        
        pipeline.df = df_filtered
        

class SplitCustomerProductByGroupStep(PipelineStep):
    """
    Divide el DataFrame original en N subsets, cada uno con todas las fechas pero solo un subconjunto de combinaciones (product_id, customer_id).
    """
    def __init__(self, n_splits=3, name: Optional[str] = None):
        super().__init__(name)
        self.n_splits = n_splits
        
    
    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df
        combos = df[["product_id", "customer_id"]].drop_duplicates().reset_index(drop=True)
        combos = combos.sample(frac=1, random_state=42).reset_index(drop=True)  # <-- SHUFFLE
        combos["subset"] = combos.index % self.n_splits
        df_subsets = {}
        for i in range(self.n_splits):
            combos_i = combos[combos["subset"] == i][["product_id", "customer_id"]]
            merged = df.merge(combos_i, on=["product_id", "customer_id"], how="inner")
            merged = merged.sort_values(["product_id", "customer_id", "fecha"])
            df_subsets[f"subset_{i+1}"] = merged
        pipeline.df_subsets = df_subsets
        print(f"Dividido en {self.n_splits} subsets y guardado en pipeline.df_subsets.")

class CastDataTypesStep(PipelineStep):
    def __init__(self, dtypes: Dict[str, str], name: Optional[str] = None):
        super().__init__(name)
        self.dtypes = dtypes

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df
        for col, dtype in self.dtypes.items():
            df[col] = df[col].astype(dtype)
        df.info()
        pipeline.df = df


class ChangeDataTypesStep(PipelineStep):
    def __init__(self, dtypes: Dict[str, str], name: Optional[str] = None):
        super().__init__(name)
        self.dtypes = dtypes

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df
        for original_dtype, dtype in self.dtypes.items():
            for col in df.select_dtypes(include=[original_dtype]).columns:
                df[col] = df[col].astype(dtype)
        df.info()
        pipeline.df = df


class FilterFirstDateStep(PipelineStep):
    def __init__(self, first_date: str, name: Optional[str] = None):
        super().__init__(name)
        self.first_date = first_date

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df
        df = df[df["fecha"] >= self.first_date]
        print(f"Filtered DataFrame shape: {df.shape}")
        pipeline.df = df
        
class ShareTNFeaturesStep(PipelineStep):
    """
    Crea features de share de tn respecto al total por product_id y por customer_id en cada fecha.
    """
    def __init__(self, tn_col="tn", name: Optional[str] = None):
        super().__init__(name)
        self.tn_col = tn_col

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df
        # Share respecto al total del producto en la fecha
        df["share_tn_product"] = df[self.tn_col] / df.groupby(["fecha", "product_id"])[self.tn_col].transform("sum")
        # Share respecto al total del customer en la fecha
        df["share_tn_customer"] = df[self.tn_col] / df.groupby(["fecha", "customer_id"])[self.tn_col].transform("sum")
        for cat in ["cat1", "cat2", "cat3", "brand"]:
            col_name = f"share_tn_{cat}"
            df[col_name] = df[self.tn_col] / df.groupby(["fecha", cat])[self.tn_col].transform("sum")        
        pipeline.df = df

class FeatureEngineeringLagStep(PipelineStep):
    def __init__(self, lags: List[int], columns: List, name: Optional[str] = None):
        super().__init__(name)
        self.lags = lags
        self.columns = columns

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df
        for col in self.columns:
            for lag in self.lags:
                df[f"{col}_lag_{lag}"] =  df.groupby(['product_id', 'customer_id'])[col].shift(lag)
        pipeline.df = df
        
class DiferenciaVsReferenciaStep(PipelineStep):
    """
    Crea features de diferencia entre el valor actual y una columna de referencia (lag, rolling mean, etc).
    Ejemplo: tn_diff_rolling_mean_3 = tn - tn_rolling_mean_3
    """
    def __init__(self, columns: list, ref_types: list, window: list, name: Optional[str] = None):
        super().__init__(name)
        self.columns = columns
        self.ref_types = ref_types  # Ej: ["lag", "rolling_mean"]
        self.window = window      # Ej: [1, 3, 6, 12]

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df
        for col in self.columns:
            for ref_type in self.ref_types:
                for window in self.window:
                    ref_col = f"{col}_{ref_type}_{window}"
                    diff_col = f"{col}_diff_{ref_type}_{window}"
                    if ref_col in df.columns:
                        df[diff_col] = df[col] - df[ref_col]
        pipeline.df = df
        
class DiferenciaRelativaVsReferenciaStep(PipelineStep):
    """
    Crea features de diferencia relativa entre el valor actual y una columna de referencia (lag, rolling mean, etc).
    Ejemplo: tn_reldiff_rolling_mean_3 = (tn / tn_rolling_mean_3) - 1
    """
    def __init__(self, columns: list, ref_types: list, window: list, name: Optional[str] = None):
        super().__init__(name)
        self.columns = columns
        self.ref_types = ref_types  # Ej: ["lag", "rolling_mean"]
        self.window = window        # Ej: [1, 3, 6, 12]

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df
        for col in self.columns:
            for ref_type in self.ref_types:
                for window in self.window:
                    ref_col = f"{col}_{ref_type}_{window}"
                    reldiff_col = f"{col}_reldiff_{ref_type}_{window}"
                    if ref_col in df.columns:
                        df[reldiff_col] = np.where(
                            df[ref_col] != 0,
                            df[col] / df[ref_col] - 1,
                            np.nan
                        )
        pipeline.df = df

class RollingMeanFeatureStep(PipelineStep):
    def __init__(self, window: List[int], columns: List[str], name: Optional[str] = None, n_jobs=-1):
        super().__init__(name)
        self.window = window
        self.columns = columns
        self.n_jobs = n_jobs

    def _compute_rolling_mean(self, args):
        col, window, df_small = args
        grouped = df_small.groupby(['product_id', 'customer_id'])
        return (
            f'{col}_rolling_mean_{window}',
            grouped[col].transform(lambda x: x.rolling(window, min_periods=window).mean())
        )

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df
        df = df.sort_values(by=['product_id', 'customer_id', 'fecha'])

        tasks = []
        for col in self.columns:
            for window in self.window:
                df_small = df[['product_id', 'customer_id', 'fecha', col]].copy()
                tasks.append((col, window, df_small))

        if self.n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=self.n_jobs) as pool:
            results = pool.map(self._compute_rolling_mean, tasks)

        for col_name, series in results:
            df[col_name] = series

        # Guardar el resultado como artifact
        pipeline.df = df
        
class CustomerIdMeanTnAndRequestByDateStep(PipelineStep):
    """
    Agrega columnas con el promedio de tn y cust_request_qty de cada customer_id por fecha.
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df
        # Promedio de tn por customer y fecha
        mean_tn = (
            df.groupby(["customer_id", "fecha"])["tn"]
            .mean()
            .reset_index()
            .rename(columns={"tn": "customer_mean_tn_by_fecha"})
        )
        # Promedio de cust_request_qty por customer y fecha
        mean_cust_req = (
            df.groupby(["customer_id", "fecha"])["cust_request_qty"]
            .mean()
            .reset_index()
            .rename(columns={"cust_request_qty": "customer_mean_cust_request_qty_by_fecha"})
        )
        # Merge al dataframe original
        df = df.merge(mean_tn, on=["customer_id", "fecha"], how="left")
        df = df.merge(mean_cust_req, on=["customer_id", "fecha"], how="left")
        pipeline.df = df

class ProductIdMeanTnAndRequestByCustomerStep(PipelineStep):
    """
    Agrega columnas con el promedio de tn y cust_request_qty por customer para cada product_id.
    Es decir, para cada fila, la media de tn y cust_request_qty de ese producto entre todos sus customers.
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df
        # Promedio de tn por product_id (entre sus customers)
        mean_tn = (
            df.groupby("product_id")["tn"]
            .mean()
            .reset_index()
            .rename(columns={"tn": "product_mean_tn_by_customer"})
        )
        # Promedio de cust_request_qty por product_id (entre sus customers)
        mean_cust_req = (
            df.groupby("product_id")["cust_request_qty"]
            .mean()
            .reset_index()
            .rename(columns={"cust_request_qty": "product_mean_cust_request_qty_by_customer"})
        )
        # Merge al dataframe original
        df = df.merge(mean_tn, on="product_id", how="left")
        df = df.merge(mean_cust_req, on="product_id", how="left")
        pipeline.df = df

class RollingMaxFeatureStep(PipelineStep):
    def __init__(self, window: List[int], columns: List[str], name: Optional[str] = None, n_jobs=-1):
        super().__init__(name)
        self.window = window
        self.columns = columns
        self.n_jobs = n_jobs

    def _compute_rolling_features(self, args):
        col, window, df_small = args
        grouped = df_small.groupby(['product_id', 'customer_id'])
        rolling_max = grouped[col].transform(lambda x: x.rolling(window, min_periods=window).max())
        is_max = (df_small[col] == rolling_max).astype(int)
        return (
            f'{col}_rolling_max_{window}', rolling_max,
            f'{col}_is_rolling_max_{window}', is_max
        )

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df
        df = df.sort_values(by=['product_id', 'customer_id', 'fecha'])

        tasks = []
        for col in self.columns:
            for window in self.window:
                df_small = df[['product_id', 'customer_id', 'fecha', col]].copy()
                tasks.append((col, window, df_small))

        if self.n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()

        with multiprocessing.Pool(processes=self.n_jobs) as pool:
            results = pool.map(self._compute_rolling_features, tasks)

        for max_name, max_series, ismax_name, ismax_series in results:
            df[max_name] = max_series
            df[ismax_name] = ismax_series

        pipeline.df = df
    

class RollingMinFeatureStep(PipelineStep):
    def __init__(self, window: List[int], columns: List[str], name: Optional[str] = None, n_jobs=-1):
        super().__init__(name)
        self.window = window
        self.columns = columns
        self.n_jobs = n_jobs

    def _compute_rolling_features(self, args):
        col, window, df_small = args
        grouped = df_small.groupby(['product_id', 'customer_id'])
        rolling_min = grouped[col].transform(lambda x: x.rolling(window, min_periods=window).min())
        is_min = (df_small[col] == rolling_min).astype(int)
        return (
            f'{col}_rolling_min_{window}', rolling_min,
            f'{col}_is_rolling_min_{window}', is_min
        )

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df
        df = df.sort_values(by=['product_id', 'customer_id', 'fecha'])

        tasks = []
        for col in self.columns:
            for window in self.window:
                df_small = df[['product_id', 'customer_id', 'fecha', col]].copy()
                tasks.append((col, window, df_small))

        if self.n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()

        with multiprocessing.Pool(processes=self.n_jobs) as pool:
            results = pool.map(self._compute_rolling_features, tasks)

        for min_name, min_series, ismin_name, ismin_series in results:
            df[min_name] = min_series
            df[ismin_name] = ismin_series

        pipeline.df = df
                
class RollingStdFeatureStep(PipelineStep):
    def __init__(self, window: List[int], columns: List[str], name: Optional[str] = None, n_jobs=-1):
        super().__init__(name)
        self.window = window
        self.columns = columns
        self.n_jobs = n_jobs

    def _compute_rolling_std(self, args):
        col, window, df_small = args
        grouped = df_small.groupby(['product_id', 'customer_id'])
        return (
            f'{col}_rolling_std_{window}',
            grouped[col].transform(lambda x: x.rolling(window, min_periods=window).std())
        )

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df
        df = df.sort_values(by=['product_id', 'customer_id', 'fecha'])

        tasks = []
        for col in self.columns:
            for window in self.window:
                df_small = df[['product_id', 'customer_id', 'fecha', col]].copy()
                tasks.append((col, window, df_small))

        if self.n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=self.n_jobs) as pool:
            results = pool.map(self._compute_rolling_std, tasks)

        for col_name, series in results:
            df[col_name] = series

        pipeline.df = df

class RollingIsMaxFeatureStep(PipelineStep):
    def __init__(self, window: List[int], columns: List, name: Optional[str] = None):
        super().__init__(name)
        self.window = window
        self.columns = columns

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df
        for col in self.columns:
            for win in self.window:
                max_rolling = (
                    df.groupby(['product_id', 'customer_id'])[col]
                    .transform(lambda x: x.rolling(win, min_periods=1).max())
                )
                df[f"{col}_is_rolling_max_{win}"] = (df[col] == max_rolling).astype(int)
        pipeline.df = df
        
class RollingIsMinFeatureStep(PipelineStep):
    def __init__(self, window: List[int], columns: List, name: Optional[str] = None):
        super().__init__(name)
        self.window = window
        self.columns = columns

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df
        for col in self.columns:
            for win in self.window:
                min_rolling = (
                    df.groupby(['product_id', 'customer_id'])[col]
                    .transform(lambda x: x.rolling(win, min_periods=1).min())
                )
                df[f"{col}_is_rolling_min_{win}"] = (df[col] == min_rolling).astype(int)
        pipeline.df = df
    
class CreateTotalCategoryStep(PipelineStep):
    def __init__(self, name: Optional[str] = None, cat: str = "cat1", tn: str = "tn"):
        super().__init__(name)
        self.cat = cat
        self.tn = tn
    
    def execute(self, pipeline: "Pipeline") -> None:
        df = pipeline.df
        df = df.sort_values(['fecha', self.cat])
        df[f"{self.tn}_{self.cat}_vendidas"] = (
            df.groupby(['fecha', self.cat])[self.tn]
              .transform('sum')
        )
        pipeline.df = df

class FeatureEngineeringProductInteractionStep(PipelineStep):

    def execute(self, pipeline: Pipeline) -> None:
        """
        El dataframe tiene una columna product_id y customer_id y fecha.
        Quiero obtener los x productos con mas tn del ultimo mes y crear x nuevas columnas que es la suma de tn de esos productos.
        se deben agregan entonces respetando la temporalidad la columna product_{product_id}_total_tn
        """
        df = pipeline.df
        last_date = df["fecha"].max()
        last_month_df = df[df["fecha"] == last_date]
        top_products = last_month_df.groupby("product_id").aggregate({"tn": "sum"}).nlargest(10, "tn").index.tolist()
        #mejor agruparlo por categoria y hacer una columna por cada categoria tanto de agrup por product como por customer
        for product_id in tqdm.tqdm(top_products):
            # creo un subset que es el total de product_id vendidos para todos los customer en cada t y lo mergeo a df
            product_df = df[df["product_id"] == product_id].groupby("fecha").aggregate({"tn": "sum"}).reset_index()
            product_df = product_df.rename(columns={"tn": f"product_{product_id}_total_tn"})
            product_df = product_df[["fecha", f"product_{product_id}_total_tn"]]
            df = df.merge(product_df, on="fecha", how="left")
        pipeline.df = df
        
class FeatureEngineeringTop20ProductsStep(PipelineStep):
    def __init__(self, tn_col: str = "tn", top_n: int = 20, name: Optional[str] = None):
        super().__init__(name)
        self.tn_col = tn_col
        self.top_n = top_n

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df

        # 1) Obtener última fecha
        last_date = df["fecha"].max()

        # 2) Seleccionar top N productos por tn en la última fecha
        last_month_df = df[df["fecha"] == last_date]
        top_products = (
            last_month_df.groupby("product_id")[self.tn_col]
            .sum()
            .nlargest(self.top_n)
            .index
            .tolist()
        )

        # 3) Para cada producto top, crear columna con suma tn por fecha
        for product_id in top_products:
            product_agg = (
                df[df["product_id"] == product_id]
                .groupby("fecha")[self.tn_col]
                .sum()
                .reset_index()
                .rename(columns={self.tn_col: f"product_{product_id}_sum_tn"})
            )
            df = df.merge(product_agg, on="fecha", how="left")

        pipeline.df = df
        
class ProductInteractionLagsStep(PipelineStep):
    """
    Calcula lags para las columnas product_{product_id}_total_tn.
    """
    def __init__(self, lags=[1,2,3], name: Optional[str] = None):
        super().__init__(name)
        self.lags = lags

    def execute(self, pipeline: "Pipeline") -> None:
        df = pipeline.df
        product_total_cols = [col for col in df.columns if col.startswith("product_") and col.endswith("_total_tn")]
        df = df.sort_values(by=['fecha'])
        for col in product_total_cols:
            for lag in self.lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        pipeline.df = df

class ProductInteractionRollingMeanStep(PipelineStep):
    """
    Calcula medias móviles para las columnas product_{product_id}_total_tn.
    """
    def __init__(self, windows=[3,6,12], name: Optional[str] = None):
        super().__init__(name)
        self.windows = windows

    def execute(self, pipeline: "Pipeline") -> None:
        df = pipeline.df
        product_total_cols = [col for col in df.columns if col.startswith("product_") and col.endswith("_total_tn")]
        df = df.sort_values(by=['fecha'])
        for col in product_total_cols:
            for window in self.windows:
                df[f"{col}_rolling_mean_{window}"] = df[col].rolling(window, min_periods=1).mean()
        pipeline.df = df

class ProductInteractionRollingMinMaxStep(PipelineStep):
    """
    Calcula mínimos y máximos móviles para las columnas product_{product_id}_total_tn.
    """
    def __init__(self, windows=[3,6,12], name: Optional[str] = None):
        super().__init__(name)
        self.windows = windows

    def execute(self, pipeline: "Pipeline") -> None:
        df = pipeline.df
        product_total_cols = [col for col in df.columns if col.startswith("product_") and col.endswith("_total_tn")]
        df = df.sort_values(by=['fecha'])
        for col in product_total_cols:
            for window in self.windows:
                df[f"{col}_rolling_max_{window}"] = df[col].rolling(window, min_periods=1).max()
                df[f"{col}_rolling_min_{window}"] = df[col].rolling(window, min_periods=1).min()
        pipeline.df = df


class FeatureEngineeringProductCatInteractionStep(PipelineStep):

    def __init__(self, cat="cat1", name: Optional[str] = None):
        super().__init__(name)
        self.cat = cat


    def execute(self, pipeline: Pipeline) -> None:
        # agrupo el dataframe por cat1 (sumando), obteniendo fecha, cat1 y
        # luego paso el dataframe a wide format, donde cada columna es una categoria  y la fila es la suma de tn para cada cat1
        # luego mergeo al dataframe original por fecha y product_id
        df = pipeline.df
        df_cat = df.groupby(["fecha", self.cat]).agg({"tn": "sum"}).reset_index()
        df_cat = df_cat.pivot(index="fecha", columns=self.cat, values="tn").reset_index()
        df = df.merge(df_cat, on="fecha", how="left")
        pipeline.df = df
        
class OutlierPasoFeatureStep(PipelineStep):
    def __init__(self, fecha_outlier: str = "2019-08-01", name: Optional[str] = None):
        super().__init__(name)
        self.fecha_outlier = pd.to_datetime(fecha_outlier)

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df
        df["outlier_paso"] = (df["fecha"] == self.fecha_outlier).astype(np.uint8)
        pipeline.df = df
        
class PeriodsSinceLastPurchaseStep(PipelineStep):
    def __init__(self, tn_col: str = "tn", name: Optional[str] = None):
        super().__init__(name)
        self.tn_col = tn_col

    def execute(self, pipeline: "Pipeline") -> None:
        df = pipeline.df
        df = df.sort_values(['customer_id', 'product_id', 'fecha'])
        
        def periods_since_last_purchase(series):
            last_purchase_idx = -1
            result = []
            for i, val in enumerate(series):
                if val > 0:
                    last_purchase_idx = i
                    result.append(0)
                elif last_purchase_idx == -1:
                    result.append(None)  # Nunca compró antes
                else:
                    result.append(i - last_purchase_idx)
            return result

        df['periodos_desde_ultima_compra'] = (
            df.groupby(['customer_id', 'product_id'])[self.tn_col]
              .transform(periods_since_last_purchase)
        )
        pipeline.df = df
        
class CantidadUltimaCompraStep(PipelineStep):
    """
    Crea una feature 'cantidad_ultima_compra' que, para cada fila, contiene el valor de tn de la última compra (tn>0)
    previa para cada combinación de product_id y customer_id. Si nunca hubo compra previa, devuelve np.nan.
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df
        df = df.sort_values(['product_id', 'customer_id', 'fecha'])
        def get_last_purchase(series):
            last = np.nan
            result = []
            for val in series:
                result.append(last)
                if val > 0:
                    last = val
            return result
        df['cantidad_ultima_compra'] = (
            df.groupby(['product_id', 'customer_id'])['tn']
            .transform(get_last_purchase)
        )
        pipeline.df = df
        
class DiferenciaTNUltimaCompraStep(PipelineStep):
    """
    Crea una feature 'diferencia_tn_ultima_compra' que es la diferencia entre tn y cantidad_ultima_compra para cada registro.
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df
        if "cantidad_ultima_compra" not in df.columns:
            raise ValueError("La columna 'cantidad_ultima_compra' no existe. Ejecuta CantidadUltimaCompraStep antes.")
        df["diferencia_tn_ultima_compra"] = df["tn"] - df["cantidad_ultima_compra"]
        pipeline.df = df
        
class PeriodsSinceLastPurchaseCustomerLevelStep(PipelineStep):
    """
    Crea una feature 'periodos_desde_ultima_compra_customer' que indica, para cada fila,
    la cantidad de períodos desde la última compra de ese customer (sin importar el producto).
    """
    def __init__(self, tn_col: str = "tn", name: Optional[str] = None):
        super().__init__(name)
        self.tn_col = tn_col

    def execute(self, pipeline: "Pipeline") -> None:
        df = pipeline.df
        df = df.sort_values(['customer_id', 'fecha'])

        def periods_since_last_purchase(series):
            last_purchase_idx = -1
            result = []
            for i, val in enumerate(series):
                if val > 0:
                    last_purchase_idx = i
                    result.append(0)
                elif last_purchase_idx == -1:
                    result.append(None)  # Nunca compró antes
                else:
                    result.append(i - last_purchase_idx)
            return result

        df['periodos_desde_ultima_compra_customer'] = (
            df.groupby(['customer_id'])[self.tn_col]
              .transform(periods_since_last_purchase)
        )
        pipeline.df = df
        
class ProductIdBuyersCountByDateStep(PipelineStep):
    """
    Agrega una columna 'product_buyers_count_fecha' al DataFrame, que indica para cada fila
    cuántos customers compraron ese product_id en esa fecha (tn > 0).
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df
        # Contar compradores únicos por producto y fecha (tn > 0)
        buyers_count = (
            df[df["tn"] > 0]
            .groupby(["product_id", "fecha"])["customer_id"]
            .nunique()
            .reset_index()
            .rename(columns={"customer_id": "product_id_unique_customers"})
        )
        # Mergear al dataframe original
        df = df.merge(buyers_count, on=["product_id", "fecha"], how="left")
        # Si algún producto no tuvo compradores esa fecha, poner 0
        df["product_id_unique_customers"] = df["product_id_unique_customers"].fillna(0).astype(int)
        pipeline.df = df

class CustomerIdUniqueProductsByDateStep(PipelineStep):
    """
    Agrega una columna 'customer_unique_products_fecha' al DataFrame, que indica para cada fila
    cuántos productos diferentes compró ese customer en esa fecha (tn > 0).
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df
        # Contar productos únicos por customer y fecha (tn > 0)
        products_count = (
            df[df["tn"] > 0]
            .groupby(["customer_id", "fecha"])["product_id"]
            .nunique()
            .reset_index()
            .rename(columns={"product_id": "customer_id_unique_products_purchased"})
        )
        # Mergear al dataframe original
        df = df.merge(products_count, on=["customer_id", "fecha"], how="left")
        # Si algún customer no compró productos esa fecha, poner 0
        df["customer_id_unique_products_purchased"] = df["customer_id_unique_products_purchased"].fillna(0).astype(int)
        pipeline.df = df      
        
class ProductSizeCategoryStep(PipelineStep):
    def __init__(self, sku_size_col: str = "sku_size", name: Optional[str] = None):
        super().__init__(name)
        self.sku_size_col = sku_size_col

    def execute(self, pipeline: "Pipeline") -> None:
        df = pipeline.df
        def categorize(size):
            if size < 200:
                return "small"
            elif size < 800:
                return "medium"
            else:
                return "large"
        df["product_size"] = df[self.sku_size_col].apply(categorize)
        pipeline.df = df
        
class DateRelatedFeaturesStep(PipelineStep):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df
        df["year"] = df["fecha"].dt.year
        df["quarter"] = df["fecha"].dt.quarter
        df["mes"] = df["fecha"].dt.month
        df["dias_en_mes"] = df["fecha"].dt.days_in_month
        # Features cíclicas senos y cosenos para mes y quarter:
        df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
        df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)
        df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
        df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)    
        # Feature de periodo secuencial
        fechas_ordenadas = np.sort(df["fecha"].unique())
        fecha_a_periodo = {fecha: i+1 for i, fecha in enumerate(fechas_ordenadas)}
        df["periodo"] = df["fecha"].map(fecha_a_periodo)
        # Feriados Argentina
        years = df["year"].unique()
        ar_holidays = holidays.country_holidays('AR', years=years)
        # Crear DataFrame de feriados
        feriados_df = pd.DataFrame({'fecha': list(ar_holidays.keys())})
        feriados_df['year'] = pd.to_datetime(feriados_df['fecha']).dt.year
        feriados_df['mes'] = pd.to_datetime(feriados_df['fecha']).dt.month
        # Contar feriados por año y mes
        feriados_mes = feriados_df.groupby(['year', 'mes']).size().reset_index(name='feriados_en_mes')
        # Merge con el df original
        df = df.merge(feriados_mes, on=['year', 'mes'], how='left')
        df['feriados_en_mes'] = df['feriados_en_mes'].fillna(0).astype(int)     
        pipeline.df = df

class EdadProductoStep(PipelineStep):
    def __init__(self, tn_col: str = "tn", name: Optional[str] = None):
        super().__init__(name)
        self.tn_col = tn_col

    def execute(self, pipeline: "Pipeline") -> None:
        df = pipeline.df
        
        # Ordenar por producto y fecha
        df = df.sort_values(["product_id", "fecha"])

        # Obtener el índice (posición) de la primera fila donde tn > 0 para cada producto
        def compute_edad(serie):
            # índice relativo dentro del grupo
            edad = list(range(len(serie)))
            # encontrar primer índice donde hay compra
            try:
                first_buy_idx = next(i for i, v in enumerate(serie) if v > 0)
                return [i - first_buy_idx if i >= first_buy_idx else None for i in edad]
            except StopIteration:
                # nunca se compró, todos son None
                return [None] * len(serie)

        df["edad_producto"] = (
            df.groupby("product_id")[self.tn_col]
              .transform(compute_edad)
        )

        pipeline.df = df
        
class EdadClienteStep(PipelineStep):
    def __init__(self, tn_col: str = "tn", name: Optional[str] = None):
        super().__init__(name)
        self.tn_col = tn_col

    def execute(self, pipeline: "Pipeline") -> None:
        df = pipeline.df

        # Ordenar por cliente y fecha para mantener la secuencia temporal
        df = df.sort_values(["customer_id", "fecha"])

        # Función para calcular edad del cliente en períodos desde su primera compra
        def compute_edad(serie):
            edad = list(range(len(serie)))
            try:
                first_buy_idx = next(i for i, v in enumerate(serie) if v > 0)
                return [i - first_buy_idx if i >= first_buy_idx else None for i in edad]
            except StopIteration:
                return [None] * len(serie)  # nunca compró

        # Aplicar por cliente
        df["edad_cliente"] = (
            df.groupby("customer_id")[self.tn_col]
              .transform(compute_edad)
        )

        pipeline.df = df

class EdadCustomerProductoStep(PipelineStep):
    def __init__(self, tn_col: str = "tn", name: Optional[str] = None):
        super().__init__(name)
        self.tn_col = tn_col

    def execute(self, pipeline: "Pipeline") -> None:
        df = pipeline.df

        # Ordenar para mantener la secuencia temporal dentro de cada combinación
        df = df.sort_values(["customer_id", "product_id", "fecha"])

        # Función para calcular edad desde la primera compra
        def compute_edad(serie):
            edad = list(range(len(serie)))
            try:
                first_buy_idx = next(i for i, v in enumerate(serie) if v > 0)
                return [i - first_buy_idx if i >= first_buy_idx else None for i in edad]
            except StopIteration:
                return [None] * len(serie)  # nunca compró

        # Agrupar por combinación customer-product
        df["edad_customer_producto"] = (
            df.groupby(["customer_id", "product_id"])[self.tn_col]
              .transform(compute_edad)
        )

        pipeline.df = df
        
class SplitDataFrameStep(PipelineStep):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df
        sorted_dated = sorted(df["fecha"].unique())
        last_date = sorted_dated[-1] # es 12-2019
        last_test_date = sorted_dated[-3] # needs a gap because forecast moth+2
        last_train_date = sorted_dated[-4] #
        
        kaggle_pred = df[df["fecha"] == last_date]
        test = df[df["fecha"] == last_test_date]
        eval_data = df[df["fecha"] == last_train_date]
        train = df[(df["fecha"] < last_train_date)]
        df_final = df[df["fecha"] <= last_test_date] # Incluye Octubre para Kaggle
        df_intermedio = df[df["fecha"] <= last_train_date] # Incluye Septiembre para testear Octubre
        pipeline.train = train
        pipeline.eval_data = eval_data
        pipeline.test = test
        pipeline.kaggle_pred = kaggle_pred
        pipeline.df_intermedio = df_intermedio
        pipeline.df_final = df_final
        
        if hasattr(pipeline, "sample_weights"):
            full_weights = pipeline.sample_weights

            # train + eval para Optuna
            idx_train_eval = train.index.union(eval_data.index)
            weights_train = full_weights.reindex(idx_train_eval).fillna(1.0)
            pipeline.sample_weights_train = weights_train

            # df_final para entrenamiento final
            weights_final = full_weights.reindex(df_final.index).fillna(1.0)
            pipeline.sample_weights_train_final = weights_final

            # Logging de alineación
            pipeline.logger.info(f"[SplitDataFrameStep] sample_weights_train: {len(weights_train)} obs, expected {len(idx_train_eval)}")
            pipeline.logger.info(f"[SplitDataFrameStep] sample_weights_train_final: {len(weights_final)} obs, expected {len(df_final)}")

            # Check adicional de desalineación
            if weights_train.isnull().any():
                pipeline.logger.warning("[SplitDataFrameStep] sample_weights_train tiene valores nulos tras reindex.")
            if weights_final.isnull().any():
                pipeline.logger.warning("[SplitDataFrameStep] sample_weights_train_final tiene valores nulos tras reindex.")

class CustomMetric:
    def __init__(self, df_eval, product_id_col='product_id', scaler=None):
        self.scaler = scaler
        self.df_eval = df_eval
        self.product_id_col = product_id_col
    
    def __call__(self, preds, train_data):
        labels = train_data.get_label()
        df_temp = self.df_eval.copy()
        df_temp['preds'] = preds
        df_temp['labels'] = labels

        if self.scaler:
            df_temp['preds'] = self.scaler.inverse_transform(df_temp[['preds']])
            df_temp['labels'] = self.scaler.inverse_transform(df_temp[['labels']])
        
        # Agrupar por product_id y calcular el error
        por_producto = df_temp.groupby(self.product_id_col).agg({'labels': 'sum', 'preds': 'sum'})
        
        # Calcular el error personalizado
        error = np.sum(np.abs(por_producto['labels'] - por_producto['preds'])) / np.sum(por_producto['labels'])
        
        # LightGBM espera que el segundo valor sea mayor cuando el modelo es mejor
        return 'custom_error', error, False
    
class PrepareXYStep(PipelineStep):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, pipeline: Pipeline) -> None:
        train = pipeline.train
        eval_data = pipeline.eval_data
        test = pipeline.test
        kaggle_pred = pipeline.kaggle_pred
        df_intermedio = pipeline.df_intermedio
        df_final = pipeline.df_final

        features = [col for col in train.columns if col not in
                        ['fecha', 'target']]
        target = 'target'

        X_train = pd.concat([train[features], eval_data[features]]) # [train + eval] + [eval] -> [test] 
        y_train = pd.concat([train[target], eval_data[target]])

        X_train_alone = train[features]
        y_train_alone = train[target]

        X_eval = eval_data[features]
        y_eval = eval_data[target]

        X_test = test[features]
        y_test = test[['product_id', 'target']]

        X_train_final = df_final[features]
        y_train_final = df_final[target]
        X_kaggle = kaggle_pred[features]
        
        X_train_intermedio = df_intermedio[features]
        y_train_intermedio = df_intermedio[target]
                
        pipeline.X_train = X_train
        pipeline.y_train = y_train
        pipeline.X_train_alone = X_train_alone
        pipeline.y_train_alone = y_train_alone
        pipeline.X_eval = X_eval
        pipeline.y_eval = y_eval
        pipeline.X_test = X_test
        pipeline.y_test = y_test
        pipeline.X_train_intermedio = X_train_intermedio
        pipeline.y_train_intermedio = y_train_intermedio
        pipeline.X_train_final = X_train_final
        pipeline.y_train_final = y_train_final
        pipeline.X_kaggle = X_kaggle
        

class TrainModelLGBStep(PipelineStep):
    def __init__(self, params: Dict = {}, train_eval_sets = {}, num_boost_round: int = 1000, name: Optional[str] = None):
        super().__init__(name)
        if not params:
            params = {
                "objective": "regression",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.1,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1
            }
        if not train_eval_sets:
            train_eval_sets = {
                "X_train": "X_train",
                "y_train": "y_train",
                "X_eval": "X_eval",
                "y_eval": "y_eval",
                "eval_data": "eval_data",
            }
        self.params = params
        self.train_eval_sets = train_eval_sets
        self.num_boost_round = num_boost_round

    def execute(self, pipeline: Pipeline) -> None:
        X_train = pipeline.X_train
        y_train = pipeline.y_train
        X_eval = pipeline.X_eval
        y_eval = pipeline.y_eval
        df_eval = pipeline.eval_data

        cat_features = [col for col in X_train.columns if X_train[col].dtype.name == 'category']

        scaler = pipeline.get_artifact("scaler")
        scaler_target = pipeline.get_artifact("scaler_target")
        if scaler:
            X_train[scaler.feature_names_in_] = scaler.transform(X_train[scaler.feature_names_in_])
            X_eval[scaler.feature_names_in_] = scaler.transform(X_eval[scaler.feature_names_in_])
            y_train = pd.Series(
                scaler_target.transform(y_train.values.reshape(-1, 1)).flatten(),
                index=y_train.index,
            )
            y_eval = pd.Series(
                scaler_target.transform(y_eval.values.reshape(-1, 1)).flatten(),
                index=y_eval.index,
            )

        
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
        eval_data = lgb.Dataset(X_eval, label=y_eval, reference=train_data, categorical_feature=cat_features)
        custom_metric = CustomMetric(df_eval, product_id_col='product_id', scaler=scaler_target)
        callbacks = [
            lgb.early_stopping(250),
            lgb.log_evaluation(100),
        ]
        model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=[eval_data],
            feval=custom_metric,
            callbacks=callbacks
        )
        # Save the model
        pipeline.model = model
        
class OptunaLGBMOptimizationStep(PipelineStep):
    def __init__(self, n_trials=50,study_name="optuna_lgbm", db_path="optuna_study.db", name: Optional[str] = None):
        super().__init__(name)
        self.n_trials = n_trials
        self.study_name = study_name
        self.db_path = db_path  # Ruta local al .db        

    def execute(self, pipeline: Pipeline) -> None:
        
        # Ruta incremental al bucket
        self.results_path = os.path.join(
            "/home/tomifernandezlabo3/gcs-bucket",
            "experiments",
            pipeline.experiment_name,
            "optuna_trials_incremental.csv"
        )
                
        try:
            scaler = pipeline.get_artifact("scaler")
        except ValueError:
            scaler = None
            pipeline.logger.warning("Scaler not found. Proceeding without scaling.")

        try:
            scaler_target = pipeline.get_artifact("scaler_target")
        except ValueError:
            scaler_target = None
            pipeline.logger.warning("Scaler target not found. Proceeding without target scaling.")
                
        X_train = pipeline.X_train
        y_train = pipeline.y_train
        X_eval = pipeline.X_eval
        y_eval = pipeline.y_eval
        df_eval = pipeline.eval_data
     
        if scaler_target is not None:
            custom_metric = CustomMetricDelta(df_eval, product_id_col='product_id', scaler=scaler_target)
        else:
            pipeline.logger.warning("CustomMetric created without scaler_target.")
            custom_metric = CustomMetricDelta(df_eval, product_id_col='product_id')        
            
        cat_features = [col for col in X_train.columns if X_train[col].dtype.name == 'category']

        def objective(trial):
            try:
                train_data = lgb.Dataset(X_train, label=y_train, weight=pipeline.sample_weights_train, categorical_feature=cat_features)
                eval_data = lgb.Dataset(X_eval, label=y_eval, reference=train_data, categorical_feature=cat_features)
                callbacks = [lgb.early_stopping(150)]
                param = {
                    'num_leaves': trial.suggest_int('num_leaves', 100, 2000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'random_state': 42,
                    'boosting_type': 'gbdt',
                    'objective': 'regression',
                    #'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.9),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.8, 0.9),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.8, 0.9),
                    'bagging_freq': trial.suggest_int('bagging_freq', 50, 200),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
                    'verbose': -1,
                    'max_bin': trial.suggest_int('max_bin', 255, 1000),
                    'lambda_l1': trial.suggest_float('lambda_l1', 1e-4, 10.0, log=True),
                    'lambda_l2': trial.suggest_float('lambda_l2', 1e-4, 10.0, log=True),
                    'n_jobs': -1,
                }
                num_boost_rounds = trial.suggest_int('num_boost_rounds', 1000, 3000)

                model = lgb.train(
                    param,
                    train_data,
                    num_boost_round=num_boost_rounds,
                    valid_sets=[eval_data],
                    feval=custom_metric,
                    callbacks=callbacks
                )            
                preds = model.predict(X_eval)
                _, score, _ = custom_metric(preds, lgb.Dataset(X_eval, label=y_eval))

                trial.set_user_attr("score", score)  # Save score in trial object
                trial.set_user_attr("num_boost_rounds", num_boost_rounds)
                
                # Guardar incrementalmente en CSV
                row = {
                    'trial': trial.number,
                    'score': score,
                    'num_boost_rounds': num_boost_rounds,
                    **trial.params
                }
                df_trial = pd.DataFrame([row])
                df_trial.to_csv(self.results_path, mode='a', index=False, header=not os.path.exists(self.results_path))
                
                return score
            
            except Exception as e:
                pipeline.logger.warning(f"Trial {trial.number} failed: {e}")
                raise TrialPruned()  # Ignora el trial pero continua la ejecucion.
        
        # Crear/recuperar el estudio con almacenamiento persistente
        storage_url = f"sqlite:///{self.db_path}"
        study = optuna.create_study(
            direction='minimize',
            study_name=self.study_name,
            storage=storage_url,
            load_if_exists=True
        )        
        
        pipeline.logger.info(f"Starting optimization with {self.n_trials} trials.")
        study.optimize(objective, n_trials=self.n_trials)
        pipeline.logger.info("Optimization completed.")
        best_trial = study.best_trial
        best_params = best_trial.params.copy()
        best_num_boost_rounds = best_trial.user_attrs.get("num_boost_rounds", 1000)
        pipeline.best_params = best_params
        pipeline.best_num_boost_rounds = best_num_boost_rounds

        # -----------------------------
        # Guardar tabla completa de trials
        # -----------------------------
        trial_rows = []
        for trial in study.trials:
            row = {
                'trial': trial.number,
                'score': trial.user_attrs.get("score", None),
                'num_boost_rounds': trial.user_attrs.get("num_boost_rounds", None),
                **trial.params}
            trial_rows.append(row)
            
        df_trials = pd.DataFrame(trial_rows).sort_values(by="score")
        pipeline.optuna_trials_df = df_trials
        
class PredictStep(PipelineStep):
    def __init__(self, predict_set: str, name: Optional[str] = None):
        super().__init__(name)
        self.predict_set = predict_set

    def execute(self, pipeline: Pipeline) -> None:
        # Obtener el DataFrame de predicción directamente de la memoria
        if self.predict_set == "X_test":
            X_predict = pipeline.X_test
        elif self.predict_set == "X_kaggle":
            X_predict = pipeline.X_kaggle
        elif self.predict_set == "X_train_final":
            X_predict = pipeline.X_train_final
        else:
            # Fallback a get_artifact si no es uno de los DataFrames en memoria
            X_predict = pipeline.get_artifact(self.predict_set)

        # Intentar obtener el scaler
        try:
            scaler = pipeline.get_artifact("scaler")
        except ValueError:
            scaler = None
            pipeline.logger.warning("Scaler not found. Proceeding without feature scaling.")

        # Si hay scaler y tiene feature_names_in_, aplicar transformación
        if scaler and hasattr(scaler, "feature_names_in_"):
            cols_to_scale = scaler.feature_names_in_
            X_predict[cols_to_scale] = scaler.transform(X_predict[cols_to_scale])

        # Obtener el modelo directamente de la memoria o como artifact
        try:
            model = pipeline.model  # Modelo en memoria
        except AttributeError:
            # Fallback a get_artifact
            model = pipeline.get_artifact("model_testing")  # o "model" para modelo final

        predictions = model.predict(X_predict)

        # Intentar obtener el scaler_target
        try:
            scaler_target = pipeline.get_artifact("scaler_target")
        except ValueError:
            scaler_target = None
            pipeline.logger.warning("Scaler target not found. Proceeding without target inverse scaling.")

        # Inversión de escala del target si corresponde
        if scaler_target:
            predictions = scaler_target.inverse_transform(predictions.reshape(-1, 1)).flatten()

        # Formatear predicciones
        predictions = pd.DataFrame(predictions, columns=["predictions"], index=X_predict.index)
        predictions["product_id"] = X_predict["product_id"]

        # Guardar predicciones en memoria y como artifact
        pipeline.predictions = predictions

class EvaluatePredictionsSteps(PipelineStep):
    def __init__(self, y_actual_df: str, name: Optional[str] = None):
        super().__init__(name)
        self.y_actual_df = y_actual_df

    def execute(self, pipeline: Pipeline) -> None:
        # Obtener predicciones directamente de la memoria
        try:
            predictions = pipeline.predictions
        except AttributeError:
            # Fallback a get_artifact
            predictions = pipeline.get_artifact("predictions")

        # Obtener y_actual usando el mapeo de nombres comunes
        if self.y_actual_df == "y_test":
            y_actual = pipeline.y_test
        else:
            # Fallback a get_artifact para otros casos
            y_actual = pipeline.get_artifact(self.y_actual_df)

        product_actual = y_actual.groupby("product_id")["target"].sum()
        product_pred = predictions.groupby("product_id")["predictions"].sum()

        eval_df = pd.DataFrame({
            "product_id": product_actual.index,
            "tn_real": product_actual.values,
            "tn_pred": product_pred.values
        })

        total_error = np.sum(np.abs(eval_df['tn_real'] - eval_df['tn_pred'])) / np.sum(eval_df['tn_real'])
        print(f"Error en test: {total_error:.4f}")
        print("\nTop 10 productos con mayor error absoluto:")
        eval_df['error_absoluto'] = np.abs(eval_df['tn_real'] - eval_df['tn_pred'])
        print(eval_df.sort_values('error_absoluto', ascending=False).head(10))
        
        # Guardar en memoria y como artifacts
        pipeline.eval_df = eval_df
        pipeline.total_error = total_error

class TrainFinalModelLGBTestingStep(PipelineStep):
    """
    Entrena el modelo final LightGBM usando el dataset hasta Septiembre para testear sobre Octubre
    con los mejores hiperparámetros encontrados por Optuna.
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, pipeline: Pipeline) -> None:
        # Cargar datos directamente de la memoria
        X_train_final = pipeline.X_train_intermedio
        y_train_final = pipeline.y_train_intermedio

        # Intentar obtener scaler
        try:
            scaler = pipeline.get_artifact("scaler")
        except ValueError:
            scaler = None
            pipeline.logger.warning("Scaler not found. Proceeding without scaling.")

        try:
            scaler_target = pipeline.get_artifact("scaler_target")
        except ValueError:
            scaler_target = None
            pipeline.logger.warning("Scaler target not found. Proceeding without target scaling.")

        # Escalar si corresponde
        if scaler:
            X_train_final[scaler.feature_names_in_] = scaler.transform(X_train_final[scaler.feature_names_in_])
        if scaler_target:
            y_train_final = pd.Series(
                scaler_target.transform(y_train_final.values.reshape(-1, 1)).flatten(),
                index=y_train_final.index,
            )

        # Cargar hiperparámetros óptimos directamente de la memoria
        best_params = pipeline.best_params
        best_num_boost_rounds = pipeline.best_num_boost_rounds

        # Categorías
        cat_features = [col for col in X_train_final.columns if X_train_final[col].dtype.name == 'category']

        # Dataset final
        train_data = lgb.Dataset(X_train_final, label=y_train_final, categorical_feature=cat_features)

        # Entrenar modelo final
        model = lgb.train(
            best_params,
            train_data,
            num_boost_round=best_num_boost_rounds
        )
        # Guardar modelo en memoria y como artifact
        pipeline.model_testing = model

class TrainFinalModelLGBKaggleStep(PipelineStep):
    """
    Entrena el modelo final LightGBM usando todo el dataset (X_train_final, y_train_final)
    con los mejores hiperparámetros encontrados por Optuna.
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, pipeline: Pipeline) -> None:
        # Cargar datos directamente de la memoria
        X_train_final = pipeline.X_train_final
        y_train_final = pipeline.y_train_final

        # Intentar obtener scaler
        try:
            scaler = pipeline.get_artifact("scaler")
        except ValueError:
            scaler = None
            pipeline.logger.warning("Scaler not found. Proceeding without scaling.")

        try:
            scaler_target = pipeline.get_artifact("scaler_target")
        except ValueError:
            scaler_target = None
            pipeline.logger.warning("Scaler target not found. Proceeding without target scaling.")

        # Escalar si corresponde
        if scaler:
            X_train_final[scaler.feature_names_in_] = scaler.transform(X_train_final[scaler.feature_names_in_])
        if scaler_target:
            y_train_final = pd.Series(
                scaler_target.transform(y_train_final.values.reshape(-1, 1)).flatten(),
                index=y_train_final.index,
            )

        # Cargar hiperparámetros óptimos directamente de la memoria
        best_params = pipeline.best_params
        best_num_boost_rounds = pipeline.best_num_boost_rounds

        # Categorías
        cat_features = [col for col in X_train_final.columns if X_train_final[col].dtype.name == 'category']

        # Dataset final
        train_data = lgb.Dataset(X_train_final, label=y_train_final, weight=pipeline.sample_weights_train_final, categorical_feature=cat_features)

        # Entrenamiento del modelo final
        model = lgb.train(
            best_params,
            train_data,
            num_boost_round=best_num_boost_rounds
        )

        # Guardar modelo en memoria
        pipeline.model = model        
        
class SaveFeatureImportanceStep(PipelineStep):
    def execute(self, pipeline: Pipeline) -> None:
        # Obtener el modelo directamente de la memoria
        try:
            model = pipeline.model
        except AttributeError:
            # Fallback a get_artifact
            model = pipeline.get_artifact("model")
        
        # Obtener importancia y nombres de features
        importance = model.feature_importance(importance_type='split')
        features = model.feature_name()
        df_importance = pd.DataFrame({
            "feature": features,
            "importance": importance
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        
        # Guardar en memoria y como artefacto
        pipeline.feature_importance_df = df_importance      
        
class FilterProductsIDStep(PipelineStep):
    def __init__(self, product_file = "/home/tomifernandezlabo3/labo3_tomi/product_id_apredecir201912.txt", dfs=["df"], name: Optional[str] = None):
        super().__init__(name)
        self.file = product_file
        self.dfs = dfs

    def execute(self, pipeline: Pipeline) -> None:
        """ el txt es un csv que tiene columna product_id separado por tabulaciones """
        converted_dfs = {}
        product_ids = pd.read_csv(self.file, sep="\t")["product_id"].tolist()
        
        for df_key in self.dfs:
            # Obtener DataFrame directamente de la memoria
            if df_key == "X_kaggle":
                df = pipeline.X_kaggle
            elif df_key == "kaggle_pred":
                df = pipeline.kaggle_pred
            elif df_key == "df":
                df = pipeline.df
            else:
                # Fallback a get_artifact para otros casos
                df = pipeline.get_artifact(df_key)
            
            # Filtrar por product_ids
            df = df[df["product_id"].isin(product_ids)]
            converted_dfs[df_key] = df
            print(f"Filtered DataFrame {df_key} shape: {df.shape}")
            
            # Guardar en memoria y como artifact
            if df_key == "X_kaggle":
                pipeline.X_kaggle = df
            elif df_key == "kaggle_pred":
                pipeline.kaggle_pred = df
            elif df_key == "df":
                pipeline.df = df
                            
        return converted_dfs      

class KaggleSubmissionStep(PipelineStep):
    def execute(self, pipeline: Pipeline) -> None:
        # Obtener modelo directamente de la memoria
        try:
            model = pipeline.model
        except AttributeError:
            # Fallback a get_artifact
            model = pipeline.get_artifact("model")

        # Obtener DataFrames directamente de la memoria
        kaggle_pred = pipeline.kaggle_pred
        X_kaggle = pipeline.X_kaggle

        # Intentar obtener scaler
        try:
            scaler = pipeline.get_artifact("scaler")
        except ValueError:
            scaler = None
            pipeline.logger.warning("Scaler not found. Proceeding without scaling X_kaggle.")

        try:
            scaler_target = pipeline.get_artifact("scaler_target")
        except ValueError:
            scaler_target = None
            pipeline.logger.warning("Scaler target not found. Proceeding without inverse transform.")

        # Escalar si corresponde
        if scaler:
            X_kaggle[scaler.feature_names_in_] = scaler.transform(X_kaggle[scaler.feature_names_in_])
        else:
            pipeline.logger.info("Skipping input scaling for X_kaggle.")

        # Predicción
        preds = model.predict(X_kaggle)

        # Desescalar si corresponde
        if scaler_target:
            preds = scaler_target.inverse_transform(preds.reshape(-1, 1)).flatten()
        else:
            pipeline.logger.info("Skipping inverse transform for predictions.")

        # Guardar predicciones
        kaggle_pred["tn_predicha"] = preds.copy()
        submission = kaggle_pred.groupby("product_id")["tn_predicha"].sum().reset_index()
        submission.columns = ["product_id", "tn"]

        # Solo guardar en memoria, NO como artifact
        pipeline.submission = submission
        
class KaggleSubmissionStepSubset(PipelineStep):
    """
    Kaggle submission step robusto para subsets: asegura que kaggle_pred y X_kaggle tengan el mismo índice y longitud.
    """
    def execute(self, pipeline: Pipeline) -> None:
        model = pipeline.get_artifact("model")
        kaggle_pred = pipeline.get_artifact("kaggle_pred")
        X_kaggle = pipeline.get_artifact("X_kaggle")
        scaler = pipeline.get_artifact("scaler")
        if scaler:
            X_kaggle[scaler.feature_names_in_] = scaler.transform(X_kaggle[scaler.feature_names_in_])
        preds = model.predict(X_kaggle)
        scaler_target = pipeline.get_artifact("scaler_target")
        if scaler_target:
            preds = scaler_target.inverse_transform(preds.reshape(-1, 1)).flatten()
        # Asegura que kaggle_pred y X_kaggle tengan el mismo índice y longitud
        kaggle_pred = kaggle_pred.loc[X_kaggle.index].copy()
        kaggle_pred["tn_predicha"] = preds
        submission = kaggle_pred.groupby("product_id")["tn_predicha"].sum().reset_index()
        submission.columns = ["product_id", "tn"]
        self.save_artifact(pipeline, "submission", submission)


class SaveExperimentStep(PipelineStep):
    def __init__(self, exp_name: str, save_dataframes=False, name: Optional[str] = None):
        super().__init__(name)
        self.exp_name = exp_name
        self.save_dataframes = save_dataframes

    def execute(self, pipeline: Pipeline) -> None:

        # Create the experiment directory
        exp_dir = f"experiments/{self.exp_name}"
        os.makedirs(exp_dir, exist_ok=True)

        # obtengo el model
        model = pipeline.get_artifact("model")
        # Save the model as a pickle file
        with open(os.path.join(exp_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
        # guardo el error total de test
        #total_error = pipeline.get_artifact("total_error")
        #with open(os.path.join(exp_dir, "total_error.txt"), "w") as f:
            #f.write(str(total_error))

        # Save the submission file
        submission = pipeline.get_artifact("submission")
        if submission is not None:
            submission.to_csv(os.path.join(exp_dir, f"submission_{self.exp_name}.csv"), index=False)
        
        # Guardar el feature importance si existe
        feature_importance_df = pipeline.get_artifact("feature_importance_df")
        if feature_importance_df is not None:
            feature_importance_df.to_csv(os.path.join(exp_dir, f"feature_importance_{self.exp_name}.csv"), index=False)

        # borro submission model y error de los artifacts
        for artifact_name in ["submission", "model", "feature_importance_df"]:
            if artifact_name in pipeline.artifacts:
                pipeline.del_artifact(artifact_name)
                #pipeline.del_artifact("total_error")
        
        # Guardo los artifacts restantes que son dataframes como csvs
        if self.save_dataframes:
            for artifact_name, artifact in pipeline.artifacts.items():
                if isinstance(artifact, pd.DataFrame):
                    artifact.to_csv(os.path.join(exp_dir, f"{artifact_name}.csv"), index=False)

class SaveSubsetExperimentStep(PipelineStep):
    """
    Guarda los resultados relevantes de cada subset en su carpeta correspondiente.
    """
    def __init__(self, exp_dir: str, subset_id: int, name: Optional[str] = None):
        super().__init__(name)
        self.exp_dir = exp_dir
        self.subset_id = subset_id

    def execute(self, pipeline: Pipeline) -> None:
        subset_dir = os.path.join(self.exp_dir, f"subset_{self.subset_id}")
        os.makedirs(subset_dir, exist_ok=True)

        # Guardar predicciones
        preds = pipeline.get_artifact("submission")
        if preds is not None:
            preds.to_csv(os.path.join(subset_dir, "submission.csv"), index=False)

        # Guardar feature importance
        fi = pipeline.get_artifact("feature_importance_df")
        if fi is not None:
            fi.to_csv(os.path.join(subset_dir, "feature_importance.csv"), index=False)

        # Guardar trials de Optuna
        trials = pipeline.get_artifact("optuna_trials_df")
        if trials is not None:
            trials.to_csv(os.path.join(subset_dir, "optuna_trials.csv"), index=False)

        # Guardar el modelo
        model = pipeline.get_artifact("model")
        if model is not None:
            with open(os.path.join(subset_dir, "model.pkl"), "wb") as f:
                pickle.dump(model, f)
                
class SaveResults(PipelineStep):
    """
    Guarda los resultados relevantes de cada experimento directamente en la carpeta local
    donde está montado el bucket de GCS (sin usar autenticación ni API de GCS).
    Permite parametrizar qué artefactos guardar basándose en atributos directos del pipeline.
    """
    BASE_BUCKET_PATH = "/home/tomifernandezlabo3/gcs-bucket"

    def __init__(self, exp_name: str, to_save=None, name: Optional[str] = None):
        """
        to_save: lista o set con strings que indiquen qué guardar. 
                 Opciones: "submission", "feature_importance", "optuna_trials", "total_error",
                           "model", "log", "best_params"
                 Si es None, guarda todo.
        """
        super().__init__(name)
        self.exp_name = exp_name
        self.to_save = set(to_save) if to_save is not None else {
            "submission", "feature_importance", "optuna_trials", "total_error",
            "model", "log", "best_params","df","scaler"
        }

    def _save_string_local(self, relative_path, content: str):
        full_path = os.path.join(self.BASE_BUCKET_PATH, relative_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)

    def _save_dataframe_local(self, relative_path, df):
        if df is not None:
            full_path = os.path.join(self.BASE_BUCKET_PATH, relative_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            df.to_csv(full_path, index=False)

    def _save_pickle_local(self, relative_path, obj):
        if obj is not None:
            full_path = os.path.join(self.BASE_BUCKET_PATH, relative_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "wb") as f:
                pickle.dump(obj, f)

    def execute(self, pipeline) -> None:
        total_error = getattr(pipeline, "total_error", None)
        if total_error is not None:
            exp_prefix = f"experiments/{self.exp_name}_error_test_{total_error:.4f}/"
        else:
            exp_prefix = f"experiments/{self.exp_name}/"

        if "submission" in self.to_save and hasattr(pipeline, "submission") and pipeline.submission is not None:
            self._save_dataframe_local(exp_prefix + "submission.csv", pipeline.submission)

        if "feature_importance" in self.to_save and hasattr(pipeline, "feature_importance_df") and pipeline.feature_importance_df is not None:
            self._save_dataframe_local(exp_prefix + "feature_importance.csv", pipeline.feature_importance_df)

        if "optuna_trials" in self.to_save and hasattr(pipeline, "optuna_trials_df") and pipeline.optuna_trials_df is not None:
            self._save_dataframe_local(exp_prefix + "optuna_trials.csv", pipeline.optuna_trials_df)

        if "model" in self.to_save and hasattr(pipeline, "model") and pipeline.model is not None:
            self._save_pickle_local(exp_prefix + "model.pkl", pipeline.model)

        if "total_error" in self.to_save and total_error is not None:
            self._save_string_local(exp_prefix + "total_error.txt", str(total_error))

        if "log" in self.to_save and hasattr(pipeline, "log_filename"):
            log_local_path = pipeline.log_filename
            if log_local_path and os.path.exists(log_local_path):
                log_dest_path = os.path.join(self.BASE_BUCKET_PATH, exp_prefix + "pipeline_log.txt")
                os.makedirs(os.path.dirname(log_dest_path), exist_ok=True)
                shutil.copy2(log_local_path, log_dest_path)

        if "best_params" in self.to_save and hasattr(pipeline, "best_params") and hasattr(pipeline, "best_num_boost_rounds"):
            best_config = {
                "best_params": pipeline.best_params,
                "best_num_boost_rounds": pipeline.best_num_boost_rounds
            }
            self._save_string_local(exp_prefix + "best_params.json", json.dumps(best_config, indent=4))
            
        if "df" in self.to_save and hasattr(pipeline, "df") and pipeline.df is not None:
            self._save_pickle_local(exp_prefix + "df_subsampleado.pkl", pipeline.df) #cambiar nombre
            
        if "scaler" in self.to_save and hasattr(pipeline, "scaler") and pipeline.scaler is not None:
            self._save_dataframe_local(exp_prefix + "scaler.csv", pipeline.scaler)
            
class CustomScalerStep(PipelineStep):
    """
    Calcula el std por serie (product_id, customer_id) usando solo datos
    hasta el periodo máximo definido (fecha <= max_period).
    Usa fallback al std por producto si std_cust_prod es bajo.
    Guarda en pipeline.scaler un DataFrame con ['product_id', 'customer_id', 'std_final'].
    
    Args:
        min_std_threshold (float): umbral para considerar std suficientemente alto.
        max_period (str or pd.Timestamp): fecha límite para filtrar datos (inclusive).
            Ejemplo: '2019-08'
    """
    def __init__(self, min_std_threshold: float = 0.001,
                 max_period: str = '2019-08',
                 name: Optional[str] = None):
        super().__init__(name)
        self.min_std_threshold = min_std_threshold
        self.max_period = max_period

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df

        # Verificar tipo de dato de 'fecha' y convertir a datetime solo si es necesario
        if isinstance(df['fecha'].dtype, pd.PeriodDtype):
            fecha_ts = df['fecha'].dt.to_timestamp()
        else:
            fecha_ts = pd.to_datetime(df['fecha'], format="%Y-%m")

        max_period_dt = pd.to_datetime(self.max_period, format="%Y-%m")

        # Filtrar solo datos hasta max_period (inclusive)
        df_filtered = df[fecha_ts <= max_period_dt]

        # Calcular std por (product_id, customer_id) con datos filtrados
        std_cust_prod = df_filtered.groupby(['product_id', 'customer_id'])['tn'].std().reset_index()
        std_cust_prod.rename(columns={'tn': 'std_cust_prod'}, inplace=True)

        # Calcular std por product_id con datos filtrados
        std_prod = df_filtered.groupby('product_id')['tn'].std().reset_index()
        std_prod.rename(columns={'tn': 'std_prod'}, inplace=True)

        # Merge
        scaler_df = std_cust_prod.merge(std_prod, on='product_id', how='left')

        # Crear std_final
        mask = (scaler_df['std_cust_prod'].isna()) | (scaler_df['std_cust_prod'] < self.min_std_threshold)
        scaler_df['std_final'] = scaler_df['std_cust_prod']
        scaler_df.loc[mask, 'std_final'] = scaler_df.loc[mask, 'std_prod']
        scaler_df['std_final'] = scaler_df['std_final'].fillna(1.0)

        # Guardar solo columnas necesarias
        pipeline.scaler = scaler_df[['product_id', 'customer_id', 'std_final']]
  
class ScaleTnDerivedFeaturesStep(PipelineStep):
    """
    Escala columnas derivadas de 'tn' (lags, rolling stats y diferencias absolutas relacionadas a tn)
    usando el std_final guardado en pipeline.scaler para cada (product_id, customer_id).
    Crea nuevas columnas con sufijo '_scaled' sin modificar las originales.
    """
    def __init__(self, name: Optional[str] = None, base_feature_prefix='tn'):
        super().__init__(name)
        self.base_feature_prefix = base_feature_prefix  # Ejemplo: 'tn'

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.df

        if not hasattr(pipeline, "scaler"):
            raise ValueError("pipeline.scaler no está definido. Ejecutá CustomScalerStep primero.")

        scaler_df = pipeline.scaler

        # Merge para agregar std_final a cada fila según product_id y customer_id
        df = df.merge(scaler_df, on=['product_id', 'customer_id'], how='left')

        # Columnas a escalar: lags, rolling, y diferencias que se basen en la feature base (ej. 'tn')
        cols_to_scale = []
        for col in df.columns:
            if (
                col == self.base_feature_prefix or
                col.startswith(f"{self.base_feature_prefix}_lag_") or
                col.startswith(f"{self.base_feature_prefix}_rolling_") or
                (f"{self.base_feature_prefix}_diff_" in col)
            ):
                cols_to_scale.append(col)

        for col in cols_to_scale:
            df[f"{col}_scaled"] = df[col] / df['std_final']

        # Eliminar columna temporal
        df.drop(columns=['std_final'], inplace=True)

        pipeline.df = df
        
class PrecomputeSeriesWeightsStep(PipelineStep):
    """
    Calcula el promedio de tn por (customer_id, product_id) y lo guarda
    como diccionario directamente en el pipeline (no como artefacto).
    """

    def __init__(self, tn_col: str = "tn", name: Optional[str] = None):
        super().__init__(name)
        self.tn_col = tn_col

    def execute(self, pipeline: "Pipeline") -> None:
        df = pipeline.df

        # Calcular promedio tn por serie
        avg_tn = df.groupby(["customer_id", "product_id"])[self.tn_col].mean()

        # Guardar como diccionario en memoria
        pipeline.weight_dict = avg_tn.to_dict()

class AssignPrecomputedWeightsStep(PipelineStep):
    """
    Asigna los pesos precomputados desde pipeline.weight_dict al df actual.
    Guarda el vector resultante en pipeline.sample_weights.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, pipeline: "Pipeline") -> None:
        df = pipeline.df

        # Usar el diccionario en memoria
        weight_dict = pipeline.weight_dict

        # Mapear pesos por fila
        weights = df.apply(
            lambda row: weight_dict.get((row["customer_id"], row["product_id"]), 1.0),
            axis=1
        )

        # Guardar en memoria
        weights = pd.Series(weights.values, index=df.index)
        pipeline.sample_weights = weights
        
class CustomMetricDelta:
    def __init__(self, df_eval, product_id_col='product_id', scaler=None):
        self.scaler = scaler
        self.df_eval = df_eval
        self.product_id_col = product_id_col

    def __call__(self, preds, train_data):
        import numpy as np

        labels = train_data.get_label()
        df_temp = self.df_eval.copy()
        df_temp['delta_preds'] = preds
        df_temp['delta_labels'] = labels

        if self.scaler:
            df_temp['delta_preds'] = self.scaler.inverse_transform(df_temp[['delta_preds']])
            df_temp['delta_labels'] = self.scaler.inverse_transform(df_temp[['delta_labels']])

        # Verificamos que exista 'tn' (último tn conocido)
        if 'tn' not in df_temp.columns:
            raise ValueError("df_eval debe tener la columna 'tn' con el último valor de tn conocido")

        # Invertir para obtener tn predicha y tn real a partir de delta
        df_temp['preds'] = df_temp['tn'] - df_temp['delta_preds']
        df_temp['labels'] = df_temp['tn'] - df_temp['delta_labels']

        # Agrupar por producto y sumar tn reales y predichos
        por_producto = df_temp.groupby(self.product_id_col).agg({'labels': 'sum', 'preds': 'sum'})

        # Calcular error relativo absoluto (igual que antes)
        denominador = por_producto['labels'].abs()
        denominador[denominador < 1e-6] = 1e-6  # para evitar división por cero
        error = np.sum(np.abs(por_producto['labels'] - por_producto['preds'])) / np.sum(denominador)

        return 'custom_error', error, False


                                      
#### ---- Pipeline Execution ---- ####
experiment_name = "exp_lgbm_target_delta_train_pesos_1672025" #Nombre del experimento para guardar resultados
pipeline = Pipeline(
    steps=[
        LoadDataFrameFromPickleStep(path="/home/tomifernandezlabo3/gcs-bucket/datasets/df_fe.pkl"), ## Cambiar por el path correcto del pickle
        CustomScalerStep(),
        ScaleTnDerivedFeaturesStep(),
        ReduceMemoryUsageStep(),  
        WeightedSubsampleSeriesStep(sample_fraction=0.30),
        SaveResults(exp_name=experiment_name, to_save=["df"]), #guardar df subsampleado
        PrecomputeSeriesWeightsStep(tn_col="tn"),    
        AssignPrecomputedWeightsStep(),
        CastDataTypesStep(dtypes=
            {
                "edad_customer_producto": "float32", 
                "periodos_desde_ultima_compra": "float32",
            }
        ),
        SplitDataFrameStep(),
        PrepareXYStep(),
        OptunaLGBMOptimizationStep(n_trials=100, study_name="exp_lgbm_target_delta_train_pesos_1672025"),
        SaveResults(exp_name=experiment_name,to_save=["best_params","optuna_trials","scaler","log"]),
    ],
    experiment_name=experiment_name,
    )

try:
    pipeline.run(verbose=True)

except Exception as e:
    pipeline.logger.error("Pipeline failed with an exception:", exc_info=True)

finally:
    try:
        local_log_path = pipeline.log_filename
        # Ruta dentro del bucket montado localmente
        bucket_mounted_path = "/home/tomifernandezlabo3/gcs-bucket"
        log_dest_dir = os.path.join(bucket_mounted_path, "experiments", experiment_name)
        os.makedirs(log_dest_dir, exist_ok=True)

        log_dest_path = os.path.join(log_dest_dir, "pipeline_log.txt")

        if os.path.exists(local_log_path):
            shutil.copy2(local_log_path, log_dest_path)
            print(f"Log file copied to mounted bucket path: {log_dest_path}")

    except Exception as log_upload_error:
        print("Error copying log to mounted bucket path:", log_upload_error)
        traceback.print_exc()