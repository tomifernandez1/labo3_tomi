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
    def __init__(self, input_key: str = "df", name: Optional[str] = None):
        super().__init__(name)
        self.input_key = input_key
            
    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.get_artifact(self.input_key)
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
        self.save_artifact(pipeline, "df", df) 

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
        pipeline.save_artifact("df", df)

class SplitCustomerProductByGroupStep(PipelineStep):
    """
    Divide el DataFrame original en N subsets, cada uno con todas las fechas pero solo un subconjunto de combinaciones (product_id, customer_id).
    """
    def __init__(self, n_splits=3, name: Optional[str] = None):
        super().__init__(name)
        self.n_splits = n_splits

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.get_artifact("df")
        combos = df[["product_id", "customer_id"]].drop_duplicates().reset_index(drop=True)
        combos = combos.sample(frac=1, random_state=42).reset_index(drop=True)  # <-- SHUFFLE
        combos["subset"] = combos.index % self.n_splits
        for i in range(self.n_splits):
            combos_i = combos[combos["subset"] == i][["product_id", "customer_id"]]
            merged = df.merge(combos_i, on=["product_id", "customer_id"], how="inner")
            merged = merged.sort_values(["product_id", "customer_id", "fecha"])
            self.save_artifact(pipeline, f"df_subset_{i+1}", merged)

class CastDataTypesStep(PipelineStep):
    def __init__(self, dtypes: Dict[str, str], name: Optional[str] = None):
        super().__init__(name)
        self.dtypes = dtypes

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.get_artifact("df")
        for col, dtype in self.dtypes.items():
            df[col] = df[col].astype(dtype)
        df.info()
        self.save_artifact(pipeline, "df", df)


class ChangeDataTypesStep(PipelineStep):
    def __init__(self, dtypes: Dict[str, str], name: Optional[str] = None):
        super().__init__(name)
        self.dtypes = dtypes

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.get_artifact("df")
        for original_dtype, dtype in self.dtypes.items():
            for col in df.select_dtypes(include=[original_dtype]).columns:
                df[col] = df[col].astype(dtype)
        df.info()
        self.save_artifact(pipeline, "df", df)


class FilterFirstDateStep(PipelineStep):
    def __init__(self, first_date: str, name: Optional[str] = None):
        super().__init__(name)
        self.first_date = first_date

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.get_artifact("df")
        df = df[df["fecha"] >= self.first_date]
        print(f"Filtered DataFrame shape: {df.shape}")
        self.save_artifact(pipeline, "df", df)
        
class ShareTNFeaturesStep(PipelineStep):
    """
    Crea features de share de tn respecto al total por product_id y por customer_id en cada fecha.
    """
    def __init__(self, tn_col="tn", name: Optional[str] = None):
        super().__init__(name)
        self.tn_col = tn_col

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.get_artifact("df")
        # Share respecto al total del producto en la fecha
        df["share_tn_product"] = df[self.tn_col] / df.groupby(["fecha", "product_id"])[self.tn_col].transform("sum")
        # Share respecto al total del customer en la fecha
        df["share_tn_customer"] = df[self.tn_col] / df.groupby(["fecha", "customer_id"])[self.tn_col].transform("sum")
        for cat in ["cat1", "cat2", "cat3", "brand"]:
            col_name = f"share_tn_{cat}"
            df[col_name] = df[self.tn_col] / df.groupby(["fecha", cat])[self.tn_col].transform("sum")        
        self.save_artifact(pipeline, "df", df)

class FeatureEngineeringLagStep(PipelineStep):
    def __init__(self, lags: List[int], columns: List, name: Optional[str] = None):
        super().__init__(name)
        self.lags = lags
        self.columns = columns

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.get_artifact("df")
        for col in self.columns:
            for lag in self.lags:
                df[f"{col}_lag_{lag}"] =  df.groupby(['product_id', 'customer_id'])[col].shift(lag)
        self.save_artifact(pipeline, "df", df)
        
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
        df = pipeline.get_artifact("df")
        for col in self.columns:
            for ref_type in self.ref_types:
                for window in self.window:
                    ref_col = f"{col}_{ref_type}_{window}"
                    diff_col = f"{col}_diff_{ref_type}_{window}"
                    if ref_col in df.columns:
                        df[diff_col] = df[col] - df[ref_col]
        self.save_artifact(pipeline, "df", df)

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
        df = pipeline.get_artifact("df")
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
        self.save_artifact(pipeline, "df", df)
        
class CustomerIdMeanTnAndRequestByDateStep(PipelineStep):
    """
    Agrega columnas con el promedio de tn y cust_request_qty de cada customer_id por fecha.
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.get_artifact("df")
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
        self.save_artifact(pipeline, "df", df)

class ProductIdMeanTnAndRequestByCustomerStep(PipelineStep):
    """
    Agrega columnas con el promedio de tn y cust_request_qty por customer para cada product_id.
    Es decir, para cada fila, la media de tn y cust_request_qty de ese producto entre todos sus customers.
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.get_artifact("df")
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
        self.save_artifact(pipeline, "df", df)

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
        df = pipeline.get_artifact("df")
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

        self.save_artifact(pipeline, "df", df)
    

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
        df = pipeline.get_artifact("df")
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

        self.save_artifact(pipeline, "df", df)
                
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
        df = pipeline.get_artifact("df")
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

        self.save_artifact(pipeline, "df", df)

class RollingIsMaxFeatureStep(PipelineStep):
    def __init__(self, window: List[int], columns: List, name: Optional[str] = None):
        super().__init__(name)
        self.window = window
        self.columns = columns

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.get_artifact("df")
        for col in self.columns:
            for win in self.window:
                max_rolling = (
                    df.groupby(['product_id', 'customer_id'])[col]
                    .transform(lambda x: x.rolling(win, min_periods=1).max())
                )
                df[f"{col}_is_rolling_max_{win}"] = (df[col] == max_rolling).astype(int)
        self.save_artifact(pipeline, "df", df)
        
class RollingIsMinFeatureStep(PipelineStep):
    def __init__(self, window: List[int], columns: List, name: Optional[str] = None):
        super().__init__(name)
        self.window = window
        self.columns = columns

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.get_artifact("df")
        for col in self.columns:
            for win in self.window:
                min_rolling = (
                    df.groupby(['product_id', 'customer_id'])[col]
                    .transform(lambda x: x.rolling(win, min_periods=1).min())
                )
                df[f"{col}_is_rolling_min_{win}"] = (df[col] == min_rolling).astype(int)
        self.save_artifact(pipeline, "df", df)
    
class CreateTotalCategoryStep(PipelineStep):
    def __init__(self, name: Optional[str] = None, cat: str = "cat1", tn: str = "tn"):
        super().__init__(name)
        self.cat = cat
        self.tn = tn
    
    def execute(self, pipeline: "Pipeline") -> None:
        df = pipeline.get_artifact("df")
        df = df.sort_values(['fecha', self.cat])
        df[f"{self.tn}_{self.cat}_vendidas"] = (
            df.groupby(['fecha', self.cat])[self.tn]
              .transform('sum')
        )
        self.save_artifact(pipeline, "df", df)

class FeatureEngineeringProductInteractionStep(PipelineStep):

    def execute(self, pipeline: Pipeline) -> None:
        """
        El dataframe tiene una columna product_id y customer_id y fecha.
        Quiero obtener los x productos con mas tn del ultimo mes y crear x nuevas columnas que es la suma de tn de esos productos.
        se deben agregan entonces respetando la temporalidad la columna product_{product_id}_total_tn
        """
        df = pipeline.get_artifact("df")
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
        self.save_artifact(pipeline, "df", df)
        
class ProductInteractionLagsStep(PipelineStep):
    """
    Calcula lags para las columnas product_{product_id}_total_tn.
    """
    def __init__(self, lags=[1,2,3], name: Optional[str] = None):
        super().__init__(name)
        self.lags = lags

    def execute(self, pipeline: "Pipeline") -> None:
        df = pipeline.get_artifact("df")
        product_total_cols = [col for col in df.columns if col.startswith("product_") and col.endswith("_total_tn")]
        df = df.sort_values(by=['fecha'])
        for col in product_total_cols:
            for lag in self.lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        self.save_artifact(pipeline, "df", df)

class ProductInteractionRollingMeanStep(PipelineStep):
    """
    Calcula medias móviles para las columnas product_{product_id}_total_tn.
    """
    def __init__(self, windows=[3,6,12], name: Optional[str] = None):
        super().__init__(name)
        self.windows = windows

    def execute(self, pipeline: "Pipeline") -> None:
        df = pipeline.get_artifact("df")
        product_total_cols = [col for col in df.columns if col.startswith("product_") and col.endswith("_total_tn")]
        df = df.sort_values(by=['fecha'])
        for col in product_total_cols:
            for window in self.windows:
                df[f"{col}_rolling_mean_{window}"] = df[col].rolling(window, min_periods=1).mean()
        self.save_artifact(pipeline, "df", df)

class ProductInteractionRollingMinMaxStep(PipelineStep):
    """
    Calcula mínimos y máximos móviles para las columnas product_{product_id}_total_tn.
    """
    def __init__(self, windows=[3,6,12], name: Optional[str] = None):
        super().__init__(name)
        self.windows = windows

    def execute(self, pipeline: "Pipeline") -> None:
        df = pipeline.get_artifact("df")
        product_total_cols = [col for col in df.columns if col.startswith("product_") and col.endswith("_total_tn")]
        df = df.sort_values(by=['fecha'])
        for col in product_total_cols:
            for window in self.windows:
                df[f"{col}_rolling_max_{window}"] = df[col].rolling(window, min_periods=1).max()
                df[f"{col}_rolling_min_{window}"] = df[col].rolling(window, min_periods=1).min()
        self.save_artifact(pipeline, "df", df)


class FeatureEngineeringProductCatInteractionStep(PipelineStep):

    def __init__(self, cat="cat1", name: Optional[str] = None):
        super().__init__(name)
        self.cat = cat


    def execute(self, pipeline: Pipeline) -> None:
        # agrupo el dataframe por cat1 (sumando), obteniendo fecha, cat1 y
        # luego paso el dataframe a wide format, donde cada columna es una categoria  y la fila es la suma de tn para cada cat1
        # luego mergeo al dataframe original por fecha y product_id
        df = pipeline.get_artifact("df")
        df_cat = df.groupby(["fecha", self.cat]).agg({"tn": "sum"}).reset_index()
        df_cat = df_cat.pivot(index="fecha", columns=self.cat, values="tn").reset_index()
        df = df.merge(df_cat, on="fecha", how="left")
        self.save_artifact(pipeline, "df", df)
        
class OutlierPasoFeatureStep(PipelineStep):
    def __init__(self, fecha_outlier: str = "2019-08-01", name: Optional[str] = None):
        super().__init__(name)
        self.fecha_outlier = pd.to_datetime(fecha_outlier)

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.get_artifact("df")
        df["outlier_paso"] = (df["fecha"] == self.fecha_outlier).astype(np.uint8)
        self.save_artifact(pipeline, "df", df)
        
class PeriodsSinceLastPurchaseStep(PipelineStep):
    def __init__(self, tn_col: str = "tn", name: Optional[str] = None):
        super().__init__(name)
        self.tn_col = tn_col

    def execute(self, pipeline: "Pipeline") -> None:
        df = pipeline.get_artifact("df")
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
        self.save_artifact(pipeline, "df", df)
        
class CantidadUltimaCompraStep(PipelineStep):
    """
    Crea una feature 'cantidad_ultima_compra' que, para cada fila, contiene el valor de tn de la última compra (tn>0)
    previa para cada combinación de product_id y customer_id. Si nunca hubo compra previa, devuelve np.nan.
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.get_artifact("df")
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
        self.save_artifact(pipeline, "df", df)
        
class DiferenciaTNUltimaCompraStep(PipelineStep):
    """
    Crea una feature 'diferencia_tn_ultima_compra' que es la diferencia entre tn y cantidad_ultima_compra para cada registro.
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.get_artifact("df")
        if "cantidad_ultima_compra" not in df.columns:
            raise ValueError("La columna 'cantidad_ultima_compra' no existe. Ejecuta CantidadUltimaCompraStep antes.")
        df["diferencia_tn_ultima_compra"] = df["tn"] - df["cantidad_ultima_compra"]
        self.save_artifact(pipeline, "df", df)
        
class PeriodsSinceLastPurchaseCustomerLevelStep(PipelineStep):
    """
    Crea una feature 'periodos_desde_ultima_compra_customer' que indica, para cada fila,
    la cantidad de períodos desde la última compra de ese customer (sin importar el producto).
    """
    def __init__(self, tn_col: str = "tn", name: Optional[str] = None):
        super().__init__(name)
        self.tn_col = tn_col

    def execute(self, pipeline: "Pipeline") -> None:
        df = pipeline.get_artifact("df")
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
        self.save_artifact(pipeline, "df", df)
        
class ProductIdBuyersCountByDateStep(PipelineStep):
    """
    Agrega una columna 'product_buyers_count_fecha' al DataFrame, que indica para cada fila
    cuántos customers compraron ese product_id en esa fecha (tn > 0).
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.get_artifact("df")
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
        self.save_artifact(pipeline, "df", df)

class CustomerIdUniqueProductsByDateStep(PipelineStep):
    """
    Agrega una columna 'customer_unique_products_fecha' al DataFrame, que indica para cada fila
    cuántos productos diferentes compró ese customer en esa fecha (tn > 0).
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.get_artifact("df")
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
        self.save_artifact(pipeline, "df", df)      
        
class ProductSizeCategoryStep(PipelineStep):
    def __init__(self, sku_size_col: str = "sku_size", name: Optional[str] = None):
        super().__init__(name)
        self.sku_size_col = sku_size_col

    def execute(self, pipeline: "Pipeline") -> None:
        df = pipeline.get_artifact("df")
        def categorize(size):
            if size < 200:
                return "small"
            elif size < 800:
                return "medium"
            else:
                return "large"
        df["product_size"] = df[self.sku_size_col].apply(categorize)
        self.save_artifact(pipeline, "df", df)
        
class DateRelatedFeaturesStep(PipelineStep):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.get_artifact("df")
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
        self.save_artifact(pipeline, "df", df)

        
class SplitDataFrameStep(PipelineStep):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, pipeline: Pipeline) -> None:
        df = pipeline.get_artifact("df")
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
        self.save_artifact(pipeline, "train", train)
        self.save_artifact(pipeline, "eval_data", eval_data)
        self.save_artifact(pipeline, "test", test)
        self.save_artifact(pipeline, "kaggle_pred", kaggle_pred)
        self.save_artifact(pipeline, "df_intermedio", df_intermedio)
        self.save_artifact(pipeline, "df_final", df_final)


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
        train = pipeline.get_artifact("train")
        eval_data = pipeline.get_artifact("eval_data")
        test = pipeline.get_artifact("test")
        kaggle_pred = pipeline.get_artifact("kaggle_pred")
        df_intermedio = pipeline.get_artifact("df_intermedio")
        df_final = pipeline.get_artifact("df_final")

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
        y_test = test[target]

        X_train_final = df_final[features]
        y_train_final = df_final[target]
        X_kaggle = kaggle_pred[features]
        
        X_train_intermedio = df_intermedio[features]
        y_train_intermedio = df_intermedio[target]
                
        self.save_artifact(pipeline, "X_train", X_train)
        self.save_artifact(pipeline, "y_train", y_train)
        self.save_artifact(pipeline, "X_train_alone", X_train_alone)
        self.save_artifact(pipeline, "y_train_alone", y_train_alone)
        self.save_artifact(pipeline, "X_eval", X_eval)
        self.save_artifact(pipeline, "y_eval", y_eval)
        self.save_artifact(pipeline, "X_test", X_test)
        self.save_artifact(pipeline, "y_test", y_test)
        self.save_artifact(pipeline, "X_train_intermedio", X_train_intermedio)
        self.save_artifact(pipeline, "y_train_intermedio", y_train_intermedio)
        self.save_artifact(pipeline, "X_train_final", X_train_final)
        self.save_artifact(pipeline, "y_train_final", y_train_final)
        self.save_artifact(pipeline, "X_kaggle", X_kaggle)
        

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
        X_train = pipeline.get_artifact(self.train_eval_sets["X_train"])
        y_train = pipeline.get_artifact(self.train_eval_sets["y_train"])
        X_eval = pipeline.get_artifact(self.train_eval_sets["X_eval"])
        y_eval = pipeline.get_artifact(self.train_eval_sets["y_eval"])
        df_eval = pipeline.get_artifact(self.train_eval_sets["eval_data"])

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
            lgb.early_stopping(200),
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
        self.save_artifact(pipeline, "model", model)
        
class OptunaLGBMOptimizationStep(PipelineStep):
    def __init__(self, n_trials=50, name: Optional[str] = None):
        super().__init__(name)
        self.n_trials = n_trials

    def execute(self, pipeline: Pipeline) -> None:
        
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
                
        X_train = pipeline.get_artifact("X_train")
        y_train = pipeline.get_artifact("y_train")
        X_eval = pipeline.get_artifact("X_eval")
        y_eval = pipeline.get_artifact("y_eval")
        df_eval = pipeline.get_artifact("eval_data")
     
        if scaler_target is not None:
            custom_metric = CustomMetric(df_eval, product_id_col='product_id', scaler=scaler_target)
        else:
            pipeline.logger.warning("CustomMetric created without scaler_target.")
            custom_metric = CustomMetric(df_eval, product_id_col='product_id')        
            
        cat_features = [col for col in X_train.columns if X_train[col].dtype.name == 'category']

        def objective(trial):
            train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
            eval_data = lgb.Dataset(X_eval, label=y_eval, reference=train_data, categorical_feature=cat_features)
            callbacks = [lgb.early_stopping(200)]
            param = {
                'num_leaves': trial.suggest_int('num_leaves', 100, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'random_state': 42,
                'boosting_type': 'gbdt',
                'objective': 'tweedie',
                'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.9),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.9),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 100),
                'verbose': -1,
                'max_bin': trial.suggest_int('max_bin', 255, 1000),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-4, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-4, 10.0, log=True),
                'n_jobs': -1,
            }
            num_boost_rounds = trial.suggest_int('num_boost_rounds', 500, 2000)

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
            del model
            gc.collect()
            return score
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        best_trial = study.best_trial
        best_params = best_trial.params.copy()
        best_num_boost_rounds = best_trial.user_attrs.get("num_boost_rounds", 1000)

        self.save_artifact(pipeline, "best_lgbm_params", best_params)
        self.save_artifact(pipeline, "best_num_boost_rounds", best_num_boost_rounds)

        # -----------------------------
        # Guardar tabla completa de trials
        # -----------------------------
        trial_rows = []
        for trial in study.trials:
            row = {
                'trial': trial.number,
                'score': trial.user_attrs.get("score", None),
                'num_boost_rounds': trial.user_attrs.get("num_boost_rounds", None),
                **trial.params
            }
            trial_rows.append(row)

        df_trials = pd.DataFrame(trial_rows).sort_values(by="score")
        self.save_artifact(pipeline, "optuna_trials_df", df_trials)

        
class PredictStep(PipelineStep):
    def __init__(self, predict_set: str, name: Optional[str] = None):
        super().__init__(name)
        self.predict_set = predict_set

    def execute(self, pipeline: Pipeline) -> None:
        X_predict = pipeline.get_artifact(self.predict_set)
        scaler = pipeline.get_artifact("scaler")
        if scaler:
            X_predict[scaler.feature_names_in_] = scaler.transform(X_predict[scaler.feature_names_in_])
        model = pipeline.get_artifact("model_testing")  # or "model" for final model
        predictions = model.predict(X_predict)
        # los valores de predictions que dan menores a 0 los seteo en 0
        scaler_target = pipeline.get_artifact("scaler_target")
        if scaler_target:
            predictions = scaler_target.inverse_transform(predictions.reshape(-1, 1)).flatten()
        # la columna de predictions seria "predictions" y le agrego columna de product_id
        predictions = pd.DataFrame(predictions, columns=["predictions"], index=X_predict.index)
        predictions["product_id"] = X_predict["product_id"]
        self.save_artifact(pipeline, "predictions", predictions)


class EvaluatePredictionsSteps(PipelineStep):
    def __init__(self, y_actual_df: str, name: Optional[str] = None):
        super().__init__(name)
        self.y_actual_df = y_actual_df

    def execute(self, pipeline: Pipeline) -> None:
        predictions = pipeline.get_artifact("predictions")
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
        self.save_artifact(pipeline, "eval_df", eval_df)
        self.save_artifact(pipeline, "total_error", total_error)


class PlotFeatureImportanceStep(PipelineStep):

    def execute(self, pipeline: Pipeline) -> None:
        model = pipeline.get_artifact("model")
        lgb.plot_importance(model)
        
class TrainFinalModelLGBTestingStep(PipelineStep):
    """
    Entrena el modelo final LightGBM usando el dataset hasta Septiembre para testear sobre Octubre
    con los mejores hiperparámetros encontrados por Optuna.
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, pipeline: Pipeline) -> None:
        # Cargar datos
        X_train_final = pipeline.get_artifact("X_train_intermedio")
        y_train_final = pipeline.get_artifact("y_train_intermedio")

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

        # Cargar hiperparámetros óptimos
        best_params = pipeline.get_artifact("best_lgbm_params")
        best_num_boost_rounds = pipeline.get_artifact("best_num_boost_rounds")

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
        # Guardar modelo
        self.save_artifact(pipeline, "model_testing", model)
        
class TrainFinalModelLGBKaggleStep(PipelineStep):
    """
    Entrena el modelo final LightGBM usando todo el dataset (X_train_final, y_train_final)
    con los mejores hiperparámetros encontrados por Optuna.
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def execute(self, pipeline: Pipeline) -> None:
        # Cargar datos
        X_train_final = pipeline.get_artifact("X_train_final")
        y_train_final = pipeline.get_artifact("y_train_final")

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

        # Cargar hiperparámetros óptimos
        best_params = pipeline.get_artifact("best_lgbm_params")
        best_num_boost_rounds = pipeline.get_artifact("best_num_boost_rounds")

        # Categorías
        cat_features = [col for col in X_train_final.columns if X_train_final[col].dtype.name == 'category']

        # Dataset final
        train_data = lgb.Dataset(X_train_final, label=y_train_final, categorical_feature=cat_features)

        # Entrenamiento del modelo final
        model = lgb.train(
            best_params,
            train_data,
            num_boost_round=best_num_boost_rounds
        )

        # Guardar modelo entrenado
        self.save_artifact(pipeline, "model", model)
        
class SaveFeatureImportanceStep(PipelineStep):
    def execute(self, pipeline: Pipeline) -> None:
        model = pipeline.get_artifact("model")
        # Obtener importancia y nombres de features
        importance = model.feature_importance(importance_type='split')
        features = model.feature_name()
        df_importance = pd.DataFrame({
            "feature": features,
            "importance": importance
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        # Guardar como artefacto
        self.save_artifact(pipeline, "feature_importance_df", df_importance)        
        
class FilterProductsIDStep(PipelineStep):
    def __init__(self, product_file = "product_id_apredecir201912.txt", dfs=["df"], name: Optional[str] = None):
        super().__init__(name)
        self.file = product_file
        self.dfs = dfs

    def execute(self, pipeline: Pipeline) -> None:
        """ el txt es un csv que tiene columna product_id separado por tabulaciones """
        converted_dfs = {}
        for df_key in self.dfs:
            df = pipeline.get_artifact(df_key)
            product_ids = pd.read_csv(self.file, sep="\t")["product_id"].tolist()
            df = df[df["product_id"].isin(product_ids)]
            converted_dfs[df_key] = df
            print(f"Filtered DataFrame {df_key} shape: {df.shape}")
            pipeline.save_artifact(df_key, df)
        return converted_dfs        

class KaggleSubmissionStep(PipelineStep):
    def execute(self, pipeline: Pipeline) -> None:
        model = pipeline.get_artifact("model")
        kaggle_pred = pipeline.get_artifact("kaggle_pred")
        X_kaggle = pipeline.get_artifact("X_kaggle")

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

        self.save_artifact(pipeline, "submission", submission)
        
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
    """
    # Ruta fija donde está montado el bucket de GCS
    BASE_BUCKET_PATH = "/home/tomifernandezlabo3/gcs-bucket"

    def __init__(self, exp_name: str, name: Optional[str] = None):
        super().__init__(name)
        self.exp_name = exp_name

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

    def execute(self, pipeline: Pipeline) -> None:
        total_error = pipeline.get_artifact("total_error")
        exp_prefix = f"experiments/{self.exp_name}_error_test_{total_error:.4f}/"

        self._save_dataframe_local(exp_prefix + "submission.csv", pipeline.get_artifact("submission"))
        self._save_dataframe_local(exp_prefix + "feature_importance.csv", pipeline.get_artifact("feature_importance_df"))
        self._save_dataframe_local(exp_prefix + "optuna_trials.csv", pipeline.get_artifact("optuna_trials_df"))
        self._save_string_local(exp_prefix + "total_error.txt", str(total_error))
        self._save_pickle_local(exp_prefix + "model.pkl", pipeline.get_artifact("model"))

        log_local_path = pipeline.log_filename
        if os.path.exists(log_local_path):
            log_dest_path = os.path.join(self.BASE_BUCKET_PATH, exp_prefix + "pipeline_log.txt")
            os.makedirs(os.path.dirname(log_dest_path), exist_ok=True)
            import shutil
            shutil.copy2(log_local_path, log_dest_path)
            
                    
#### ---- Pipeline Execution ---- ####
experiment_name = f"exp_lgbm_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
pipeline = Pipeline(
    steps=[
        LoadDataFrameStep(path="/home/tomifernandezlabo3/gcs-bucket/df_inicial.parquet"),
        OutlierPasoFeatureStep(fecha_outlier="2019-08-01"),
        ProductSizeCategoryStep(sku_size_col="sku_size"),
        CastDataTypesStep(dtypes=
            {
                "cat1": "category", 
                "cat2": "category",
                "cat3": "category",
                "brand": "category",
                "product_size": "category",
                "outlier_paso": "category"
            }
        ),
        PeriodsSinceLastPurchaseStep(tn_col="tn"),
        PeriodsSinceLastPurchaseCustomerLevelStep(tn_col="tn"),
        CantidadUltimaCompraStep(),
        DiferenciaTNUltimaCompraStep(),
        ShareTNFeaturesStep(),
        CreateTotalCategoryStep(cat="cat1"),
        CreateTotalCategoryStep(cat="cat2"),
        CreateTotalCategoryStep(cat="cat3"),
        CreateTotalCategoryStep(cat="brand"),
        #FeatureEngineeringProductInteractionStep(),
        ProductIdMeanTnAndRequestByCustomerStep(),
        CustomerIdMeanTnAndRequestByDateStep(),
        ProductIdBuyersCountByDateStep(),
        CustomerIdUniqueProductsByDateStep(),
        ChangeDataTypesStep(dtypes={
            "float64": "float32",
        }),
        FeatureEngineeringLagStep(lags=[1,2,3], columns=["tn"]),
        #FeatureEngineeringLagStep(lags=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36], columns=["tn", "cust_request_qty","tn_cat1_vendidas", "tn_cat2_vendidas", "tn_cat3_vendidas", "tn_brand_vendidas", "share_tn_product", "share_tn_customer", "share_tn_cat1", "share_tn_cat2", "share_tn_cat3", "share_tn_brand","product_mean_tn_by_customer","product_mean_cust_request_qty_by_customer","customer_mean_tn_by_fecha","customer_mean_cust_request_qty_by_fecha", "customer_id_unique_products_purchased", "product_id_unique_customers"]),
        RollingMeanFeatureStep(window=[2,3], columns=["tn"]),
        RollingMaxFeatureStep(window=[2,3], columns=["tn"]),
        RollingMinFeatureStep(window=[2,3], columns=["tn"]),
        RollingStdFeatureStep(window=[2,3], columns=["tn"]),
        DiferenciaVsReferenciaStep(columns=["tn"], ref_types=["lag"], window=[2,3]),
        DiferenciaVsReferenciaStep(columns=["tn"], ref_types=["rolling_mean"], window=[2,3]),
        DiferenciaVsReferenciaStep(columns=["tn"], ref_types=["rolling_max"], window=[2,3]),
        DiferenciaVsReferenciaStep(columns=["tn"], ref_types=["rolling_min"], window=[2,3]),
        #DiferenciaVsReferenciaStep(columns=["tn", "cust_request_qty","tn_cat1_vendidas", "tn_cat2_vendidas", "tn_cat3_vendidas", "tn_brand_vendidas","product_mean_tn_by_customer","product_mean_cust_request_qty_by_customer","customer_mean_tn_by_fecha","customer_mean_cust_request_qty_by_fecha", "customer_id_unique_products_purchased", "product_id_unique_customers"], ref_types=["lag"], window=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]),
        #ProductInteractionLagsStep(lags=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]),
        ReduceMemoryUsageStep(),
        DateRelatedFeaturesStep(),
        CastDataTypesStep(dtypes=
            {
                "product_id": "uint32", 
                "customer_id": "uint32",
                "mes": "uint16",
                "quarter": "uint16",
                "year": "uint16",
                "periodo": "uint16",
                "periodos_desde_ultima_compra": "float32",
            }
        ),
        SplitDataFrameStep(),
        PrepareXYStep(),
        OptunaLGBMOptimizationStep(n_trials=3),
        TrainFinalModelLGBTestingStep(),
        PredictStep(predict_set="X_test"),
        EvaluatePredictionsSteps(y_actual_df="y_test"),
        TrainFinalModelLGBKaggleStep(),
        SaveFeatureImportanceStep(),
        FilterProductsIDStep(dfs=["X_kaggle","kaggle_pred"]),   
        KaggleSubmissionStep(),
        SaveResults(exp_name=experiment_name)
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
