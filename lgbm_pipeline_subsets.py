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
    def __init__(self, steps: Optional[List[PipelineStep]] = None, optimize_arftifacts_memory: bool = True):
        """Initialize the pipeline."""
        self.steps: List[PipelineStep] = steps if steps is not None else []
        self.artifacts: Dict[str, Any] = {}
        self.last_step = None
        self.optimize_arftifacts_memory = optimize_arftifacts_memory

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
        Save an artifact from a given step.

        Args:
            artifact_name (str): Name to identify the artifact.
            artifact (Any): The artifact to save.
        """
        if not self.optimize_arftifacts_memory:
            self.artifacts[artifact_name] = artifact
        else:
            # Usa el directorio temporal del sistema operativo
            tmp_dir = tempfile.gettempdir()
            artifact_path = os.path.join(tmp_dir, artifact_name)
            with open(artifact_path, 'wb') as f:
                pickle.dump(artifact, f)
            self.artifacts[artifact_name] = artifact_path

    def get_artifact(self, artifact_name: str) -> Any:
        """
        Retrieve a stored artifact.

        Args:
            artifact_name (str): Name of the artifact to retrieve.

        Returns:
            Any: The requested artifact.
        """
        if not self.optimize_arftifacts_memory:
            return self.artifacts.get(artifact_name)
        else:
            artifact_path = self.artifacts.get(artifact_name)
            if artifact_path and os.path.exists(artifact_path):
                with open(artifact_path, 'rb') as f:
                    return pickle.load(f)
            else:
                warnings.warn(f"Artifact {artifact_name} not found in temp directory")
                return None
    
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
        """        
        
        # Run steps from the last completed step
        for step in self.steps:
            if verbose:
                print(f"Executing step: {step.name}")
            start_time = time.time()
            self.before_step_callback() 
            step.execute(self)
            self.after_step_callback()
            end_time = time.time()
            if verbose:
                print(f"Step {step.name} completed in {end_time - start_time:.2f} seconds")
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

class LoadDataFrameStep(PipelineStep):
    """
    Example step that loads a DataFrame.
    """
    def __init__(self, path: str, name: Optional[str] = None):
        super().__init__(name)
        self.path = path

    def execute(self, pipeline: Pipeline) -> None:
        df = pd.read_parquet(self.path)
        df = df.drop(columns=["periodo"])
        self.save_artifact(pipeline, "df", df)

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
    def __init__(self, window: List[int], columns: List[str], name: Optional[str] = None):
        super().__init__(name)
        self.window = window
        self.columns = columns

    def execute(self, pipeline: "Pipeline") -> None:
        df = pipeline.get_artifact("df")
        df = df.sort_values(by=['product_id', 'customer_id', 'fecha'])
        grouped = df.groupby(['product_id', 'customer_id'])
        for col in self.columns:
            for window in self.window:
                df[f'{col}_rolling_mean_{window}'] = grouped[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
        self.save_artifact(pipeline, "df", df)
    

class RollingMaxFeatureStep(PipelineStep):
    def __init__(self, window: List[int], columns: List[str], name: Optional[str] = None):
        super().__init__(name)
        self.window = window
        self.columns = columns

    def execute(self, pipeline: "Pipeline") -> None:
        df = pipeline.get_artifact("df")
        df = df.sort_values(by=['product_id', 'customer_id', 'fecha'])
        grouped = df.groupby(['product_id', 'customer_id'])
        for col in self.columns:
            for window in self.window:
                df[f'{col}_rolling_max_{window}'] = grouped[col].transform(
                    lambda x: x.rolling(window, min_periods=1).max()
                )
        self.save_artifact(pipeline, "df", df)
    

class RollingMinFeatureStep(PipelineStep):
    def __init__(self, window: List[int], columns: List[str], name: Optional[str] = None):
        super().__init__(name)
        self.window = window
        self.columns = columns

    def execute(self, pipeline: "Pipeline") -> None:
        df = pipeline.get_artifact("df")
        df = df.sort_values(by=['product_id', 'customer_id', 'fecha'])
        grouped = df.groupby(['product_id', 'customer_id'])
        for col in self.columns:
            for window in self.window:
                df[f'{col}_rolling_min_{window}'] = grouped[col].transform(
                    lambda x: x.rolling(window, min_periods=1).min()
                )
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
        # Features cíclicas senos y cosenos para mes y quarter:
        df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
        df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)
        df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
        df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)    
        # Feature de periodo secuencial
        fechas_ordenadas = np.sort(df["fecha"].unique())
        fecha_a_periodo = {fecha: i+1 for i, fecha in enumerate(fechas_ordenadas)}
        df["periodo"] = df["fecha"].map(fecha_a_periodo)
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
        self.save_artifact(pipeline, "train", train)
        self.save_artifact(pipeline, "eval_data", eval_data)
        self.save_artifact(pipeline, "test", test)
        self.save_artifact(pipeline, "kaggle_pred", kaggle_pred)


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

        X_train_final = pd.concat([train[features], eval_data[features], test[features]])
        y_train_final = pd.concat([train[target], eval_data[target], test[target]])

        X_kaggle = kaggle_pred[features]
        self.save_artifact(pipeline, "X_train", X_train)
        self.save_artifact(pipeline, "y_train", y_train)
        self.save_artifact(pipeline, "X_train_alone", X_train_alone)
        self.save_artifact(pipeline, "y_train_alone", y_train_alone)
        self.save_artifact(pipeline, "X_eval", X_eval)
        self.save_artifact(pipeline, "y_eval", y_eval)
        self.save_artifact(pipeline, "X_test", X_test)
        self.save_artifact(pipeline, "y_test", y_test)
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
        
class OptunaLGBMOptimizationStep(PipelineStep): #CHECK
    """
    Step para optimización bayesiana de hiperparámetros de LightGBM usando Optuna.
    Usa la custom metric del pipeline si está disponible.
    """
    def __init__(self, n_trials=20, name: Optional[str] = None):
        super().__init__(name)
        self.n_trials = n_trials

    def execute(self, pipeline: Pipeline) -> None:
        import optuna
        import io
        from contextlib import redirect_stdout
        from IPython.display import display, HTML
        import lightgbm as lgb
        from sklearn.metrics import mean_squared_error

        X_train = pipeline.get_artifact("X_train")
        y_train = pipeline.get_artifact("y_train")
        X_eval = pipeline.get_artifact("X_eval")
        y_eval = pipeline.get_artifact("y_eval")
        df_eval = pipeline.get_artifact("eval_data")
        scaler = pipeline.get_artifact("scaler")
        scaler_target = pipeline.get_artifact("scaler_target")

        # Usa la custom metric del pipeline si existe
        custom_metric = CustomMetric(df_eval, product_id_col='product_id', scaler=scaler_target)

        def objective(trial):
            param = {
                'num_leaves': trial.suggest_int('num_leaves', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
                'num_boost_rounds': trial.suggest_int('num_boost_rounds', 500, 3000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'random_state': 42,
                'boosting_type': 'gbdt',
                'objective': 'tweedie',
                #'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.5),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 0.9),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 0.9),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
                'verbose': -1,
                'max_bin': trial.suggest_int('max_bin', 255, 1000),
                'verbose':-1
                #'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 0.1),
                #'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 0.1),
                
            }
            model = lgb.LGBMRegressor(**param)
            model.fit(
                X_train, y_train,
                eval_set=[(X_eval, y_eval)],
                eval_metric=None,
                feval=custom_metric,
                callbacks=[lgb.early_stopping(150)],
                verbose=False
            )
            preds = model.predict(X_eval)
            # Calcula la métrica custom igual que en feval
            _, score, _ = custom_metric(preds, lgb.Dataset(X_eval, label=y_eval))
            return score

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=self.n_trials)
        output = buffer.getvalue()
        max_lines = 30
        lines = output.split('\n')
        short_output = '\n'.join(lines[:max_lines])
        display(HTML(f"<div style='max-height:300px;overflow:auto;white-space:pre;font-family:monospace'>{short_output}</div>"))

        print("Mejores hiperparámetros:", study.best_params)
        self.save_artifact(pipeline, "best_lgbm_params", study.best_params)
        
class PredictStep(PipelineStep):
    def __init__(self, predict_set: str, name: Optional[str] = None):
        super().__init__(name)
        self.predict_set = predict_set

    def execute(self, pipeline: Pipeline) -> None:
        X_predict = pipeline.get_artifact(self.predict_set)
        scaler = pipeline.get_artifact("scaler")
        if scaler:
            X_predict[scaler.feature_names_in_] = scaler.transform(X_predict[scaler.feature_names_in_])
        model = pipeline.get_artifact("model")
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
        
class TrainFinalModelLGBStep(PipelineStep):
    """
    Entrena un modelo LightGBM final usando todo el set de entrenamiento (X_train_final, y_train_final)
    sin set de evaluación ni early stopping.
    """
    def __init__(self, params: Dict = {}, name: Optional[str] = None):
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
        self.params = params

    def execute(self, pipeline: Pipeline) -> None:
        X_train_final = pipeline.get_artifact("X_train_final")
        y_train_final = pipeline.get_artifact("y_train_final")
        cat_features = [col for col in X_train_final.columns if X_train_final[col].dtype.name == 'category']

        scaler = pipeline.get_artifact("scaler")
        scaler_target = pipeline.get_artifact("scaler_target")
        if scaler:
            X_train_final[scaler.feature_names_in_] = scaler.transform(X_train_final[scaler.feature_names_in_])
            y_train_final = pd.Series(
                scaler_target.transform(y_train_final.values.reshape(-1, 1)).flatten(),
                index=y_train_final.index,
            )

        train_data = lgb.Dataset(X_train_final, label=y_train_final, categorical_feature=cat_features)
        model = lgb.train(
            self.params,
            train_data,
        )
        self.save_artifact(pipeline, "final_model", model)
        
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
        scaler = pipeline.get_artifact("scaler")
        if scaler:
            X_kaggle[scaler.feature_names_in_] = scaler.transform(X_kaggle[scaler.feature_names_in_])
        preds = model.predict(X_kaggle)
        scaler_target = pipeline.get_artifact("scaler_target")
        if scaler_target:
            preds = scaler_target.inverse_transform(preds.reshape(-1, 1)).flatten()
        #kaggle_pred["tn_predicha"] = model.predict(X_kaggle) # try using .loc[row_indexer, col_indexer] = value instead
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


        # Save a copy of the notebook
        #notebook_path = fallback_latest_notebook()
        #shutil.copy(notebook_path, os.path.join(exp_dir, f"notebook_{self.exp_name}.ipynb"))


#### ---- Pipeline Ejecution ---- ####

pipeline = Pipeline(
    steps=[
        LoadDataFrameStep(path="df_inicial.parquet"),
        SplitCustomerProductByGroupStep(n_splits=4),
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
        CantidadUltimaCompraStep(),
        DiferenciaTNUltimaCompraStep(),
        ShareTNFeaturesStep(),
        CreateTotalCategoryStep(cat="cat1"),
        CreateTotalCategoryStep(cat="cat2"),
        CreateTotalCategoryStep(cat="cat3"),
        CreateTotalCategoryStep(cat="brand"),
        FeatureEngineeringLagStep(lags=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,24,36], columns=["tn", "cust_request_qty","tn_cat1_vendidas", "tn_cat2_vendidas", "tn_cat3_vendidas", "tn_brand_vendidas", "share_tn_product", "share_tn_customer", "share_tn_cat1", "share_tn_cat2", "share_tn_cat3", "share_tn_brand"]),
        RollingMeanFeatureStep(window=[3,6,12,24,36], columns=["tn", "cust_request_qty"]),
        RollingMaxFeatureStep(window=[3,6,12,24,36], columns=["tn", "cust_request_qty"]),
        RollingMinFeatureStep(window=[3,6,12,24,36], columns=["tn", "cust_request_qty"]),
        DiferenciaVsReferenciaStep(columns=["tn", "cust_request_qty","tn_cat1_vendidas", "tn_cat2_vendidas", "tn_cat3_vendidas", "tn_brand_vendidas"], ref_types=["lag"], window=[1,2,3,4,5,6,7,8,9,10,11,12,18,24,36]),
        DiferenciaVsReferenciaStep(columns=["tn", "cust_request_qty"], ref_types=["rolling_mean"], window=[3,6,12,24,36]),
        DateRelatedFeaturesStep(),
        PeriodsSinceLastPurchaseStep(tn_col="tn"),
        PeriodsSinceLastPurchaseCustomerLevelStep(tn_col="tn"),
        #FeatureEngineeringProductInteractionStep(),
        #FeatureEngineeringProductCatInteractionStep(cat="cat1"),
        #FeatureEngineeringProductCatInteractionStep(cat="cat2"),
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
        ChangeDataTypesStep(dtypes={
            "float64": "float32",
        }),
        #FilterFirstDateStep(first_date="2018-01-01"), # para que sea mas liviano el dataset
        SplitDataFrameStep(),
        PrepareXYStep(),
        TrainModelLGBStep(params={
                "objective": "tweedie",
                "boosting_type": "gbdt",
                "num_leaves": 1000,
                "max_depth": 10,
                "learning_rate": 0.01,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "max_bin":800
        }, train_eval_sets={
                "X_train": "X_train_alone",
                "y_train": "y_train_alone",
                "X_eval": "X_eval",
                "y_eval": "y_eval",
                "eval_data": "eval_data",
            },
            num_boost_round=2000),
        SaveFeatureImportanceStep(),
        PredictStep(predict_set="X_test"),
        EvaluatePredictionsSteps(y_actual_df="test"),
        PlotFeatureImportanceStep(),
        FilterProductsIDStep(dfs=["X_kaggle"]),   
        KaggleSubmissionStepSubset(),
        SaveExperimentStep(exp_name=f"{datetime.datetime.now().strftime('%Y%m%d_%H%M')}_exp_lgbm", save_dataframes=False),
    ],
    optimize_arftifacts_memory=True
)
#pipeline.run(verbose=True)

pipeline.steps[0].execute(pipeline)  # LoadDataFrameStep
pipeline.steps[1].execute(pipeline)  # SplitCustomerProductByGroupStep

#Procesar los subsets generados por SplitCustomerProductByGroupStep
subset_keys = [k for k in pipeline.artifacts.keys() if k.startswith("df_subset_")]
subset_indices = sorted(int(k.split("_")[-1]) for k in subset_keys)

print("subset_keys:", subset_keys)
print("subset_indices:", subset_indices)

for i in subset_indices:
    df_subset = pipeline.get_artifact(f"df_subset_{i}")
    df_subset.to_csv(f"subset_{i}.csv", index=False)
    print(f"Guardado subset_{i}.csv con shape {df_subset.shape}")

"""for i in subset_indices:
    print(f"\n=== Procesando subset {i} ===")
    df_subset = pipeline.get_artifact(f"df_subset_{i}")
    pipeline.save_artifact("df", df_subset)
    for step in pipeline.steps[2:]:
        print(f"Ejecutando step: {step.name}")
        step.execute(pipeline)"""
