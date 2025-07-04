import os
import random
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from dtaidistance import dtw
import time


class DTWSeriesProcessor:
    def __init__(self,
                 parquet_path: str,
                 use_sample: bool = False,
                 sample_size: int = 3000,
                 seed: int = 42,
                 log_file: str = None,
                 output_dir: str = None):
        """
        Parameters:
        - parquet_path: Ruta al archivo .parquet
        - use_sample: Si True, calcula DTW sobre una muestra aleatoria
        - sample_size: Tamaño de la muestra si use_sample es True
        - seed: Semilla aleatoria para reproducibilidad
        - log_file: Ruta al archivo de log (opcional)
        """
        self.parquet_path = parquet_path
        self.use_sample = use_sample
        self.sample_size = sample_size
        self.seed = seed
        self.df = None
        self.df_wide = None
        self.series = []
        self.index_keys = []
        self.sample_series = []
        self.sample_keys = []
        self.distance_matrix_sample = None
        self.output_dir = output_dir or "."

        self._setup_logging(log_file)

    def _setup_logging(self, log_file: str):
        log_format = "%(asctime)s [%(levelname)s] %(message)s"
        if log_file:
            logging.basicConfig(filename=log_file, level=logging.INFO, format=log_format)
        else:
            logging.basicConfig(level=logging.INFO, format=log_format)

    def load_and_prepare_data(self):
        logging.info("Cargando datos desde parquet...")
        self.df = pd.read_parquet(self.parquet_path)
        fechas_ordenadas = np.sort(self.df["fecha"].unique())
        fecha_a_periodo = {fecha: i + 1 for i, fecha in enumerate(fechas_ordenadas)}
        self.df["periodo"] = self.df["fecha"].map(fecha_a_periodo)
        logging.info("Datos cargados y columna 'periodo' creada.")

    def pivot_to_wide_format(self):
        logging.info("Convirtiendo a formato wide (product_id - customer_id por periodo)...")
        self.df_wide = self.df.pivot(index=["product_id", "customer_id"],
                                     columns="periodo",
                                     values="tn")
        logging.info("Formato wide creado correctamente.")

    def preprocess_series(self):
        logging.info("Preprocesando series con log1p y RobustScaler...")
        self.series = []
        self.index_keys = []

        for idx, row in self.df_wide.iterrows():
            serie = row.dropna().values
            if len(serie) < 2:
                continue  # Ignorar series muy cortas

            serie_log = np.log1p(serie)
            scaler = RobustScaler()
            serie_scaled = scaler.fit_transform(serie_log.reshape(-1, 1)).flatten()

            self.series.append(serie_scaled)
            self.index_keys.append(idx)

        logging.info(f"Series preprocesadas: {len(self.series)}")

    def select_series(self):
        """Elige si usar una muestra o todas las series."""
        if self.use_sample:
            logging.info(f"Seleccionando muestra aleatoria de {self.sample_size} series...")
            random.seed(self.seed)
            max_size = min(self.sample_size, len(self.series))
            indices = random.sample(range(len(self.series)), max_size)
            self.sample_series = [self.series[i] for i in indices]
            self.sample_keys = [self.index_keys[i] for i in indices]
            logging.info("Muestra seleccionada.")
        else:
            logging.info("Usando todas las series disponibles.")
            self.sample_series = self.series
            self.sample_keys = self.index_keys

    def compute_dtw_distance_matrix(self):
        logging.info("Calculando matriz de distancias DTW paralelizada...")
        self.distance_matrix_sample = dtw.distance_matrix_fast(
            self.sample_series,
            compact=False,
            parallel=True,
            use_c=True
        )
        logging.info(f"Matriz de distancias DTW calculada: shape = {self.distance_matrix_sample.shape}")

    def export_distance_matrix(self, filename: str = "distance_matrix_sample.csv"):
        os.makedirs(self.output_dir, exist_ok=True)  # Crear carpeta si no existe
        output_path = os.path.join(self.output_dir, filename)
        logging.info(f"Exportando matriz de distancias a CSV: {output_path}")
        df_out = pd.DataFrame(self.distance_matrix_sample,
                            index=self.sample_keys,
                            columns=self.sample_keys)
        df_out.to_csv(output_path)
        logging.info("Exportación completada.")

    def run_full_pipeline(self):
        logging.info("Iniciando pipeline completo DTW...")

        start = time.time()
        self.load_and_prepare_data()
        elapsed = time.time() - start
        logging.info(f"load_and_prepare_data tardó {elapsed:.2f} segundos")

        start = time.time()
        self.pivot_to_wide_format()
        elapsed = time.time() - start
        logging.info(f"pivot_to_wide_format tardó {elapsed:.2f} segundos")

        start = time.time()
        self.preprocess_series()
        elapsed = time.time() - start
        logging.info(f"preprocess_series tardó {elapsed:.2f} segundos")

        start = time.time()
        self.select_series()
        elapsed = time.time() - start
        logging.info(f"select_series tardó {elapsed:.2f} segundos")

        start = time.time()
        self.compute_dtw_distance_matrix()
        elapsed = time.time() - start
        logging.info(f"compute_dtw_distance_matrix tardó {elapsed:.2f} segundos")

        start = time.time()
        self.export_distance_matrix()
        elapsed = time.time() - start
        logging.info(f"export_distance_matrix tardó {elapsed:.2f} segundos")

        logging.info("Pipeline DTW finalizado correctamente.")


if __name__ == "__main__":
    # --- EJEMPLOS DE USO ---

    # Para ejecutar sobre una muestra:
    # processor = DTWSeriesProcessor("df_inicial.parquet", use_sample=True, sample_size=1000)

    # Para ejecutar sobre todo el dataset:
    processor = DTWSeriesProcessor(parquet_path="/home/tomifernandezlabo3/gcs-bucket/df_inicial.parquet",use_sample=True,sample_size=3000,seed=42, log_file="dtw_pipeline.log",output_dir="/home/tomifernandezlabo3/gcs-bucket")
    processor.run_full_pipeline()
