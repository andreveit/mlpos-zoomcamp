import os
from typing import List
import sys
import pickle
import pandas as pd


class Model:   
    def __init__(self, model_filename, year: int, month: int) -> None:
        self._get_input_path(year, month)
        self._get_output_path(year, month)
        self._load_model(model_filename)
        self._load_data(self.data_filename)
    
    
    def _get_input_path(self, year: int, month: int) -> None:
        default_input_pattern = 'https://github.com/alexeygrigorev/datasets/raw/master/nyc-tlc/fhv/fhv_tripdata_{year:04d}-{month:02d}.parquet'

        input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
        self.data_filename = input_pattern.format(year=year, month=month)


    def _get_output_path(self, year: int, month: int) -> None:
        default_output_pattern = 's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
        output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
        self.output_file = output_pattern.format(year=year, month=month)


    def _load_data(self, filename: str) -> None:
        print('Loading data:\n',filename)
        self.input_data =  pd.read_parquet(filename)


    def _load_model(self, filename: str) -> None:
        with open(filename, 'rb') as f:
            self.dv, self.lr = pickle.load(f)


    def _predict(self,df: pd.DataFrame, categorical: List[str]) -> pd.DataFrame:
        df = df.copy()
        df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

        dicts = df[categorical].to_dict(orient='records')
        X_val = self.dv.transform(dicts)
        y_pred = self.lr.predict(X_val)

        df_result = pd.DataFrame()
        df_result['ride_id'] = df['ride_id']
        df_result['predicted_duration'] = y_pred

        print('predicted mean duration:', y_pred.mean())
        return df_result


    def predict(self, categorical: List[str]) -> pd.DataFrame:
        df = self.preprocess_data(self.input_data)
        return self._predict(df, categorical)


    def predict_nsave(self, categorical: List[str]) -> None:
        df = self.preprocess_data(self.input_data)
        df = self._predict(df, categorical)
        df.to_parquet(self.output_file, engine='pyarrow', index=False)


    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['duration'] = df.dropOff_datetime - df.pickup_datetime
        df['duration'] = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

        df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
        
        return df