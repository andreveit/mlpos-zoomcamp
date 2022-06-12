import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner

from datetime import datetime

import pickle

@task
def read_data(path):
    logger = get_run_logger()
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
        # print(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
        # print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    logger = get_run_logger()

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    # print(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")
    # print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    # print(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()

    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    # print(f"The MSE of validation is: {mse}")
    return


def setup_date(date=None):
    if date is None:
        date=datetime.today().date()
        date=str(date)
    
    return date

@task
def get_paths(date):
    date = setup_date(date)
    date = datetime.strptime(date, "%Y-%m-%d")

    data_basepath="./03-orchestration/data/fhv_tripdata_"
    filetype=".parquet"
    
    train_path=data_basepath+str(date.year)+"-"+str(date.month-2).zfill(2)+filetype
    val_path=data_basepath+str(date.year)+"-"+str(date.month-1).zfill(2)+filetype
    return train_path,val_path


def save_model(date, lr, dv):
    date = setup_date(date)

    with open(f'./03-orchestration/models/model-{date}.bin', 'wb') as f:
        pickle.dump(lr, f)

    with open(f'./03-orchestration/models/dv-{date}.b', 'wb') as f:
        pickle.dump(dv, f)



@flow(task_runner=SequentialTaskRunner())
def main(date = None):

    train_path, val_path = get_paths(date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    
    save_model(date, lr, dv)

    run_model(df_val_processed, categorical, dv, lr)
    print('Done!')


# main(date = "2021-08-15")

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    name="model_trainig",
    flow=main,
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="America/New_York"),
        flow_runner=SubprocessFlowRunner(),
        tags=["ml"]
)

