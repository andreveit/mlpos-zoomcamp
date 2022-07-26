from datetime import datetime
import pandas as pd
import json
from utils import Model
import pickle


def load_data(filename):
    with open('data/'+filename, 'rb') as f:
        return pickle.load(f)


def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)


data = [
    (None, None, dt(1, 2), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
    (1, 1, dt(1, 2, 0), dt(2, 2, 1)),        
]

columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
df = pd.DataFrame(data, columns=columns)
categorical = ['PUlocationID', 'DOlocationID']


class ModelMock(Model):
    def __init__(self):
        pass

def test_preprocess_data():
    expected = load_data('preprocess_data_expected.pck')

    actual = ModelMock().preprocess_data(df, categorical)
    pd.testing.assert_frame_equal(actual, expected)