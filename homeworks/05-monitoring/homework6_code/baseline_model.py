import os
import requests
import datetime
import pandas as pd

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric

from joblib import load, dump
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

BASE_DATA_PATH = '../data/'
BASE_MODEL_PATH = 'homework6_code/models'

jan_data = pd.read_parquet(os.path.join(BASE_DATA_PATH,'green_tripdata_2022-01.parquet'))

# create target
jan_data["duration_min"] = jan_data.lpep_dropoff_datetime - jan_data.lpep_pickup_datetime
jan_data.duration_min = jan_data.duration_min.apply(lambda td : float(td.total_seconds())/60)

# filter out outliers
jan_data = jan_data[(jan_data.duration_min >= 0) & (jan_data.duration_min <= 60)]
jan_data = jan_data[(jan_data.passenger_count > 0) & (jan_data.passenger_count <= 8)]

# data labeling
target = "duration_min"
num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
cat_features = ["PULocationID", "DOLocationID"]

train_data = jan_data[:30000]
val_data = jan_data[30000:]

model = LinearRegression()
model.fit(train_data[num_features + cat_features], train_data[target])

with open(os.path.join(BASE_MODEL_PATH, 'lin_reg.bin'), 'wb') as f_out:
    dump(model, f_out)