#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip freeze | grep scikit-learn')


# In[2]:


import pickle
import pandas as pd


# In[3]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[4]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[5]:


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-02.parquet')


# In[6]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# ## Q1

# In[7]:


y_pred.std()


# ## Q2

# In[8]:


df_result = pd.DataFrame()
df_result['ride_id'] = f'2022/02_' + df.index.astype('str')
df_result['y_pred'] = y_pred


# In[10]:


output_file = 'output.parquet'
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


# In[11]:


import os
size = os.path.getsize(output_file)    
print(f'DF result file size: {size / 1000} KB')


# In[12]:
import sys

def ride_duration_prediction(year: int, month: int):
    if month < 10:
        month = f'0{month}'

    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    
    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    return y_pred

def run():
    year = int(sys.argv[1]) # 2022
    month = int(sys.argv[2]) # 3

    prediction = ride_duration_prediction(
        year=year,
        month=month
    )

    print('Mean predicted duration: ', prediction.mean())

if __name__ == '__main__':
    run()