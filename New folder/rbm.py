import uvicorn
from fastapi import FastAPI,Query
import joblib
import csv
import pandas as pd

# init app
app = FastAPI()

# Routes
@app.get('/')
async def index():
    return {"text":"Hello API Masters"}

@app.get('/user/{user_id}')
async def get_users(user_id:int):
	return {"user_id":user_id}

@app.get('/product/{product_id}')
async def get_products(product_id:int):
	return {"product_id":product_id}

# ML Aspect
@app.get('/predict/{user_id},{product_id}')
async def predict(user_id:int, product_id:int):
    columns_to_keep = ['userID', 'movieID']
    dataframe = pd.read_csv("RBM.csv", usecols=columns_to_keep)
    p = dataframe.loc[(dataframe['userID'] == 1) & (dataframe['movieID'] == 100), 'prediction'].values
    print(p)
    return {"results":p}
   # p = data.loc[(top_k_df['userID'] == 1) & (top_k_df['movieID'] == 100), 'prediction'].values

if __name__ == '__main__':
	uvicorn.run(app,host="127.0.0.1",port=8000)