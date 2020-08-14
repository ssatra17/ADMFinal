import uvicorn
from fastapi import FastAPI,Query
import joblib
import pandas as pd

# Models
rlrmc = open("RLRMC_final.sav","rb")
rlrmc_model = joblib.load(rlrmc)

# init app
app = FastAPI(title="Riemannian Low-rank Matrix Completion algorithm",
              description='''Riemannian Low-rank Matrix Completion (RLRMC) is a matrix factorization based (vanilla) matrix completion algorithm that solves the optimization problem using Riemannian conjugate gradients algorithm.''',
              version="0.1.0",)

df = pd.read_csv('RLRMC_final.csv')
df['userID'] = df['userID'].apply(str)

# Routes
@app.get('/')
async def index():
	return {"text":"Hello"}

@app.get('/user/{user_id}')
async def get_users(user_id:str):
	return {"user_id":user_id}

@app.get('/product/{item_id}')
async def get_item(item_id:str):
	return {"item_id":item_id}

# ML Aspect
@app.get('/predict/')
async def predict(user_id:str = 17375, item_id:str = 22325):
    user_list = []
    item_list = []
    user_list.append(user_id)
    item_list.append(item_id)
    prediction = rlrmc_model.predict(user_list, item_list)
    if prediction[0] > 0:
        result = df.loc[df['userID'] == user_id].sort_values(by = 'prediction', ascending = False).head(5)
        result = result['itemID'].reset_index(drop = True)
        final = result.values.tolist()
    else:
        final = "Not Found"
    return final

if __name__ == '__main__':
	uvicorn.run(app,host="127.0.0.1",port=8000)
