import uvicorn
from fastapi import FastAPI,Query
import joblib

# Models
import pickle
PIK = "RBM.dat"

with open(PIK, "rb") as f:
    print(pickle.load(f))

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
    datContent = [i.strip().split() for i in open("RBM.dat").readlines()]
	
    with open("RBM.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(datContent)
    columns_to_keep = ['userID', 'movieID']
    dataframe = pd.read_csv("RBM.csv", usecols=columns_to_keep)
    print(user_id)
    print(product_id)
    prediction = dataframe.loc[(top_k_df['userID'] == user_id) & (top_k_df['movieID'] == product_id)]
    if prediction[0] > 0:
        result = prediction
    else:
        result = "User doesn't exist"
    return {"prediction":result}

if __name__ == '__main__':
	uvicorn.run(app,host="127.0.0.1",port=8001)
