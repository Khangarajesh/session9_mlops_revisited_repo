from pydantic import BaseModel
from fastapi import FastAPI
import pickle as pkl
import pathlib 
import pandas as pd

#create class for data type check
class predinputdata(BaseModel):
    col0:float
    col1:float
    col2:float
    col3:float
    col4:float
    col5:float
    col6:float
    col7:float
    col8:float
    col9:float
    col10:float
#model loading
model_path = 'models/model.pkl'
print(model_path)

with open(model_path, 'rb') as file:
    model = pkl.load(file)


#create an instance of 
app = FastAPI()

@app.get("/")
def home():
    return "helooo"
  
@app.post("/predict")  
def predict(input_data: predinputdata):
    features = {
        'col0':input_data.col0,
        'col1':input_data.col1,
        'col2':input_data.col2,
        'col3':input_data.col3,
        'col4':input_data.col4,
        'col5':input_data.col5,
        'col6':input_data.col6,
        'col7':input_data.col7,
        'col8':input_data.col8,
        'col9':input_data.col9,
        'col10':input_data.col10
    }
    
    features_df = pd.DataFrame(features, index = [0])
    ft = features_df.to_numpy()
    prediction = model.predict(ft)[0].item()
    
    return {"prediction" : prediction}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = "127.0.0.1", port = 8000)