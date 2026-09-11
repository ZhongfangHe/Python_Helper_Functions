'''
Use FastAPI to build an API based on the model built in "example_API2.py".

Before that, use the following commands in Windows CMD to create a subfolder "API_app" 
to store the API files. Note that it is a convention to write an empty file "__init__.py" there.
1. mkdir API_app
2. type nul > API_app\__init__.py

After writing the script, use the following commands in Windows CMD to build the API:
    1. myenv\Scripts\activate: This is to enter the virtual environment "myenv".
    2. py -m uvicorn API_app.main2:app --reload: Note the folder uses '.' instead of '\'
'''


from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os


# Load the model during startup
model_path = os.path.join("API_model", "linear_regression_model.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)
    

# Initialize FastAPI app
app = FastAPI(title="House Price Prediction API")


# Define the input data schema using Pydantic
class InputData(BaseModel):
    MedInc: float
    AveRooms: float
    AveOccup: float


# Create a prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    # Prepare the data for prediction
    input_features = [[data.MedInc, data.AveRooms, data.AveOccup]]
    
    # Make prediction using the loaded model
    prediction = model.predict(input_features)
    
    # Return the prediction result
    return {"predicted_house_price": prediction[0]}


# Create a health check endpoint
@app.get("/")
def health_check():
    return {"status": "healthy", "model": "house_pred_v1"}


# @app.get("/predict")
# def health_check():
#     return {"status": "healthy", "model": "house_pred_v2"}

