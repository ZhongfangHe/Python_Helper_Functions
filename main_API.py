'''
Use FastAPI to build an API based on the model built in "example_API.py".

Before that, use the following commands in Windows CMD to create a subfolder "API_app" 
to store the API files. Note that it is a convention to write an empty file "__init__.py" there.
1. mkdir API_app
2. type nul > API_app\__init__.py

After writing the script, use the following commands in Windows CMD to build the API:
    1. myenv\Scripts\activate: This is to enter the virtual environment "myenv".
    2. py -m uvicorn API_app.main:app --reload: Note the folder uses '.' instead of '\'
'''

from fastapi import FastAPI
from pydantic import BaseModel
import pickle, numpy as np, os


# Load the trained model
#file_dir = 'C:\\Users\\hezho\\Documents\\Learning Programming\\Python\\2026Feb\\API_model\\'
#file_name = 'diabetes_model.pkl'
#model_path = file_dir + file_name
model_path = os.path.join("API_model", "diabetes_model.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)
    

# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Progression Predictor",
    description="Predicts diabetes progression score from physiological features",
    version="1.0.0"
)


# Define input data schema
class PatientData(BaseModel):
    age: float
    sex: float  
    bmi: float
    bp: float   # blood pressure
    s1: float   # serum measurement 1
    s2: float   # serum measurement 2  
    s3: float   # serum measurement 3
    s4: float   # serum measurement 4
    s5: float   # serum measurement 5
    s6: float   # serum measurement 6
    
    class ConfigDict:
        json_schema_extra = {
            "example": {
                "age": 0.05,
                "sex": 0.05,
                "bmi": 0.06,
                "bp": 0.02,
                "s1": -0.04,
                "s2": -0.04,
                "s3": -0.02,
                "s4": -0.01,
                "s5": 0.01,
                "s6": 0.02
            }
        }
        

# Create a prediction endpoint
@app.post("/predict")
def predict_progression(patient: PatientData):
    """
    Predict diabetes progression score
    """
    # Convert input to 2D numpy array
    features = np.array([[
        patient.age, patient.sex, patient.bmi, patient.bp,
        patient.s1, patient.s2, patient.s3, patient.s4,
        patient.s5, patient.s6
    ]])
    
    # Make prediction
    prediction = model.predict(features)[0] #make "prediction" a number instead of an array
    
    # Return result with additional context
    return {
        "predicted_progression_score": round(prediction, 2),
        "interpretation": get_interpretation(prediction)
    }
def get_interpretation(score):
    """Provide human-readable interpretation of the score"""
    if score < 100:
        return "Below average progression"
    elif score < 150:
        return "Average progression"
    else:
        return "Above average progression"

 
# Create a health check endpoint       
@app.get("/")
def health_check():
    return {"status": "healthy", "model": "diabetes_progression_v1"}


