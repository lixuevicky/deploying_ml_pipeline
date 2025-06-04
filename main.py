from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from data import process_data
from model import inference

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

class InferenceInput(BaseModel):
    age: int
    workclass: str = Field(..., alias="workclass")
    fnlgt: int = Field(..., alias="fnlgt")
    education: str = Field(..., alias="education")
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str = Field(..., alias="occupation")
    relationship: str = Field(..., alias="relationship")
    race: str = Field(..., alias="race")
    sex: str = Field(..., alias="sex")
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }
# Load trained model
model = joblib.load('rfc_model.pkl')
encoder = joblib.load('encoder.pkl')
lb = joblib.load('lb.pkl')

# Instantiate the app.
app = FastAPI()


# Define a GET on the specified endpoint.
@app.get("/")
async def read_root() -> dict:
    return {"message": "Welcome to the inference API!"}

@app.post("/inference")
async def predict(input_data: InferenceInput):
    # Convert input to a DataFrame
    input_dict = input_data.dict(by_alias=True)
    df = pd.DataFrame([input_dict])

    # Preprocess the input
    X, _, _, _ = process_data(
    df, categorical_features=CAT_FEATURES, label=None, training=False, encoder=encoder, lb=lb)

    # Run model
    pred = inference(model, X)[0]

    return {"prediction": ">50K" if pred == 1 else "<=50K"}





