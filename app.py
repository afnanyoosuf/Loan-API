from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import logging
from logger import *

app = FastAPI()

model = joblib.load("model/loan_model.pkl")

class LoanData(BaseModel):
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float

@app.get("/")
def home():
    return {"message": "Loan Approval API"}

@app.post("/predict")
def predict(data: LoanData):

    logging.info(
        f"Prediction Request: {data.dict()}"
    )

    prediction = model.predict(
        [[
            data.ApplicantIncome,
            data.CoapplicantIncome,
            data.LoanAmount,
            data.Loan_Amount_Term,
            data.Credit_History
        ]]
    )[0]

    result = "Approved" if prediction == 1 else "Rejected"

    logging.info(
        f"Prediction Result: {result}"
    )

    return {
        "loan_status": result
    }