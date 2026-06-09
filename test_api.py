from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_prediction():

    response = client.post(
        "/predict",
        json={
            "ApplicantIncome":5000,
            "CoapplicantIncome":0,
            "LoanAmount":120,
            "Loan_Amount_Term":360,
            "Credit_History":1
        }
    )

    assert response.status_code == 200