import pandas as pd
import numpy as np
import sqlite3
from faker import Faker
from fastapi import FastAPI, HTTPException
import uvicorn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

fake = Faker()
app = FastAPI()
fraud_model = None

def generate_banking_data(num_records=2000):
    data = []
    for _ in range(num_records):
        account_id = fake.uuid4()
        transaction_id = fake.uuid4()
        transaction_amount = round(np.random.uniform(10, 5000), 2)
        transaction_type = np.random.choice(["deposit", "withdrawal", "transfer", "payment"])
        location = fake.city()
        is_fraud = np.random.choice([0, 1], p=[0.98, 0.02])  # 2% fraud rate
        credit_score = np.random.randint(300, 850)
        income = round(np.random.uniform(20000, 150000), 2)

        data.append([ 
            account_id, transaction_id, transaction_amount, transaction_type, location, is_fraud, credit_score, income
        ])
        
    df = pd.DataFrame(data, columns=[
        "account_id", 
        "transaction_id", 
        "transaction_amount", 
        "transaction_type", 
        "location", 
        "is_fraud", 
        "credit_score", 
        "income"])

    df.to_csv("banking_data.csv", index=False)
    print("CSV file generated successfully!")
    return df   

# Database setup
def store_data_in_db(df):
    conn = sqlite3.connect("banking_data.db")
    df.to_sql("transactions", conn, if_exists="replace", index=False)
    conn.close()
    print("Data stored in database successfully.")

# Train fraud detection model
def train_fraud_model():
    conn = sqlite3.connect("banking_data.db")
    df = pd.read_sql("SELECT * FROM transactions", conn)
    conn.close()
    
    # Adjust column names to match the CSV generated columns
    X = df[["transaction_amount", "credit_score", "income"]]  # Corrected column names
    y = df["is_fraud"]  # Corrected target column

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Fraud detection model trained with accuracy: {accuracy:.2f}")
    
    return model 

@app.get("/transactions/")
def get_transactions():
    conn = sqlite3.connect("banking_data.db")
    df = pd.read_sql("SELECT * FROM transactions", conn)
    conn.close()
    return df.to_dict(orient="records")

@app.get("/frauds/")
def get_fraud_transactions():
    conn = sqlite3.connect("banking_data.db")
    df = pd.read_sql("SELECT * FROM transactions WHERE is_fraud = 1", conn)  # Corrected column name
    conn.close()
    return df.to_dict(orient="records")

@app.post("/predict/")
def predict_fraud(amount: float, credit_score: int, income: float):
    if fraud_model is None:
        raise HTTPException(status_code=500, detail="Fraud model is not trained.")
    
    prediction = fraud_model.predict([[amount, credit_score, income]])[0]
    return {"fraud_prediction": int(prediction)}

if __name__ == "__main__":
    df = generate_banking_data()
    store_data_in_db(df)
    fraud_model = train_fraud_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)
