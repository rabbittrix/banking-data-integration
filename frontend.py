import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import requests

st.title("Banking Fraud Detection & Customer Potential Dashboard")

# Conectar ao banco de dados SQLite
conn = sqlite3.connect("banking_data.db")
df = pd.read_sql("SELECT * FROM transactions", conn)
conn.close()

# Exibir visão geral das transações
st.subheader("Transaction Overview")
st.write(df.head())

# Corrigir o nome da coluna de fraude para 'is_fraud'
fraud_count = df["is_fraud"].value_counts()  # Alterado 'Fraud' para 'is_fraud'
fig_fraud = px.bar(fraud_count, x=fraud_count.index, y=fraud_count.values, labels={"x": "Fraud Status", "y": "Count"}, title="Fraudulent vs. Non-Fraudulent Transactions")
st.plotly_chart(fig_fraud)

# Exibir distribuição de renda
fig_income = px.histogram(df, x="income", title="Income Distribution of Customers")  # Certifique-se de usar o nome correto da coluna "income"
st.plotly_chart(fig_income)

# Exibir distribuição do score de crédito
fig_credit = px.histogram(df, x="credit_score", title="Credit Score Distribution")  # Certifique-se de usar o nome correto da coluna "credit_score"
st.plotly_chart(fig_credit)

# Exibir clientes potenciais de alto valor
st.subheader("Potential High-Value Customers")
potential_customers = df[(df["credit_score"] > 700) & (df["income"] > 80000) & (df["is_fraud"] == 0)]  # Alterado 'Fraud' para 'is_fraud'
st.write(potential_customers)

# Previsão de transações fraudulentas
st.subheader("Predict Fraudulent Transaction")
amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
income = st.number_input("Income", min_value=0.0, value=50000.0)

if st.button("Predict Fraud"):
    response = requests.post("http://localhost:8000/predict/", params={"amount": amount, "credit_score": credit_score, "income": income})
    if response.status_code == 200:
        prediction = response.json()["fraud_prediction"]
        st.write("Fraud Prediction:", "Fraudulent" if prediction == 1 else "Not Fraudulent")
    else:
       st.write("Error in prediction API.")
