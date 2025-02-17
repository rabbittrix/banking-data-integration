# Banking Fraud Detection & Customer Potential Analysis

## Overview
This project is a complete banking fraud detection and customer potential analysis system. It generates random banking transaction data, integrates it into a database, and applies AI/ML models to:

- Detect fraudulent transactions
- Analyze customer financial behavior
- Identify potential high-value customers for bank services
- Provide an interactive dashboard for data visualization

## Tech Stack
- **Backend:** FastAPI (Python)
- **Frontend:** Streamlit (Python)
- **Database:** SQLite
- **Machine Learning:** Scikit-learn (Random Forest Classifier)
- **Data Generation:** Faker, Pandas, NumPy

## Project Structure
```
.
├── backend.py            # FastAPI backend for data processing and ML model
├── frontend.py           # Streamlit dashboard for visualization
├── banking_data.csv      # Generated transaction dataset
├── banking_data.db       # SQLite database
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
```

## Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/rabbittrix/banking-data-integration.git
   cd banking-data-integration
   ```
2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project
### 1. Start the Backend (FastAPI)
```bash
python backend.py
```
This will:
- Generate a dataset of 2000+ random banking transactions
- Store the data in an SQLite database
- Train a fraud detection ML model
- Start a REST API server on `http://localhost:8000`

### 2. Start the Frontend (Streamlit Dashboard)
```bash
streamlit run frontend.py
```
This will launch the interactive dashboard where users can:
- View transaction summaries and fraud detection analysis
- Explore customer credit scores and income distribution
- Identify high-value customers for banking services
- Use an AI-powered fraud prediction tool

## API Endpoints
| Method | Endpoint | Description |
|--------|---------|-------------|
| GET | `/transactions/` | Fetch all transactions |
| GET | `/frauds/` | Get fraudulent transactions |
| POST | `/predict/` | Predict if a transaction is fraudulent |

## Dashboard Features
- **Transaction Overview:** Summary of transactions with fraud detection
- **Customer Insights:** Credit score, income analysis
- **Fraud Detection:** Visualization of fraudulent vs. normal transactions
- **Potential Clients:** List of customers eligible for bank services
- **AI Fraud Prediction:** Predict fraud using user-input transaction details

## License
This project is licensed under the MIT License.

## Author
Developed by Roberto de Souza - rabbittrix@hotmail.com.

