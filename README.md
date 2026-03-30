Suspicious Transaction Detection
A machine learning project that tries to catch fraudulent or suspicious financial transactions before they cause real damage — using a mix of supervised (Random Forest) and unsupervised (Isolation Forest) techniques.

Problem Statement
Anyone who uses UPI, net banking, or a debit card has probably had that moment of panic — an unknown transaction on the statement, a message that wasn't you. Financial fraud isn't abstract; it hits real people. And with the sheer volume of digital transactions today, no team of humans can realistically review each one.
This project builds a pipeline that looks at how a transaction behaves — not just how much it's for — and decides whether it warrants a closer look. Late-night activity, sudden spikes in spending, suspicious merchants, transactions far from home — these are the kinds of signals the model learns to pick up on.

What the Project Does

Builds a realistic synthetic transaction dataset that mimics real-world spending patterns
Extracts and engineers features that actually mean something
Trains a Random Forest classifier to detect fraud using labeled examples
Trains an Isolation Forest as a backup — for when you don't have labeled data at all
Evaluates both models properly (not just accuracy — that's a trap with imbalanced data)
Includes a simple function to run a single transaction through the model in real time


Project Structure
suspicious_transaction_detection/
│
├── detection_model.py        # Everything — data, features, training, evaluation
├── requirements.txt          # What you need to install
└── README.md                 # You're reading it
Running the script will also generate three charts:

confusion_matrix.png
feature_importance.png
amount_distribution.png


Setup Instructions
1. Clone the repository
bashgit clone https://github.com/<your-username>/suspicious-transaction-detection.git
cd suspicious-transaction-detection
2. Set up a virtual environment (do this, it saves headaches)
bashpython -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
3. Install the dependencies
bashpip install -r requirements.txt
4. Run it
bashpython detection_model.py
```

---

## How It Works

### Features Used

| Feature | What it captures |
|---|---|
| `log_amount` | Transaction amount, log-transformed to reduce skew from outliers |
| `hour` | What time of day the transaction happened |
| `transactions_per_day` | How many transactions the user made that day |
| `distance_from_home_km` | How far the transaction was from the user's usual location |
| `is_foreign` | Whether the merchant is based outside the user's country |
| `merchant_risk_score` | A 0–1 risk score pre-assigned to the merchant |
| `is_night` | 1 if the transaction happened between 10 PM and 5 AM |
| `high_frequency` | 1 if the user made more than 8 transactions in a single day |
| `far_from_home` | 1 if the distance was over 100 km |
| `risk_x_foreign` | Combination of merchant risk and foreign flag — amplifies the signal when both are true |

### Models

**Random Forest Classifier**
Learns from labeled examples of normal and fraudulent transactions. It handles the class imbalance problem (there's a lot more normal data than fraud) by using `class_weight='balanced'`, which stops it from just predicting everything as normal and calling it a day. It also tells you *how confident* it is, not just what it predicted.

**Isolation Forest**
Doesn't need labels at all. The idea is simple: if a data point is weird, it should be easy to isolate with random cuts. Points that get isolated quickly are flagged as anomalies. It's less precise than the supervised model, but it's useful when you're starting from scratch with no fraud history to learn from.

---

## Sample Output
```
[1] Generating synthetic transaction data …
    Dataset shape : (5000, 13)
    Fraud count   : 250  (5.0%)

[6] Demo — single transaction inference …
    Transaction  : {'amount': 18500, 'hour': 2, 'transactions_per_day': 15, ...}
    Prediction   : SUSPICIOUS  (confidence = 0.9312)

Limitations
Worth being honest about what this doesn't do yet:

The data is synthetic. The model works well on it, but real transaction data is messier and the performance will likely differ.
There's no time-series memory — the model doesn't know what you spent last week, only what it sees in the current record.
Class imbalance is handled through class_weight, not SMOTE. It works, but it's not the most sophisticated approach.


What Could Be Better

Swap in real data from a bank export or open dataset
Add rolling features — spending velocity over 3 days, 7 days, etc.
Try XGBoost or LightGBM and see if they outperform Random Forest here
Build a small Streamlit interface so you can paste in a transaction and get an answer
Use SHAP values to explain why a specific transaction was flagged, not just that it was


Dependencies

Python 3.8+
pandas, numpy
scikit-learn
matplotlib, seaborn


Author
Made for the Fundamentals of AI and ML course as part of the BYOP (Bring Your Own Project) submission. The problem felt worth solving — so that's where it started.
