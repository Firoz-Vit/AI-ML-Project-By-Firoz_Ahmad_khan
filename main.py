import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ── data generation ──────────────────────────────────────────────────────────

def generate_transaction_data(n_samples=5000, fraud_ratio=0.05, random_state=42):
    """
    Simulate a realistic transaction dataset.
    Most transactions are normal; a small fraction are fraudulent/suspicious.
    """
    np.random.seed(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    # normal transactions
    normal = pd.DataFrame({
        'amount':           np.random.lognormal(mean=4.5, sigma=1.2, size=n_normal),
        'hour':             np.random.randint(7, 22, size=n_normal),      # business hours
        'transactions_per_day': np.random.poisson(3, size=n_normal),
        'distance_from_home_km': np.random.exponential(scale=15, size=n_normal),
        'is_foreign':       np.random.choice([0, 1], size=n_normal, p=[0.92, 0.08]),
        'merchant_risk_score': np.random.uniform(0.0, 0.3, size=n_normal),
        'label':            0
    })

    # fraudulent / suspicious transactions
    fraud = pd.DataFrame({
        'amount':           np.random.lognormal(mean=6.5, sigma=1.8, size=n_fraud),
        'hour':             np.random.choice(list(range(0, 6)) + list(range(22, 24)),
                                             size=n_fraud),                # odd hours
        'transactions_per_day': np.random.poisson(12, size=n_fraud),       # burst activity
        'distance_from_home_km': np.random.exponential(scale=200, size=n_fraud),
        'is_foreign':       np.random.choice([0, 1], size=n_fraud, p=[0.35, 0.65]),
        'merchant_risk_score': np.random.uniform(0.5, 1.0, size=n_fraud),
        'label':            1
    })

    df = pd.concat([normal, fraud], ignore_index=True).sample(frac=1, random_state=random_state)
    df['amount'] = df['amount'].clip(upper=50000).round(2)
    df['distance_from_home_km'] = df['distance_from_home_km'].round(2)
    return df


# ── feature engineering ───────────────────────────────────────────────────────

def engineer_features(df):
    df = df.copy()
    df['log_amount'] = np.log1p(df['amount'])
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['high_frequency'] = (df['transactions_per_day'] > 8).astype(int)
    df['far_from_home'] = (df['distance_from_home_km'] > 100).astype(int)
    df['risk_x_foreign'] = df['merchant_risk_score'] * df['is_foreign']
    return df


# ── model training ────────────────────────────────────────────────────────────

def train_supervised(X_train, y_train):
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X_train, y_train)
    return clf


def train_anomaly_detector(X_train):
    iso = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )
    iso.fit(X_train)
    return iso


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(clf, X_test, y_test, model_name="Random Forest"):
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None

    print(f"\n{'='*55}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*55}")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Suspicious"]))

    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
        print(f"  ROC-AUC Score : {auc:.4f}")

    return y_pred


def plot_confusion_matrix(y_test, y_pred, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Suspicious'],
                yticklabels=['Normal', 'Suspicious'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix — Suspicious Transaction Detection')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {save_path}")


def plot_feature_importance(clf, feature_names, save_path="feature_importance.png"):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(9, 5))
    plt.bar(range(len(feature_names)), importances[indices], color='steelblue')
    plt.xticks(range(len(feature_names)),
               [feature_names[i] for i in indices], rotation=30, ha='right')
    plt.ylabel('Importance Score')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Feature importance chart saved → {save_path}")


def plot_amount_distribution(df, save_path="amount_distribution.png"):
    plt.figure(figsize=(9, 4))
    for label, color in [(0, 'steelblue'), (1, 'tomato')]:
        subset = df[df['label'] == label]['amount']
        plt.hist(subset, bins=60, alpha=0.6,
                 label='Normal' if label == 0 else 'Suspicious',
                 color=color, density=True)
    plt.xlabel('Transaction Amount (₹)')
    plt.ylabel('Density')
    plt.title('Transaction Amount Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Amount distribution chart saved → {save_path}")


# ── inference helper ──────────────────────────────────────────────────────────

def predict_single(clf, scaler, feature_names, transaction: dict):
    """
    Classify a single transaction dict.
    Returns (label, confidence_score).
    """
    row = pd.DataFrame([transaction])
    row = engineer_features(row)
    row = row[feature_names]
    row_scaled = scaler.transform(row)
    label = clf.predict(row_scaled)[0]
    confidence = clf.predict_proba(row_scaled)[0][1]
    return ("SUSPICIOUS" if label == 1 else "NORMAL"), round(confidence, 4)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n[1] Generating synthetic transaction data …")
    df = generate_transaction_data(n_samples=5000, fraud_ratio=0.05)
    print(f"    Dataset shape : {df.shape}")
    print(f"    Fraud count   : {df['label'].sum()}  "
          f"({df['label'].mean()*100:.1f}%)")

    print("\n[2] Engineering features …")
    df = engineer_features(df)

    feature_cols = [
        'log_amount', 'hour', 'transactions_per_day',
        'distance_from_home_km', 'is_foreign', 'merchant_risk_score',
        'is_night', 'high_frequency', 'far_from_home', 'risk_x_foreign'
    ]

    X = df[feature_cols]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print("\n[3] Training Random Forest classifier …")
    rf_model = train_supervised(X_train_sc, y_train)
    y_pred_rf = evaluate_model(rf_model, X_test_sc, y_test, "Random Forest")

    print("\n[4] Training Isolation Forest (unsupervised baseline) …")
    iso_model = train_anomaly_detector(X_train_sc)
    iso_preds_raw = iso_model.predict(X_test_sc)
    iso_preds = np.where(iso_preds_raw == -1, 1, 0)   # -1 → suspicious
    print(f"    Isolation Forest flagged : {iso_preds.sum()} transactions as suspicious")

    print("\n[5] Generating visualisations …")
    plot_confusion_matrix(y_test, y_pred_rf)
    plot_feature_importance(rf_model, feature_cols)
    plot_amount_distribution(df)

    print("\n[6] Demo — single transaction inference …")
    sample_txn = {
        'amount': 18500,
        'hour': 2,
        'transactions_per_day': 15,
        'distance_from_home_km': 430,
        'is_foreign': 1,
        'merchant_risk_score': 0.82,
    }
    result, score = predict_single(rf_model, scaler, feature_cols, sample_txn)
    print(f"    Transaction  : {sample_txn}")
    print(f"    Prediction   : {result}  (confidence = {score})")

    print("\n✓ Pipeline complete.\n")


if __name__ == "__main__":
    main()
