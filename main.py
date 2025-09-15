import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor, export_text
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# ------------------ Two-Tuple Fuzzy Model ------------------
class TwoTupleFuzzy:
    def __init__(self, labels=None):
        if labels is None:
            labels = ["VeryLow", "Low", "Medium", "High", "VeryHigh"]
        self.S = labels
        self.g = len(self.S) - 1

    def delta_S(self, value):
        i = int(round(value))
        i = max(0, min(self.g, i))
        alpha = value - i
        return (self.S[i], float(alpha))

    def numeric_to_2tuple(self, x):
        v = max(0.0, min(1.0, x))
        val_g = v * self.g
        return self.delta_S(val_g)

# ------------------ بارگذاری و آماده‌سازی ------------------
df = pd.read_csv("lending_club_loan_two.csv")

# هدف: early default (1=نکول زودهنگام، 0=پرداخت کامل)
df["early_default"] = df["loan_status"].apply(
    lambda x: 1 if x in ["Charged Off", "Late (31-120 days)", "Default"] else 0
)

# حذف ستون‌های غیرضروری
drop_cols = ["emp_title", "title", "address", "loan_status", "purpose"]
df = df.drop(columns=drop_cols)

# تبدیل term
df["term"] = df["term"].str.extract(r"(\d+)").astype(float)
df["term_60_months"] = (df["term"] == 60).astype(int)
df = df.drop(columns=["term"])

# تاریخ‌ها → طول سابقه اعتباری
df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%y", errors="coerce")
df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%y", errors="coerce")
df["credit_history_len"] = (df["issue_d"] - df["earliest_cr_line"]).dt.days
df = df.drop(columns=["issue_d", "earliest_cr_line"])

def clean_emp_length(val):
    if pd.isna(val):
        return 0
    val = str(val).lower().strip()
    if val == "n/a":
        return 0
    if "< 1" in val:
        return 0
    if "10+" in val:
        return 10
    digits = "".join([c for c in val if c.isdigit()])
    return int(digits) if digits else 0

df["emp_length"] = df["emp_length"].apply(clean_emp_length).astype(int)

# دسته‌ای‌ها → عدد
categorical_cols = ["grade", "sub_grade", "home_ownership",
                    "verification_status", "initial_list_status", "application_type"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# ------------------ X و y ------------------
X = df.drop(columns=["early_default"])
X = X.dropna(axis=1, how="all")
for col in X.columns:
    if X[col].dtype in ["float64", "int64"]:
        X[col] = X[col].fillna(X[col].median())
    else:
        X[col] = X[col].fillna("Unknown")
y = df["early_default"]

# تقسیم و بالانس
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

# ------------------ XGBoost ------------------
clf = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    max_depth=6,
    gamma=10,
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)
clf.fit(X_train_bal, y_train_bal)

# ارزیابی
probs = clf.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)
print("AUC:", roc_auc_score(y_test, probs))
print("Accuracy:", accuracy_score(y_test, preds))

# ------------------ Surrogate Tree ------------------
# پیش‌بینی روی کل داده
probs_blackbox = clf.predict_proba(X)[:, 1]

# surrogate regressor
tree = DecisionTreeRegressor(max_depth=4, random_state=42)
tree.fit(X, probs_blackbox)
pred_surrogate = tree.predict(X)
print("Surrogate R^2:", r2_score(probs_blackbox, pred_surrogate))

# استخراج قواعد درخت
rules_text = export_text(tree, feature_names=list(X.columns))
print("\n--- Surrogate Rules (Raw) ---")
print(rules_text)

# ------------------ تبدیل به برچسب فازی ------------------
fuzzy = TwoTupleFuzzy()

print("\n--- Sample Predictions with Fuzzy Labels ---")
sample_probs = probs[:10]
for i, p in enumerate(sample_probs):
    label, alpha = fuzzy.numeric_to_2tuple(p)
    print(f"Loan {i+1}: p={p:.3f} → {label} (Δ={alpha:+.2f})")
