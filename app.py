import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Credit Score Predictor", page_icon="💳", layout="wide")

MODEL_PATH = Path("outputs/credit_score_prediction_pipeline.joblib")
SCORE_MIN, SCORE_MAX = 300, 850
ARC_LEN = 3.14159265 * 80  # length of the gauge's half-circle path

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
:root {
    --bg: #0C1220; --surface: #131B2E; --line: #25304A;
    --text: #E6EAF4; --muted: #8E99B5; --accent: #7B9CFF;
}
.stApp { background: var(--bg); }
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp input,
.hero, .sec-title, .result { font-family: 'Manrope', system-ui, sans-serif; }
.block-container { max-width: 1180px; padding-top: 2.5rem; }

/* Header */
.hero { margin-bottom: 1.5rem; }
.pill { display: inline-block; padding: .2rem .75rem; margin-bottom: .9rem; border: 1px solid var(--line);
        border-radius: 999px; color: var(--accent); font-size: .78rem; font-weight: 600; }
.hero-title { margin: 0; color: var(--text); font-size: 2.1rem; font-weight: 800; letter-spacing: -.02em; }
.hero-sub { max-width: 62ch; margin: .45rem 0 0; color: var(--muted); line-height: 1.55; }

/* Form cards */
[data-testid="stForm"] { border: none; padding: 0; }
div[class*="st-key-card"] { background: var(--surface); border: 1px solid var(--line);
                            border-radius: 14px; padding: 1.1rem 1.25rem 1.3rem; }
.sec-title { margin: 0 0 .5rem; color: var(--text); font-size: 1.02rem; font-weight: 700; }
.sub-title { margin: .9rem 0 .2rem; color: var(--muted); font-size: .85rem; font-weight: 600; }
[data-testid="stWidgetLabel"] p { color: var(--muted); font-size: .85rem; font-weight: 600; }
.stApp [data-baseweb="input"], .stApp [data-baseweb="select"] > div { border-radius: 10px; }

/* Submit button */
div[data-testid="stFormSubmitButton"] > button { width: 100%; height: 3.1em; border: none; border-radius: 12px;
                                                  background: var(--accent); transition: background .15s; }
div[data-testid="stFormSubmitButton"] > button:hover { background: #9BB5FF; }
div[data-testid="stFormSubmitButton"] > button:focus-visible { outline: 2px solid #fff; outline-offset: 2px; }
div[data-testid="stFormSubmitButton"] > button p { color: #0C1220; font-size: 1rem; font-weight: 700; }

/* Result */
.result { display: flex; flex-wrap: wrap; align-items: center; gap: 2rem; margin-top: 1.5rem;
          padding: 1.5rem 2rem; background: var(--surface); border: 1px solid var(--line); border-radius: 14px; }
.result svg { width: 250px; height: auto; flex-shrink: 0; }
.gauge-score { fill: var(--text); font-size: 30px; font-weight: 800; font-family: 'Manrope', system-ui, sans-serif; }
.gauge-end { fill: var(--muted); font-size: 9px; font-weight: 600; font-family: 'Manrope', system-ui, sans-serif; }
.result-label { color: var(--muted); font-size: .9rem; font-weight: 600; }
.badge { display: inline-block; margin-top: .5rem; padding: .3rem .85rem; border-radius: 999px;
         font-size: .9rem; font-weight: 700; }
.result-hint { margin-top: .6rem; color: var(--muted); font-size: .85rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def section(title: str):
    st.markdown(f'<div class="sec-title">{title}</div>', unsafe_allow_html=True)


def sub_section(title: str):
    st.markdown(f'<div class="sub-title">{title}</div>', unsafe_allow_html=True)


def risk_band(score: float):
    if score > 650:
        return "Low risk", "#38D9A9", "rgba(56,217,169,.15)", "Score above 650."
    if score > 500:
        return "Moderate risk", "#F4B860", "rgba(244,184,96,.15)", "Score between 501 and 650."
    return "High risk", "#F0697B", "rgba(240,105,123,.15)", "Score of 500 or below."


def result_card(score: float) -> str:
    label, color, tint, note = risk_band(score)
    frac = min(max((score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN), 0), 1)
    html = f"""
    <div class="result">
      <svg viewBox="0 0 200 124" role="img" aria-label="Predicted credit score {score:.1f} on a scale of {SCORE_MIN} to {SCORE_MAX}">
        <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="#25304A" stroke-width="14" stroke-linecap="round"/>
        <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="{color}" stroke-width="14" stroke-linecap="round" stroke-dasharray="{frac * ARC_LEN:.1f} {ARC_LEN:.1f}"/>
        <text x="100" y="94" text-anchor="middle" class="gauge-score">{score:.1f}</text>
        <text x="20" y="120" text-anchor="middle" class="gauge-end">{SCORE_MIN}</text>
        <text x="180" y="120" text-anchor="middle" class="gauge-end">{SCORE_MAX}</text>
      </svg>
      <div>
        <div class="result-label">Predicted credit score</div>
        <span class="badge" style="color:{color}; background:{tint};">{label}</span>
        <div class="result-hint">{note} The scale runs from {SCORE_MIN} to {SCORE_MAX}.</div>
      </div>
    </div>
    """
    return "".join(line.strip() for line in html.splitlines())


model = load_model()

# Header
st.markdown("""
<div class="hero">
  <span class="pill">Academic prototype</span>
  <h1 class="hero-title">Credit Score Predictor</h1>
  <p class="hero-sub">Estimate a customer's credit score from their profile, balances and spending, using an optimized Extra Trees Regressor pipeline.</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("Trained model pipeline not found in `outputs/`. Run `python main.py` in your terminal first.")
    st.stop()

with st.form("prediction_form"):
    left, right = st.columns(2, gap="large")

    with left:
        with st.container(key="card_profile"):
            section("Customer profile")
            c1, c2 = st.columns(2)
            age = c1.number_input("Age", min_value=18, max_value=100, value=32)
            gender = c2.selectbox("Gender", ["M", "F"])
            annual_income = c1.number_input("Annual income (₹)", min_value=0, value=850000)
            occupation = c2.selectbox("Occupation", ["Salaried", "Self-Employed", "Business", "Retired", "Student"])

        with st.container(key="card_account"):
            section("Card account")
            c1, c2 = st.columns(2)
            card_type = c1.selectbox("Card type", ["Silver", "Gold", "Platinum", "Basic"])
            credit_limit = c2.number_input("Credit limit (₹)", min_value=0, value=250000)
            card_age_months = c1.number_input("Card age (months)", min_value=0, value=36)

        with st.container(key="card_balances"):
            section("Balances and payments")
            c1, c2 = st.columns(2)
            outstanding_balance = c1.number_input("Outstanding balance (₹)", min_value=0.0, value=45000.0)
            statement_balance = c2.number_input("Statement balance (₹)", min_value=0.0, value=55000.0)
            payment_amount = c1.number_input("Payment amount (₹)", min_value=0.0, value=35000.0)
            cash_advance_amount = c2.number_input("Cash advance amount (₹)", min_value=0.0, value=0.0)

    with right:
        with st.container(key="card_spending"):
            section("Spending behavior")
            c1, c2 = st.columns(2)
            monthly_spending = c1.number_input("Total monthly spending (₹)", min_value=0.0, value=40000.0)
            monthly_transactions = c2.number_input("Monthly transactions", min_value=0, value=65)
            avg_transaction_value = c1.number_input("Avg transaction value (₹)", min_value=0.0, value=615.0)

            sub_section("By category")
            c1, c2 = st.columns(2)
            online_shopping = c1.number_input("Online shopping (₹)", min_value=0.0, value=12000.0)
            grocery = c2.number_input("Grocery (₹)", min_value=0.0, value=9000.0)
            fuel = c1.number_input("Fuel (₹)", min_value=0.0, value=4000.0)
            dining = c2.number_input("Dining (₹)", min_value=0.0, value=5000.0)
            travel = c1.number_input("Travel (₹)", min_value=0.0, value=3000.0)
            entertainment = c2.number_input("Entertainment (₹)", min_value=0.0, value=2000.0)
            utility = c1.number_input("Utility bills (₹)", min_value=0.0, value=5000.0)

        with st.container(key="card_activity"):
            section("Activity and rewards")
            m1, m2, m3 = st.columns(3)
            emi_count = m1.number_input("EMI count", min_value=0, value=1)
            intl_txns = m2.number_input("Intl. transactions", min_value=0, value=0)
            mobile_login = m3.number_input("App logins", min_value=0, value=18)
            r1, r2 = st.columns(2)
            rewards_earned = r1.number_input("Reward points earned", min_value=0, value=750)
            rewards_redeemed = r2.number_input("Reward points redeemed", min_value=0, value=200)

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Predict credit score", type="primary")

if submitted:
    input_data = pd.DataFrame([{
        "Age": age, "Gender": gender, "Annual_Income": annual_income,
        "Occupation": occupation, "Card_Type": card_type, "Credit_Limit": credit_limit,
        "Card_Age_Months": card_age_months, "Monthly_Spending": monthly_spending,
        "Monthly_Transactions": monthly_transactions, "Avg_Transaction_Value": avg_transaction_value,
        "Online_Shopping_Spending": online_shopping, "Grocery_Spending": grocery,
        "Fuel_Spending": fuel, "Dining_Spending": dining, "Travel_Spending": travel,
        "Entertainment_Spending": entertainment, "Utility_Bill_Spending": utility,
        "Outstanding_Balance": outstanding_balance, "Statement_Balance": statement_balance,
        "Payment_Amount": payment_amount, "Cash_Advance_Amount": cash_advance_amount,
        "EMI_Count": emi_count, "International_Transactions": intl_txns,
        "Reward_Points_Earned": rewards_earned, "Reward_Points_Redeemed": rewards_redeemed,
        "Mobile_App_Login": mobile_login
    }])

    prediction = float(model.predict(input_data)[0])
    st.markdown(result_card(prediction), unsafe_allow_html=True)

st.caption("This prediction comes from an academic prototype model trained on synthetic data.")