import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Street Food Body Damage Predictor",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #ff6b35 0%, #f7931e 50%, #ff4757 100%);
        padding: 2rem 1.5rem;
        border-radius: 18px;
        text-align: center;
        color: white !important;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px rgba(255, 107, 53, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.3rem;
        font-weight: 700;
        margin: 0;
        color: white !important;
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        color: white !important;
    }
    
    .result-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid #eee;
        margin-top: 1rem;
    }
    
    .damage-value {
        font-size: 4.5rem;
        font-weight: 700;
        line-height: 1.1;
        margin: 0.3rem 0;
    }
    
    .risk-low { color: #2ed573; }
    .risk-medium { color: #ffa502; }
    .risk-high { color: #ff6b35; }
    .risk-extreme { color: #ff4757; }
    
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.05rem;
        margin-top: 0.5rem;
    }
    
    .badge-low { background: #d4edda; color: #155724; }
    .badge-medium { background: #fff3cd; color: #856404; }
    .badge-high { background: #ffe8d6; color: #9a3412; }
    .badge-extreme { background: #f8d7da; color: #721c24; }
    
    .contrib-bar {
        height: 12px;
        border-radius: 8px;
        background: #e9ecef;
        overflow: hidden;
        margin: 0.3rem 0 0.7rem 0;
    }
    
    .contrib-fill {
        height: 100%;
        border-radius: 8px;
    }
    
    .info-box {
        background: #fff8f0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border-left: 5px solid #ff6b35;
        margin-top: 1rem;
        font-size: 0.95rem;
        color: #333 !important;
    }
    
    .placeholder-card {
        background: #f8f9fa;
        border-radius: 18px;
        padding: 2.5rem 2rem;
        text-align: center;
        border: 2px dashed #ced4da;
        color: #6c757d;
        margin-top: 1rem;
    }
    
    div.stButton > button {
        background: linear-gradient(135deg, #ff6b35 0%, #ff4757 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.9rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
        box-shadow: 0 8px 25px rgba(255, 107, 53, 0.4) !important;
        width: 100% !important;
    }
    

    .body-visual-wrap {
        background: #fff;
        border-radius: 20px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        border: 1px solid #eee;
        margin-top: 1rem;
    }
    .body-visual-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.3rem;
    }
    .body-visual-note {
        color: #666;
        font-size: 0.85rem;
        margin-top: 0.4rem;
    }
    .body-percent {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ff4757;
        margin-top: 0.3rem;
    }

    .footer {
        text-align: center;
        color: #999;
        font-size: 0.85rem;
        margin-top: 2.5rem;
        padding: 1rem;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ====================== MODEL ======================
FEATURE_NAMES = [
    "Momo", "Pizza", "Samosa", "Pani_Puri", "French_Fries",
    "Soya_Bari", "Burger", "Patties", "Chowmein", "Dosa"
]

COEFFICIENTS = np.array([
    0.7360, 0.7149, 0.7553, 0.4749, 0.5390,
    0.5805, 0.6035, 0.6737, 0.6944, 0.5600
])

INTERCEPT = -0.0852

def predict_damage(inputs):
    x = np.array([inputs[f] for f in FEATURE_NAMES])
    return float(np.clip(INTERCEPT + np.dot(COEFFICIENTS, x), 0, 100))


def render_body_damage(prediction):
    # Fill the human silhouette from the bottom upward according to the score.
    fill_height = 220 * (prediction / 100.0)
    fill_y = 240 - fill_height

    body_svg = f"""
    <div class="body-visual-wrap">
      <div class="body-visual-title">🧍 Body Risk Visualizer</div>
      <svg viewBox="0 0 120 250" width="190" role="img"
           aria-label="Illustrative body visualization: {prediction:.1f} percent">
        <defs>
          <clipPath id="humanShape">
            <circle cx="60" cy="27" r="18"/>
            <path d="M43 52
                     Q60 45 77 52
                     L84 103
                     L77 108
                     L72 76
                     L73 132
                     L86 205
                     L72 208
                     L59 151
                     L47 208
                     L33 205
                     L47 132
                     L48 76
                     L43 108
                     L36 103 Z"/>
        </clipPath>
      </defs>

      <g clip-path="url(#humanShape)">
        <rect x="20" y="5" width="80" height="235" fill="#e7ebf0"/>
        <rect x="20" y="{fill_y:.2f}" width="80" height="{fill_height:.2f}"
              fill="#ff3548"/>
      </g>

      <circle cx="60" cy="27" r="18" fill="none"
              stroke="#7b8794" stroke-width="1.5"/>
      <path d="M43 52 Q60 45 77 52 L84 103 L77 108 L72 76
               L73 132 L86 205 L72 208 L59 151 L47 208 L33 205
               L47 132 L48 76 L43 108 L36 103 Z"
            fill="none" stroke="#7b8794" stroke-width="1.5"
            stroke-linejoin="round"/>
      <line x1="15" y1="240" x2="105" y2="240"
            stroke="#d5dbe2" stroke-width="2"/>
    </svg>
      <div class="body-percent">{prediction:.1f}%</div>
      <div class="body-visual-note">
        Red fill visualizes the model score only — it is not a medical measurement.
      </div>
    </div>
    """
    st.markdown(body_svg, unsafe_allow_html=True)

FOOD_INFO = {
    "Momo":         {"emoji": "🥟", "tip": "Steam better than fried"},
    "Pizza":        {"emoji": "🍕", "tip": "High cheese & refined flour"},
    "Samosa":       {"emoji": "🔺", "tip": "Deep fried + maida"},
    "Pani_Puri":    {"emoji": "🥣", "tip": "Water quality risk + oil"},
    "French_Fries": {"emoji": "🍟", "tip": "High trans-fat risk"},
    "Soya_Bari":    {"emoji": "🫘", "tip": "Often deep fried"},
    "Burger":       {"emoji": "🍔", "tip": "Processed meat + bun"},
    "Patties":      {"emoji": "🥐", "tip": "Maida + oil heavy"},
    "Chowmein":     {"emoji": "🍜", "tip": "MSG + refined oil"},
    "Dosa":         {"emoji": "🥞", "tip": "Relatively better if not oily"},
}

# ====================== HEADER ======================
st.markdown("""
<div class="main-header">
    <h1>🍔 Indian Street Food Body Damage Predictor</h1>
    <p>Tell how many times you ate these in the last <b>30 days</b> → Get your yearly body damage score</p>
</div>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.markdown("### ⚙️ How to use")
    st.markdown("""
    1. Set frequency for each food (**last 30 days**)  
    2. Scroll down and click **Calculate**  
    3. See your damage score + risk level  
    4. Check which food is hurting you the most
    """)
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.caption("• Algorithm: Linear Regression only")
    st.caption("• R² Score: ~0.96")
    st.caption("• Trained on 50,000 samples")
    st.markdown("---")
    st.markdown("💡 **Tip:** Values between 0–30 recommended")

# ====================== INPUTS ======================
st.markdown("### 🧾 Consumption in Last 30 Days")
st.caption("How many times did you eat each item in the last 30 days?")

col_a, col_b = st.columns(2)
inputs = {}

with col_a:
    for food in FEATURE_NAMES[:5]:
        info = FOOD_INFO[food]
        st.markdown(
            f"<h4 style='color:white; margin-bottom:2px;'>{info['emoji']} {food.replace('_', ' ')}</h4>"
            f"<p style='color:#444444; font-size:0.85rem; margin-top:0;'>{info['tip']}</p>",
            unsafe_allow_html=True
        )
        inputs[food] = st.slider(
            label=food,
            min_value=0,
            max_value=30,
            value=0,
            key=food,
            label_visibility="collapsed"
        )

with col_b:
    for food in FEATURE_NAMES[5:]:
        info = FOOD_INFO[food]
        st.markdown(
            f"<h4 style='color:white; margin-bottom:2px;'>{info['emoji']} {food.replace('_', ' ')}</h4>"
            f"<p style='color:#444444; font-size:0.85rem; margin-top:0;'>{info['tip']}</p>",
            unsafe_allow_html=True
        )
        inputs[food] = st.slider(
            label=food,
            min_value=0,
            max_value=30,
            value=0,
            key=food,
            label_visibility="collapsed"
        )

# ====================== BUTTON ======================
st.markdown("---")
calculate = st.button("🔥 Calculate Body Damage", type="primary", width="stretch")

# ====================== RESULT ======================
if calculate:
    prediction = predict_damage(inputs)

    if prediction < 30:
        risk_class, badge_class = "risk-low", "badge-low"
        risk_text = "LOW RISK 🟢"
        risk_msg = "Your body is in good shape. Keep it controlled!"
    elif prediction < 50:
        risk_class, badge_class = "risk-medium", "badge-medium"
        risk_text = "MEDIUM RISK 🟡"
        risk_msg = "Moderate damage. Try reducing fried items."
    elif prediction < 70:
        risk_class, badge_class = "risk-high", "badge-high"
        risk_text = "HIGH RISK 🟠"
        risk_msg = "Alert! Start cutting down street food now."
    else:
        risk_class, badge_class = "risk-extreme", "badge-extreme"
        risk_text = "EXTREME RISK 🔴"
        risk_msg = "Danger zone! Serious lifestyle change needed."

    result_col, body_col = st.columns([1.25, 1])

    with result_col:
        st.markdown(f"""
        <div class="result-card">
            <div style="font-size:1.1rem; color:#555;">Model Score (Illustrative)</div>
            <div class="damage-value {risk_class}">{prediction:.1f}%</div>
            <div class="risk-badge {badge_class}">{risk_text}</div>
            <p style="margin-top:1rem; color:#444;">{risk_msg}</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(min(prediction / 100.0, 1.0))

    with body_col:
        render_body_damage(prediction)

    st.markdown("#### 🔍 Food Contribution to Damage")
    st.caption("How much each food is contributing")

    contributions = []
    for food, coef in zip(FEATURE_NAMES, COEFFICIENTS):
        contrib = max(0.0, coef * inputs[food])
        contributions.append((food, contrib, FOOD_INFO[food]["emoji"]))

    contributions.sort(key=lambda x: x[1], reverse=True)
    max_contrib = max((c[1] for c in contributions), default=1)

    for food, contrib, emoji in contributions:
        if contrib > 0.05:
            pct = (contrib / max_contrib) * 100 if max_contrib > 0 else 0
            color = "#ff4757" if pct > 70 else "#ffa502" if pct > 40 else "#2ed573"
            st.markdown(f"""
            <div style="margin-bottom:0.4rem;">
                <div style="display:flex; justify-content:space-between; font-size:0.95rem; color:#111;">
                    <span><b>{emoji} {food.replace('_', ' ')}</b></span>
                    <span style="font-weight:600;">{contrib:.1f}</span>
                </div>
                <div class="contrib-bar">
                    <div class="contrib-fill" style="width:{pct}%; background:{color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    if contributions and contributions[0][1] > 1:
        top_food = contributions[0][0].replace("_", " ")
        st.markdown(f"""
        <div class="info-box">
            💡 <b>Biggest Culprit:</b> {top_food}<br>
            Reducing this will drop your damage score the most.
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="placeholder-card">
        <div style="font-size:2.8rem;">🧮</div>
        <h3 style="margin:0.5rem 0; color:#495057;">Ready to Calculate</h3>
        <p style="margin:0; color:#6c757d;">
            Set your 30-day consumption above and click the <b>Calculate</b> button
        </p>
    </div>
    """, unsafe_allow_html=True)

# ====================== MODEL DETAILS ======================
st.markdown("---")
with st.expander("ℹ️ Model Details (Linear Regression only)"):
    st.markdown("""
    **Algorithm:** Multiple Linear Regression (no advanced models)
    
    **Formula:**
    """)
coef_df = pd.DataFrame({
    "Food": [f.replace("_", " ") for f in FEATURE_NAMES],
    "Coefficient": [round(c, 4) for c in COEFFICIENTS]
}).sort_values("Coefficient", ascending=False)
st.dataframe(coef_df, width="stretch", hide_index=True)
st.markdown(f"""
- **Intercept:** `{INTERCEPT}`
- Higher coefficient = more damage per serving
- **Samosa** has the highest impact
- Model R² ≈ **0.96**
""")

st.markdown("""
<div class="footer">
Made with ❤️ using Streamlit + Linear Regression | For educational purposes only<br>
Always consult a doctor for real health decisions
</div>
""", unsafe_allow_html=True)