import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(
    page_title="Credit Risk Intelligence Pro",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced Enterprise Aesthetics ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    :root {
        --primary: #0A2540;
        --secondary: #0066CC;
        --accent: #00D1FF;
        --bg-light: #F4F7FA;
    }
    
    .stApp {
        background: linear-gradient(135deg, #F8F9FB 0%, #E9EFF5 100%);
    }
    
    /* Premium Sidebar */
    [data-testid="stSidebar"] {
        background-color: white;
        border-right: 1px solid #E0E4E8;
    }
    
    /* Cards & Containers */
    .metric-container {
        background: white;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border: 1px solid #EFF2F5;
        margin-bottom: 20px;
    }
    
    .status-badge {
        padding: 8px 16px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        margin-bottom: 12px;
    }
    
    .badge-green { background: #E6F9F0; color: #10B981; }
    .badge-orange { background: #FFF7ED; color: #F59E0B; }
    .badge-red { background: #FEF2F2; color: #EF4444; }
    
    /* Typography */
    h1 {
        font-weight: 700;
        letter-spacing: -0.5px;
        color: var(--primary);
    }
    
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--primary);
        margin-bottom: 1rem;
        border-left: 4px solid var(--secondary);
        padding-left: 12px;
    }

    .info-label {
        color: #64748B;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    
    .big-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--primary);
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
col_head1, col_head2 = st.columns([2, 1])
with col_head1:
    st.title("🏦 Credit Risk Intelligence Pro")
    st.markdown("<p style='color:#64748B; font-size:1.1rem;'>System klasy Enterprise do zaawansowanej oceny wiarygodności kredytowej i wyjaśnialności decyzji AI.</p>", unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.header("📋 Dane Wnioskodawcy")

with st.sidebar:
    kwota = st.number_input("Kwota pożyczki (PLN)", 1000, 1000000, 50000, 5000)
    oprocentowanie = st.slider("Oprocentowanie (%)", 1.0, 25.0, 8.5, 0.5, help="Roczne oprocentowanie nominalne.")
    okres = st.slider("Okres spłaty (miesiace)", 6, 120, 36)
    st.markdown("---")
    dochod = st.number_input("Dochód netto (PLN/msc)", 1000, 100000, 8000, 500)
    wiek = st.slider("Wiek (lata)", 18, 80, 35)
    staz = st.number_input("Staż pracy (lata)", 0, 50, 5)
    osoby = st.number_input("Osoby na utrzymaniu", 0, 10, 1)

    st.info("💡 **DTI (Debt-to-Income):** Wskaźnik ten rośnie wraz z kwotą i oprocentowaniem, co bezpośrednio wpływa na ryzyko niewypłacalności.")

# --- Calculation Engine ---
rata_miesieczna = (kwota * (oprocentowanie/100/12)) / (1 - (1 + oprocentowanie/100/12)**(-okres))
calkowity_koszt = rata_miesieczna * okres - kwota

# --- Synthetic Data with Interest & Duration ---
@st.cache_data
def load_enhanced_data():
    np.random.seed(42)
    n = 1500
    age = np.random.randint(20, 70, n)
    income = np.random.randint(3000, 30000, n)
    employment = np.random.randint(0, 40, n)
    dependents = np.random.randint(0, 5, n)
    loan_amount = np.random.randint(5000, 300000, n)
    interest = np.random.uniform(3, 20, n)
    
    # Calculate synthetic installment
    r = interest / 100 / 12
    inst = (loan_amount * r) / (1 - (1 + r)**(-36)) # Assuming constant 36m for base training simplicity
    
    dti = inst / (income + 1)
    
    # Risk Logic (Higher interest = Higher risk)
    z = -2.0 - 0.04 * age - 0.12 * employment + 0.5 * dependents + 8.0 * dti + 0.05 * interest
    p = 1 / (1 + np.exp(-z))
    default = np.random.binomial(1, p)
    
    return pd.DataFrame({
        'Wiek': age,
        'Dochod': income,
        'Staz': employment,
        'Utrzymanie': dependents,
        'Kwota': loan_amount,
        'Oprocentowanie': interest,
        'Default': default
    })

if st.button("🚀 Przeprowadź Analizę Ryzyka", use_container_width=True):
    with st.spinner("Przetwarzanie danych i analiza portfela..."):
        df = load_enhanced_data()
        X = df.drop(columns=['Default'])
        y = df['Default']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        lr_model = LogisticRegression(class_weight='balanced') # Balanced for better sensitivity
        lr_model.fit(X_scaled, y)
        
        dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
        dt_model.fit(X, y)
        
        # User input for prediction
        user_input = pd.DataFrame({
            'Wiek': [wiek],
            'Dochod': [dochod],
            'Staz': [staz],
            'Utrzymanie': [osoby],
            'Kwota': [kwota],
            'Oprocentowanie': [oprocentowanie]
        })
        user_scaled = scaler.transform(user_input)
        
        prob_default = lr_model.predict_proba(user_scaled)[0, 1]
        risk = round(prob_default * 100, 2)
        
        # Decision Logic
        if risk < 35:
            status, badge, color, msg = "AKCEPTACJA", "badge-green", "#10B981", "Wysoka zdolność kredytowa. Klient spełnia kryteria bezpiecznego finansowania."
        elif risk < 65:
            status, badge, color, msg = "WERYFIKACJA", "badge-orange", "#F59E0B", "Ryzyko umiarkowane. Zalecane zabezpieczenie lub dodatkowe dokumenty dochodowe."
        else:
            status, badge, color, msg = "ODMOWA", "badge-red", "#EF4444", "Zbyt wysokie ryzyko niewypłacalności. Wniosek odrzucony systemowo."
            
        # --- UI LAYOUT: RESULTS ---
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1.2])
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="info-label">Rata Miesięczna</div>
                <div class="big-value">{rata_miesieczna:,.2f} PLN</div>
                <p style='color:#64748B; font-size:0.8rem; margin-top:10px;'>Całkowity koszt odsetek: {calkowity_koszt:,.2f} PLN</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-container">
                <div class="info-label">Wskaźnik DTI</div>
                <div class="big-value">{ (rata_miesieczna/dochod)*100 :.1f}%</div>
                <p style='color:#64748B; font-size:0.8rem; margin-top:10px;'>Obciążenie dochodu ratą.</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-container" style="border-top: 5px solid {color};">
                <span class="status-badge {badge}">{status}</span>
                <div class="info-label">Rekomendacja Systemu</div>
                <p style="font-size: 0.95rem; margin-bottom:0;">{msg}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-container">
                <div class="info-label">Prawdopodobieństwo PD</div>
                <div class="big-value" style="color:{color};">{risk}%</div>
                <p style='color:#64748B; font-size:0.8rem; margin-top:10px;'>Prawdopodobieństwo Niewypłacalności.</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = risk,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Credit Risk Score", 'font': {'size': 20, 'color': '#0A2540'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#E0E4E8",
                    'steps': [
                        {'range': [0, 35], 'color': '#F0FDF4'},
                        {'range': [35, 65], 'color': '#FFFBEB'},
                        {'range': [65, 100], 'color': '#FEF2F2'}],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': risk}}
            ))
            fig.update_layout(paper_bgcolor = "rgba(0,0,0,0)", font = {'color': "darkblue", 'family': "Inter"}, height=350, margin=dict(t=50, b=0))
            st.plotly_chart(fig, use_container_width=True)

        # --- SHAP ANALYSIS ---
        st.markdown("<p class='section-title'>🔍 Explainable AI (XAI): Analiza Wpływu Cech</p>", unsafe_allow_html=True)
        st.write("Składowe decyzji: Wartości dodatnie (czerwone) zwiększają ryzyko, ujemne (niebieskie) działają na korzyść klienta.")
        
        explainer = shap.LinearExplainer(lr_model, X_scaled)
        shap_values = explainer.shap_values(user_scaled)
        
        try:
            explanation = shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=user_input.iloc[0],
                feature_names=user_input.columns
            )
            fig_shap, ax = plt.subplots(figsize=(12, 5))
            shap.plots.waterfall(explanation, show=False)
            plt.title("Wpływ parametrów na wynik ryzyka (Waterfall Plot)", fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            st.pyplot(fig_shap)
        except:
            st.error("Błąd generowania wykresu waterfall. Wyświetlanie alternatywne.")
        
        # --- DECISION TREE ---
        with st.expander("🌳 Przejrzyj Logikę Drzewa Decyzyjnego (Heurystyka)"):
            st.write("Poniższe drzewo wizualizuje uproszczone reguły klasyfikacji ('jeśli-to') wydzielone z danych benchmarkowych.")
            fig_tree, ax_tree = plt.subplots(figsize=(16, 7), dpi=150)
            plot_tree(dt_model, feature_names=X.columns, class_names=['Niskie Ryzyko', 'Wysokie Ryzyko'], 
                      filled=True, rounded=True, ax=ax_tree, fontsize=8, precision=1)
            plt.tight_layout()
            st.pyplot(fig_tree)

else:
    # Landing message if no analysis yet
    st.markdown("""
    <div style="background: white; padding: 40px; border-radius: 20px; text-align: center; border: 1px dashed #CBD5E1; margin-top:40px;">
        <h2 style="color:#64748B; margin-bottom:10px;">Gotowy do analizy?</h2>
        <p style="color:#94A3B8;">Uzupełnij dane finansowe i osobiste w panelu bocznym,<br>a następnie kliknij przycisk powyżej, aby uruchomić silnik Intelligence Pro.</p>
    </div>
    """, unsafe_allow_html=True)
