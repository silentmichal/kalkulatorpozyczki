import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Risk Intelligence", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Enterprise Blue ---
st.markdown("""
<style>
    :root {
        --primary: #0A2540;
        --secondary: #0066CC;
        --highlight: #E5F0FA;
        --text: #333333;
    }
    .stApp {
        background-color: #F8F9FA;
    }
    h1, h2, h3 {
        color: var(--primary) !important;
    }
    .stButton>button {
        background-color: var(--secondary);
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: var(--primary);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-top: 4px solid var(--secondary);
    }
</style>
""", unsafe_allow_html=True)

st.title("🏦 Credit Risk Intelligence")
st.markdown("**White-Box AI System do oceny ryzyka kredytowego klienta w pełni zoptymalizowany pod kątem transparentności.**")

# --- Sidebar ---
st.sidebar.header("📋 Parametry Aplikanta")

kwota = st.sidebar.number_input("Kwota pożyczki (PLN)", min_value=1000, max_value=1000000, value=50000, step=1000, help="Całkowita wnioskowana kwota.")
dochod = st.sidebar.number_input("Dochód miesięczny netto (PLN)", min_value=1000, max_value=100000, value=8000, step=500, help="Miesięczne zarobki po odliczeniu podatków.")
wiek = st.sidebar.slider("Wiek (lata)", min_value=18, max_value=80, value=35)
staz = st.sidebar.number_input("Staż pracy (lata)", min_value=0, max_value=50, value=5, help="Stabilność zatrudnienia.")
osoby = st.sidebar.number_input("Liczba osób na utrzymaniu", min_value=0, max_value=10, value=1)

dti_tooltip = "Debt-to-Income (DTI) - Wskaźnik określający stosunek zobowiązań kredytowych do dochodów. Kluczowy parametr oceny ryzyka."
st.sidebar.info(f"💡 **Słowniczek:**\n\n{dti_tooltip}")

# --- Data Generation & Modeling ---
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 1000
    age = np.random.randint(20, 70, n)
    income = np.random.randint(3000, 30000, n)
    employment = np.random.randint(0, 40, n)
    dependents = np.random.randint(0, 5, n)
    loan_amount = np.random.randint(5000, 200000, n)
    
    # Syntetyczna logika ryzyka
    dti = loan_amount / (income * 12 + 1)
    
    # Wzór logit - wyższe wartości z to większe ryzyko defaultu (1)
    # Starsi, wyższy dochód, dłuższy staż = mniejsze ryzyko
    # Wyższe DTI, więcej osób na utrzymaniu = większe ryzyko
    z = -1.5 - 0.03 * age - 0.15 * employment + 0.4 * dependents + 5 * dti
    p = 1 / (1 + np.exp(-z))
    
    default = np.random.binomial(1, p)
    
    df = pd.DataFrame({
        'Wiek': age,
        'Dochod': income,
        'Staz': employment,
        'Utrzymanie': dependents,
        'Kwota': loan_amount,
        'Default': default
    })
    return df

if st.button("🚀 Load Benchmark Data & Analyze", use_container_width=True):
    with st.spinner("Trenowanie modeli AI (White-box)..."):
        df = load_data()
        X = df.drop(columns=['Default'])
        y = df['Default']
        
        # Skalowanie (niezbędne dla stabilności regresji logistycznej)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Logistic Regression dla głównego silnika predykcji (Vercel memory friendly)
        lr_model = LogisticRegression()
        lr_model.fit(X_scaled, y)
        
        # Decision Tree dla edukacyjnej wizualizacji drzewa logicznego
        dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
        dt_model.fit(X, y)
        
        # Dane bieżącego klienta z Sidebar
        input_data = pd.DataFrame({
            'Wiek': [wiek],
            'Dochod': [dochod],
            'Staz': [staz],
            'Utrzymanie': [osoby],
            'Kwota': [kwota]
        })
        
        input_scaled = scaler.transform(input_data)
        
        # Predykcja
        prob_default = lr_model.predict_proba(input_scaled)[0, 1]
        risk_percentage = round(prob_default * 100, 2)
        
        # Logika biznesowa - Decyzja
        if risk_percentage < 30:
            status, color, msg = "Zatwierdzono", "green", "Niskie ryzyko kredytowe. Brak przeciwwskazań do udzielenia finansowania."
        elif risk_percentage < 70:
            status, color, msg = "Wymagana dodatkowa weryfikacja", "orange", "Umiarkowane ryzyko. Rekomendowana manualna analiza zdolności przez analityka."
        else:
            status, color, msg = "Odrzucono", "red", "Wysokie ryzyko niewypłacalności (Default Risk). Aplikacja kategorycznie odrzucona."
        
        # ==========================================
        # SEKCJA 1: WYNIK RYZYKA (GAUGE CHART)
        # ==========================================
        st.markdown("---")
        st.subheader("📊 Wynik Oceny Ryzyka")
        
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="border-top-color: {color}; height: 100%;">
                <h3 style="margin-top:0; color:{color} !important;">Status decyzyjny:<br>{status}</h3>
                <p style="font-size: 1.1em; color: gray; margin-top:20px;">{msg}</p>
                <hr>
                <p style="font-size: 1.25em;"><b>Prawdopodobieństwo Defaultu:</b> {risk_percentage}%</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_percentage,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Poziom Ryzyka (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "rgba(0,0,0,0.2)"},
                    'steps': [
                        {'range': [0, 30], 'color': "#d4edda"},   # light green
                        {'range': [30, 70], 'color': "#fff3cd"},  # light orange
                        {'range': [70, 100], 'color': "#f8d7da"}  # light red
                    ],
                    'threshold': {
                        'line': {'color': color, 'width': 6},
                        'thickness': 0.8,
                        'value': risk_percentage
                    }
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        # ==========================================
        # SEKCJA 2: TRANSPARENTNOŚĆ (SHAP VALUES)
        # ==========================================
        st.markdown("---")
        st.subheader("🔍 Dlaczego podjęto taką decyzję? (Analiza SHAP)")
        st.write("Wykres pokazuje tzw. Wartości SHAP (SHapley Additive exPlanations) - obrazuje jak dany parametr wpłynął na OSTATECZNE RYZYKO w porównaniu do średniej wartości w portfelu. Wartości dodatnie podnoszą ryzyko (źle), ujemne obniżają (dobrze).")
        
        # Wyliczanie SHAP tylko dla 1 konkretnego przypadku (optymalizacja limitu pamięci Vercel)
        explainer = shap.LinearExplainer(lr_model, X_scaled)
        shap_values = explainer.shap_values(input_scaled)
        
        try:
            # SHAP 0.40+ style waterall
            expl = shap.Explanation(values=shap_values[0], 
                                    base_values=explainer.expected_value, 
                                    data=input_data.iloc[0], 
                                    feature_names=input_data.columns)
            fig_shap, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(expl, show=False)
            plt.tight_layout()
            st.pyplot(fig_shap)
        except Exception as e:
            # Fallback for older SHAP syntax/versions or rendering issues
            fig_shap, ax = plt.subplots(figsize=(10, 5))
            shap.summary_plot(shap_values, input_scaled, feature_names=input_data.columns, plot_type="bar", show=False)
            plt.tight_layout()
            st.pyplot(fig_shap)
        
        # ==========================================
        # SEKCJA 3: DRZEWO DECYZYJNE
        # ==========================================
        st.markdown("---")
        st.subheader("🌳 Wizualizacja Logiki (Uproszczone Drzewo Decyzyjne)")
        
        with st.expander("Rozwiń graf drzewa 'Jeśli-To' dla ujęcia regułowego", expanded=False):
            st.write("Drzewo ukazuje uproszczone granice decyzyjne na bazie zebranych danych analitycznych (0 = Spłaci, 1 = Default). Modele oparte o Logistic Regression dzielą przestrzeń inaczej, ale drzewo to doskonałe uzupełnienie edukacyjne (White-box).")
            fig_tree, ax_tree = plt.subplots(figsize=(14, 6), dpi=150)
            plot_tree(dt_model, feature_names=X.columns, class_names=['Spłaci', 'Default'], 
                      filled=True, rounded=True, ax=ax_tree, fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_tree)

else:
    st.info("Wprowadź docelowe parametry kandydata w lewym panelu bocznym i wywołaj **'Load Benchmark Data & Analyze'**, aby uzyskać dedykowaną ocenę ryzyka (Credit Risk Score).")
