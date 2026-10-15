import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="ChurnPredict", page_icon="📉", layout="wide")

API_URL = os.getenv("API_URL", "http://api:8000")
APP_TITLE = "ChurnPredict"

# ---------------------------------------------------------------- Estilo
PRIMARY = "#2563EB"  # color principal
ACCENT = "#1E3A8A"  # acento (barra destacada)
BLUE_SOFT = "#DBEAFE"
GREEN = "#16A34A"
AMBER = "#F59E0B"
RED = "#E11D48"
TEXT = "#1F2937"
MUTED = "#8A8FA3"
GRID = "#EEF0F6"
TRACK = "#E5ECF6"

RISK_LEVELS = [  # (etiqueta, color, desde, hasta)
    ("Bajo", GREEN, 0.0, 0.30),
    ("Medio", AMBER, 0.30, 0.60),
    ("Alto", RED, 0.60, 1.01),
]

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stApp {{ font-family: 'Poppins', 'Inter', sans-serif; color: {TEXT}; }}
    .stApp {{ background: #F5F8FC; }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    #MainMenu, footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 2rem; max-width: 1240px; }}

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {{ background: #EEF4FC; border-right: 1px solid #DFE8F5; }}
    section[data-testid="stSidebar"] * {{ color: {TEXT}; }}
    section[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{ display: none; }}
    section[data-testid="stSidebar"] [role="radiogroup"] label {{
        padding: .6rem 1rem; border-radius: 12px; margin-bottom: 4px; width: 100%;
        transition: background .15s ease;
    }}
    section[data-testid="stSidebar"] [role="radiogroup"] label p {{ font-weight: 500; opacity: .85; }}
    section[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: #E1EBF9; }}
    section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
        background: #FFFFFF; box-shadow: inset 3px 0 0 {PRIMARY}, 0 2px 8px rgba(37, 99, 235, .08);
    }}
    section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{
        opacity: 1; font-weight: 600; color: {PRIMARY};
    }}
    .brand {{ display: flex; align-items: center; gap: 10px; font-weight: 700;
              font-size: 1.1rem; padding: .25rem .5rem 1.6rem; }}
    .profile {{ text-align: center; padding-bottom: 1.6rem; }}
    .brand svg {{ color: {PRIMARY}; }}
    .avatar {{ width: 86px; height: 86px; border-radius: 50%; background: {BLUE_SOFT};
               color: {PRIMARY} !important; border: 3px solid #FFFFFF; display: flex; align-items: center; justify-content: center;
               margin: 0 auto .7rem; font-size: 1.8rem; font-weight: 600;
               box-shadow: 0 6px 18px rgba(37, 99, 235, .15); }}
    .profile small {{ color: {MUTED} !important; }}
    .nav-section {{ font-size: .68rem; font-weight: 600; letter-spacing: .12em; color: {MUTED} !important;
                    text-transform: uppercase; padding: 0 1rem .5rem; }}

    /* ---- Tarjetas ---- */
    [class*="st-key-card"] {{
        background: #FFFFFF; border: none !important; border-radius: 22px;
        box-shadow: 0 10px 30px rgba(30, 64, 175, .06), 0 1px 3px rgba(16, 24, 40, .04);
        padding: 1.3rem 1.5rem;
    }}
    .page-title {{ font-size: 1.6rem; font-weight: 700; margin: 0; }}
    .page-sub {{ color: {MUTED}; font-size: .92rem; margin: .1rem 0 1.4rem; }}
    .card-title {{ font-size: 1rem; font-weight: 600; margin-bottom: .7rem; color: {TEXT}; }}

    [class*="st-key-card_k"] {{ height: 108px; justify-content: center; }}
    .kpi {{ display: flex; align-items: center; gap: 16px; }}
    .kpi > div:last-child {{ min-width: 0; }}
    .kpi-icon {{ width: 52px; height: 52px; border-radius: 16px; display: flex; flex-shrink: 0;
                 align-items: center; justify-content: center; color: #FFFFFF !important; }}
    .kpi-value {{ font-size: 1.55rem; font-weight: 700; line-height: 1.15; }}
    .kpi-label {{ color: {MUTED}; font-size: .8rem; }}

    .rings {{ display: flex; justify-content: space-around; flex-wrap: wrap; gap: 12px; padding: .4rem 0; }}
    .ring-wrap {{ text-align: center; }}
    .ring-label {{ font-size: .82rem; color: {MUTED}; margin-top: .3rem; }}

    .legend-row {{ display: flex; justify-content: space-between; align-items: center;
                   padding: .35rem 0; font-size: .85rem; }}
    .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%;
            border: 2.5px solid; margin-right: 10px; }}
    .pill {{ display: inline-block; padding: .25rem .8rem; border-radius: 999px;
             font-size: .8rem; font-weight: 600; }}
    .reason {{ padding: .6rem 0; border-bottom: 1px solid {GRID}; font-size: .9rem; }}
    .reason:last-child {{ border-bottom: none; }}
    .reason small {{ color: {MUTED}; }}

    .stButton > button, .stFormSubmitButton > button {{
        border-radius: 999px; padding: .45rem 1.4rem; font-weight: 500; }}
    .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primaryFormSubmit"] {{
        background: {PRIMARY}; border: none; color: #FFFFFF;
        box-shadow: 0 4px 12px rgba(37, 99, 235, .2);
    }}
    .stTextInput input {{ border-radius: 12px; }}

    /* ---- Formulario ---- */
    .form-section {{ display: flex; gap: 14px; align-items: flex-start; margin: 1.2rem 0 .4rem;
                     padding-top: 1.3rem; border-top: 1px solid {GRID}; }}
    .form-section.first {{ border-top: none; margin-top: 0; padding-top: 0; }}
    .step {{ width: 32px; height: 32px; border-radius: 50%; background: {BLUE_SOFT}; color: {PRIMARY};
             font-weight: 600; display: flex; align-items: center; justify-content: center;
             flex-shrink: 0; font-size: .9rem; }}
    .form-title {{ font-weight: 600; font-size: 1rem; line-height: 1.3; }}
    .form-sub {{ color: {MUTED}; font-size: .82rem; }}
    [class*="st-key-card_form"] [data-testid="stWidgetLabel"] p {{ font-weight: 500; font-size: .86rem; }}
    [class*="st-key-card_form"] div[data-baseweb="select"] > div,
    [class*="st-key-card_form"] [data-testid="stNumberInputContainer"] {{ border-radius: 10px; }}
    [data-testid="stSegmentedControl"] button[aria-checked="true"] {{
        background: {BLUE_SOFT}; color: {PRIMARY}; border-color: {BLUE_SOFT}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

ICONS = {
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>'
             '<path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "alert": '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>'
             '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "percent": '<line x1="19" y1="5" x2="5" y2="19"/><circle cx="6.5" cy="6.5" r="2.5"/>'
               '<circle cx="17.5" cy="17.5" r="2.5"/>',
    "logo": '<polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/>',
    "money": '<line x1="12" y1="1" x2="12" y2="23"/>'
             '<path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
}


def icon(name: str, size: int = 22) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')


def page_header(title: str, subtitle: str):
    st.markdown(f'<div class="page-title">{title}</div><div class="page-sub">{subtitle}</div>',
                unsafe_allow_html=True)


def kpi_card(key: str, icon_name: str, value: str, label: str, color: str = PRIMARY):
    with st.container(key=f"card_{key}"):
        st.markdown(
            f'<div class="kpi"><div class="kpi-icon" style="background:linear-gradient(135deg,{color},{color}B3);'
            f'box-shadow:0 6px 14px {color}40">'
            f'{icon(icon_name)}</div><div><div class="kpi-value">{value}</div>'
            f'<div class="kpi-label">{label}</div></div></div>',
            unsafe_allow_html=True,
        )


def risk_level(prob: float):
    for label, color, lo, hi in RISK_LEVELS:
        if lo <= prob < hi:
            return label, color
    return RISK_LEVELS[-1][:2]


def ring(pct: float, color: str, label: str, size: int = 120, stroke: int = 9) -> str:
    """Anillo de progreso SVG (estilo medidor circular)."""
    r = (size - stroke) / 2
    circ = 2 * 3.14159 * r
    filled = max(0.0, min(pct, 100.0)) / 100 * circ
    c = size / 2
    return (
        f'<div class="ring-wrap"><svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" '
        f'role="img" aria-label="{label}: {pct:.0f}%">'
        f'<circle cx="{c}" cy="{c}" r="{r}" fill="none" stroke="{TRACK}" stroke-width="{stroke}" '
        f'stroke-dasharray="4 5"/>'
        f'<circle cx="{c}" cy="{c}" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-dasharray="{filled:.1f} {circ:.1f}" '
        f'transform="rotate(-90 {c} {c})"/>'
        f'<text x="50%" y="50%" text-anchor="middle" dominant-baseline="central" '
        f'font-size="{size * 0.2:.0f}" font-weight="700" fill="{TEXT}">{pct:.0f}%</text></svg>'
        f'<div class="ring-label">{label}</div></div>'
    )


def plot(fig: go.Figure, height: int):
    if fig.layout.margin.r is None:
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    fig.update_layout(
        height=height, showlegend=False,
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=GRID, font=dict(color=TEXT)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Poppins, sans-serif", color=TEXT, size=12),
    )
    st.plotly_chart(fig, config={"displayModeBar": False})


# ------------------------------------------------------------------- API
@st.cache_data(ttl=300, show_spinner=False)
def load_analytics():
    res = requests.get(f"{API_URL}/api/v1/dashboard/analytics", timeout=60)
    res.raise_for_status()
    return res.json()


@st.cache_data(ttl=300, show_spinner=False)
def load_customers(page: int, page_size: int):
    res = requests.get(f"{API_URL}/api/v1/customers",
                       params={"page": page, "page_size": page_size}, timeout=60)
    res.raise_for_status()
    return res.json()


def get_customer(customer_id: str):
    res = requests.get(f"{API_URL}/api/v1/customers/{customer_id}", timeout=60)
    return res.json() if res.status_code == 200 else None


def connection_error():
    st.error("No pudimos conectar con el servidor. Intenta de nuevo en unos minutos.")


# --------------------------------------------- Traducción a lenguaje claro
VALUE_LABELS = {
    "Yes": "Sí", "No": "No", "Male": "Hombre", "Female": "Mujer",
    "Month-to-month": "Mes a mes", "One year": "1 año", "Two year": "2 años",
    "Fiber optic": "Fibra óptica", "DSL": "DSL",
    "No internet service": "Sin internet", "No phone service": "Sin teléfono",
    "Electronic check": "Cheque electrónico", "Mailed check": "Cheque por correo",
    "Bank transfer (automatic)": "Transferencia automática",
    "Credit card (automatic)": "Tarjeta de crédito automática",
    "True": "Sí", "False": "No", "Sí": "Sí",
}

# Variable del modelo -> (nombre claro, campo del cliente)
FEATURES = {
    "tenure": ("Tiempo como cliente", "tenure"),
    "MonthlyCharges": ("Cargo mensual", "monthly_charges"),
    "TotalCharges": ("Total facturado", "total_charges"),
    "SeniorCitizen": ("Adulto mayor (65+)", "senior_citizen"),
    "gender": ("Género", "gender"),
    "Partner": ("Tiene pareja", "partner"),
    "Dependents": ("Tiene dependientes", "dependents"),
    "PhoneService": ("Servicio telefónico", "phone_service"),
    "MultipleLines": ("Varias líneas", "multiple_lines"),
    "InternetService": ("Tipo de internet", "internet_service"),
    "OnlineSecurity": ("Seguridad online", "online_security"),
    "OnlineBackup": ("Copia de seguridad online", "online_backup"),
    "DeviceProtection": ("Protección de dispositivo", "device_protection"),
    "TechSupport": ("Soporte técnico", "tech_support"),
    "StreamingTV": ("TV por streaming", "streaming_tv"),
    "StreamingMovies": ("Películas por streaming", "streaming_movies"),
    "Contract": ("Tipo de contrato", "contract"),
    "PaperlessBilling": ("Factura electrónica", "paperless_billing"),
    "PaymentMethod": ("Método de pago", "payment_method"),
}


def plain(value) -> str:
    value = str(value)
    if value.endswith(" m"):  # grupos de antigüedad: "13-24 m"
        return value[:-2].replace("-", "–") + " meses"
    return VALUE_LABELS.get(value, value)


def customer_value(field: str, customer: dict) -> str:
    v = customer.get(field)
    if v is None:
        return "—"
    if field == "tenure":
        return f"{v} mes" if v == 1 else f"{v} meses"
    if field in ("monthly_charges", "total_charges"):
        return f"${float(v):,.2f}"
    return plain(v)


def group_reasons(shap_values: dict) -> list[tuple[str, str, float]]:
    """Suma el impacto de las columnas one-hot por variable original."""
    totals: dict[str, float] = {}
    for name, val in shap_values.items():
        base = name.split("_")[0]
        totals[base] = totals.get(base, 0.0) + float(val)
    return [(FEATURES.get(k, (k, ""))[0], FEATURES.get(k, (k, ""))[1], v) for k, v in totals.items()]


# ------------------------------------------------------------- Navegación
PAGES = ["Resumen", "Consultar cliente", "Clientes", "Mantenimiento"]

with st.sidebar:
    st.markdown(
        f'<div class="brand">{icon("logo")}{APP_TITLE}</div>'
        f'<div class="profile"><div class="avatar">CP</div>'
        f'<b>Retención de clientes</b><br><small>Predicción de abandono</small></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="nav-section">Menú</div>', unsafe_allow_html=True)
    page = st.radio("Menú", PAGES, label_visibility="collapsed")


# ---------------------------------------------------------------- Resumen
def render_overview():
    head, action = st.columns([5, 1])
    with head:
        page_header("Resumen", "Cuántos clientes podrían irse y dónde se concentra el riesgo.")
    with action:
        if st.button("Actualizar", width="stretch"):
            load_analytics.clear()

    try:
        data = load_analytics()
    except Exception:  # noqa: BLE001
        connection_error()
        return

    cust, preds = data["customers"], data["predictions"]
    scored = preds.get("customers_scored") or 0
    at_risk = preds.get("predicted_churn") or 0

    k = st.columns(4)
    with k[0]:
        kpi_card("k1", "users", f"{cust.get('total', 0):,}", "Clientes totales")
    with k[1]:
        kpi_card("k2", "alert", f"{at_risk:,}" if scored else "—", "Clientes en riesgo de irse", RED)
    with k[2]:
        kpi_card("k3", "percent", f"{at_risk / scored * 100:.1f}%" if scored else "—",
                 "Porcentaje en riesgo", AMBER)
    with k[3]:
        kpi_card("k4", "money", f"${cust.get('total_monthly_revenue', 0):,.0f}",
                 "Facturación mensual", GREEN)

    st.write("")

    if not scored:
        with st.container(key="card_empty"):
            st.markdown("**Todavía no se ha analizado a los clientes.**")
            st.caption("Ejecuta el análisis una vez para ver quién está en riesgo. Tarda unos minutos.")
            if st.button("Analizar a todos los clientes", type="primary"):
                run_batch()
        return

    top_row = st.columns([1.7, 1], gap="medium")

    with top_row[0]:
        with st.container(key="card_dist"):
            st.markdown('<div class="card-title">Distribución de la probabilidad de abandono</div>',
                        unsafe_allow_html=True)
            sample = pd.DataFrame(cust.get("sample", []))
            probs = (sample["churn_probability"].dropna().astype(float) * 100
                     if "churn_probability" in sample else pd.Series(dtype=float))
            if probs.empty:
                st.caption("Sin datos para esta vista.")
            else:
                bins = pd.cut(probs, bins=range(0, 105, 5), right=False, include_lowest=True)
                hist = bins.value_counts(sort=False)
                x = [iv.left + 2.5 for iv in hist.index]
                fig = go.Figure(go.Scatter(
                    x=x, y=hist.values, mode="lines+markers", fill="tozeroy",
                    line=dict(color=PRIMARY, width=2, shape="spline", smoothing=0.6),
                    fillcolor="rgba(37, 99, 235, .10)",
                    marker=dict(size=8, color="#FFFFFF", line=dict(color=PRIMARY, width=2)),
                    customdata=[f"{iv.left:.0f}–{iv.right:.0f}%" for iv in hist.index],
                    hovertemplate="%{customdata}<br>%{y:,} clientes<extra></extra>",
                ))
                threshold = float(((data.get("model") or {}).get("active") or {}).get("threshold") or 0.30) * 100
                fig.add_vline(x=threshold, line=dict(color=MUTED, width=1, dash="dot"),
                              annotation_text=f"Umbral {threshold:.0f}%",
                              annotation_font=dict(color=MUTED, size=11))
                fig.update_xaxes(range=[0, 100], ticksuffix="%", showgrid=False,
                                 linecolor=GRID, tickfont=dict(color=MUTED))
                fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED))
                plot(fig, 260)
                st.caption("Clientes según su probabilidad estimada de irse (muestra).")

    with top_row[1]:
        with st.container(key="card_seg"):
            st.markdown('<div class="card-title">¿Dónde se concentra el riesgo?</div>',
                        unsafe_allow_html=True)
            seg = cust.get("segments", {})
            options = {
                "Contrato": seg.get("Contrato", []),
                "Antigüedad": cust.get("tenure_groups", []),
                "Pago": seg.get("Método de pago", []),
                "Internet": seg.get("Internet", []),
            }
            choice = st.segmented_control("Agrupar por", list(options), default="Contrato",
                                          label_visibility="collapsed") or "Contrato"
            df = pd.DataFrame(options[choice])
            df = df.dropna(subset=["churn_rate"]) if "churn_rate" in df else df.iloc[0:0]
            if df.empty:
                st.caption("Sin datos para esta vista.")
            else:
                if choice == "Antigüedad":
                    df = df.sort_values("value", key=lambda s: s.str.extract(r"(\d+)")[0].astype(int))
                else:
                    df = df.sort_values("churn_rate")
                df["label"] = df["value"].map(plain)
                top = df["churn_rate"].max()
                fig = go.Figure(go.Bar(
                    x=df["churn_rate"], y=df["label"], orientation="h",
                    marker=dict(color=[ACCENT if v == top else "#93C5FD" for v in df["churn_rate"]],
                                cornerradius=6),
                    text=[f"{v:.0f}%" for v in df["churn_rate"]], textposition="outside",
                    cliponaxis=False, customdata=df["customers"],
                    hovertemplate="%{y}<br>%{x:.1f}% en riesgo<br>%{customdata:,} clientes<extra></extra>",
                ))
                fig.update_xaxes(visible=False, range=[0, max(top * 1.3, 10)])
                fig.update_yaxes(showgrid=False, ticksuffix="  ")
                fig.update_layout(bargap=0.5, margin=dict(l=0, r=36, t=10, b=0))
                plot(fig, 60 + 42 * len(df))
                worst = df.loc[df["churn_rate"].idxmax()]
                st.caption(f"Mayor riesgo: **{worst['label']}** "
                           f"({worst['churn_rate']:.0f}% podría irse).")

    st.write("")
    with st.container(key="card_levels"):
        st.markdown('<div class="card-title">Nivel de riesgo de la cartera</div>',
                    unsafe_allow_html=True)
        levels = preds.get("risk_levels", {})
        counts = [sum(v for k, v in levels.items() if k.startswith(lbl)) for lbl, *_ in RISK_LEVELS]
        total = sum(counts) or 1
        rings = "".join(
            ring(n / total * 100, color, f"Riesgo {lbl.lower()} · {n:,} clientes")
            for (lbl, color, *_), n in zip(RISK_LEVELS, counts)
        )
        st.markdown(f'<div class="rings">{rings}</div>', unsafe_allow_html=True)
        st.caption("Bajo: menos de 30% · Medio: 30–60% · Alto: 60% o más de probabilidad de irse.")


# ------------------------------------------------------ Consultar cliente
PLACEHOLDER = "Selecciona una opción"
YES_NO = ["Yes", "No"]
ADDON_FIELDS = [
    ("online_security", "Seguridad online"),
    ("online_backup", "Copia de seguridad online"),
    ("device_protection", "Protección de dispositivo"),
    ("tech_support", "Soporte técnico"),
    ("streaming_tv", "TV por streaming"),
    ("streaming_movies", "Películas por streaming"),
]
FORM_KEYS = ["f_gender", "f_senior", "f_partner", "f_dependents", "f_tenure", "f_contract",
             "f_payment", "f_paperless", "f_monthly", "f_total", "f_phone", "f_lines",
             "f_internet"] + [f"f_{field}" for field, _ in ADDON_FIELDS]


def form_section(number: int, title: str, subtitle: str, first: bool = False):
    st.markdown(
        f'<div class="form-section{" first" if first else ""}"><span class="step">{number}</span>'
        f'<div><div class="form-title">{title}</div><div class="form-sub">{subtitle}</div></div></div>',
        unsafe_allow_html=True,
    )


def select(label: str, options: list[str], key: str, help: str | None = None):
    return st.selectbox(label, options, index=None, placeholder=PLACEHOLDER,
                        format_func=plain, key=key, help=help)


def clear_form():
    for key in FORM_KEYS:
        st.session_state[key] = None
    st.session_state.pop("lookup_result", None)


def build_payload(v: dict) -> tuple[dict | None, list[str]]:
    """Valida el formulario y arma el cuerpo para /predict-data."""
    required = {
        "Género": v["gender"], "Adulto mayor": v["senior"], "Tiene pareja": v["partner"],
        "Tiene dependientes": v["dependents"], "Meses como cliente": v["tenure"],
        "Tipo de contrato": v["contract"], "Método de pago": v["payment"],
        "Factura electrónica": v["paperless"], "Cargo mensual": v["monthly"],
        "Servicio telefónico": v["phone"], "Tipo de internet": v["internet"],
    }
    if v["phone"] == "Yes":
        required["Varias líneas"] = v["lines"]
    if v["internet"] in ("DSL", "Fiber optic"):
        required.update({label: v["addons"][field] for field, label in ADDON_FIELDS})
    missing = [label for label, value in required.items() if value is None]
    if missing:
        return None, missing

    no_net = v["internet"] == "No"
    total = v["total"] if v["total"] is not None else v["tenure"] * v["monthly"]
    return {
        "gender": v["gender"],
        "senior_citizen": v["senior"] == "Yes",
        "partner": v["partner"],
        "dependents": v["dependents"],
        "tenure": int(v["tenure"]),
        "phone_service": v["phone"],
        "multiple_lines": "No phone service" if v["phone"] == "No" else v["lines"],
        "internet_service": v["internet"],
        **{field: "No internet service" if no_net else v["addons"][field]
           for field, _ in ADDON_FIELDS},
        "contract": v["contract"],
        "paperless_billing": v["paperless"],
        "payment_method": v["payment"],
        "monthly_charges": float(v["monthly"]),
        "total_charges": float(total),
    }, []


def render_lookup():
    page_header("Consultar cliente",
                "Completa los datos del cliente para estimar si podría irse y por qué.")

    form_col, result_col = st.columns([1.45, 1], gap="large")

    with form_col:
        with st.container(key="card_form"):
            with st.form("customer_form", border=False, enter_to_submit=False):
                form_section(1, "Datos personales", "Información básica del titular.", first=True)
                a, b = st.columns(2)
                gender = a.selectbox("Género", ["Female", "Male"], index=None,
                                     placeholder=PLACEHOLDER, format_func=plain, key="f_gender")
                senior = b.selectbox("Adulto mayor (65+)", YES_NO, index=None,
                                     placeholder=PLACEHOLDER, format_func=plain, key="f_senior")
                partner = a.selectbox("Tiene pareja", YES_NO, index=None,
                                      placeholder=PLACEHOLDER, format_func=plain, key="f_partner")
                dependents = b.selectbox("Tiene dependientes", YES_NO, index=None,
                                         placeholder=PLACEHOLDER, format_func=plain,
                                         key="f_dependents")

                form_section(2, "Contrato y facturación", "Condiciones actuales de su cuenta.")
                a, b = st.columns(2)
                tenure = a.number_input("Meses como cliente", min_value=0, max_value=72, step=1,
                                        value=None, placeholder="Ej. 12", key="f_tenure")
                contract = b.selectbox("Tipo de contrato",
                                       ["Month-to-month", "One year", "Two year"], index=None,
                                       placeholder=PLACEHOLDER, format_func=plain, key="f_contract")
                payment = a.selectbox("Método de pago",
                                      ["Electronic check", "Mailed check",
                                       "Bank transfer (automatic)", "Credit card (automatic)"],
                                      index=None, placeholder=PLACEHOLDER, format_func=plain,
                                      key="f_payment")
                paperless = b.selectbox("Factura electrónica", YES_NO, index=None,
                                        placeholder=PLACEHOLDER, format_func=plain,
                                        key="f_paperless")
                monthly = a.number_input("Cargo mensual ($)", min_value=0.0, max_value=500.0,
                                         step=5.0, value=None, placeholder="Ej. 70.00",
                                         key="f_monthly")
                total = b.number_input("Total facturado ($) · opcional", min_value=0.0,
                                       step=50.0, value=None, placeholder="Meses × cargo mensual",
                                       key="f_total",
                                       help="Si lo dejas vacío se calcula como meses × cargo mensual.")

                form_section(3, "Servicios contratados", "Qué servicios tiene activos.")
                a, b = st.columns(2)
                with a:
                    phone = select("Servicio telefónico", YES_NO, "f_phone")
                with b:
                    lines = select("Varias líneas", YES_NO, "f_lines",
                                   help="No aplica si no tiene servicio telefónico.")
                internet = select("Tipo de internet", ["DSL", "Fiber optic", "No"], "f_internet")
                st.caption("Servicios adicionales de internet (no aplican si no tiene internet):")
                addon_cols = st.columns(2)
                addons = {}
                for i, (field, label) in enumerate(ADDON_FIELDS):
                    with addon_cols[i % 2]:
                        addons[field] = select(label, YES_NO, f"f_{field}")

                st.write("")
                c_clear, _, c_submit = st.columns([1, 1.2, 1.4])
                c_clear.form_submit_button("Limpiar", on_click=clear_form, width="stretch")
                submitted = c_submit.form_submit_button("Calcular riesgo", type="primary",
                                                        width="stretch")

        if submitted:
            payload, missing = build_payload({
                "gender": gender, "senior": senior, "partner": partner,
                "dependents": dependents, "tenure": tenure, "contract": contract,
                "payment": payment, "paperless": paperless, "monthly": monthly, "total": total,
                "phone": phone, "lines": lines, "internet": internet, "addons": addons,
            })
            if missing:
                st.warning("Faltan campos por completar: " + ", ".join(missing) + ".")
            else:
                with st.spinner("Analizando…"):
                    try:
                        res = requests.post(f"{API_URL}/api/v1/predictions/predict-data",
                                            json=payload, timeout=90)
                    except Exception:  # noqa: BLE001
                        res = None
                if res is None:
                    connection_error()
                elif res.status_code != 200:
                    st.error("No se pudo calcular el riesgo en este momento.")
                else:
                    st.session_state["lookup_result"] = res.json()
                    st.session_state["lookup_customer"] = payload

    with result_col:
        render_result()


def render_result():
    result = st.session_state.get("lookup_result")
    if not result:
        with st.container(key="card_score_empty"):
            st.markdown('<div class="card-title">Resultado</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="display:flex;justify-content:center;opacity:.5">'
                f'{ring(0, TRACK, "", size=150, stroke=12)}</div>',
                unsafe_allow_html=True,
            )
            st.caption("Completa el formulario y pulsa **Calcular riesgo** para ver la "
                       "probabilidad de que el cliente se vaya.")
        return

    prob = float(result["prediction"]["churn_probability"])
    label, color = risk_level(prob)
    customer = st.session_state.get("lookup_customer", {})
    advice = {
        "Alto": "Contactar esta semana con una oferta de retención.",
        "Medio": "Hacer seguimiento y revisar su satisfacción.",
        "Bajo": "No requiere acción por ahora.",
    }[label]

    with st.container(key="card_score"):
        st.markdown('<div class="card-title">Probabilidad de que se vaya</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div style="display:flex;justify-content:center">'
            f'{ring(prob * 100, color, "", size=150, stroke=12)}</div>'
            f'<div style="margin:.8rem 0;text-align:center"><span class="pill" '
            f'style="background:{color}1A;color:{color}">Riesgo {label.lower()}</span></div>'
            f'<div style="font-size:.9rem"><b>Qué hacer:</b> {advice}</div>',
            unsafe_allow_html=True,
        )

    st.write("")
    with st.container(key="card_reasons"):
        st.markdown('<div class="card-title">Por qué</div>', unsafe_allow_html=True)
        shap = result.get("explainability", {}).get("all_shap_values") or {}
        if not shap:
            st.caption("No hay una explicación disponible para esta predicción.")
            return
        reasons = group_reasons(shap)
        up = sorted([r for r in reasons if r[2] > 0], key=lambda r: -r[2])[:3]
        down = sorted([r for r in reasons if r[2] < 0], key=lambda r: r[2])[:3]

        def block(title: str, items, c: str):
            html = "".join(
                f'<div class="reason"><span class="dot" style="border-color:{c}"></span>{name}'
                f'<br><small style="margin-left:22px">{customer_value(field, customer)}</small></div>'
                for name, field, _ in items
            ) or '<div class="reason"><small>Nada destacable</small></div>'
            st.markdown(f"**{title}**{html}", unsafe_allow_html=True)

        block("Aumenta el riesgo", up, RED)
        st.write("")
        block("Reduce el riesgo", down, GREEN)


# --------------------------------------------------------------- Clientes
def render_customers():
    page_header("Clientes", "Listado de clientes registrados.")
    page_size = 20
    page_no = st.session_state.get("customers_page", 1)

    try:
        data = load_customers(page_no, page_size)
    except Exception:  # noqa: BLE001
        connection_error()
        return

    df = pd.DataFrame(data["customers"])
    with st.container(key="card_table"):
        if df.empty:
            st.caption("No hay clientes registrados.")
            return
        view = pd.DataFrame({
            "Código": df["customer_id"],
            "Género": df["gender"].map(plain),
            "Meses como cliente": df["tenure"],
            "Contrato": df["contract"].map(plain),
            "Internet": df["internet_service"].map(plain),
            "Método de pago": df["payment_method"].map(plain),
            "Cargo mensual": df["monthly_charges"].astype(float),
        })
        st.dataframe(
            view, width="stretch", hide_index=True, height=740,
            column_config={"Cargo mensual": st.column_config.NumberColumn(format="$%.2f")},
        )

        pages = max(1, -(-data["total"] // page_size))
        c1, c2, c3 = st.columns([1, 3, 1], vertical_alignment="center")
        if c1.button("← Anterior", disabled=page_no <= 1, width="stretch"):
            st.session_state["customers_page"] = page_no - 1
            st.rerun()
        c2.markdown(f'<div style="text-align:center;color:{MUTED};font-size:.85rem">'
                    f'Página {page_no} de {pages} · {data["total"]:,} clientes</div>',
                    unsafe_allow_html=True)
        if c3.button("Siguiente →", disabled=page_no >= pages, width="stretch"):
            st.session_state["customers_page"] = page_no + 1
            st.rerun()


# ----------------------------------------------------------- Mantenimiento
def run_batch():
    with st.spinner("Analizando a todos los clientes… puede tardar unos minutos."):
        try:
            res = requests.post(f"{API_URL}/api/v1/predictions/batch", timeout=180)
        except Exception:  # noqa: BLE001
            connection_error()
            return
    if res.status_code == 200:
        r = res.json()
        load_analytics.clear()
        st.success(f"Listo: se analizaron {r['total_customers']:,} clientes; "
                   f"{r['churn_predicted']:,} están en riesgo.")
    else:
        st.error("El análisis no se pudo completar.")


def render_maintenance():
    page_header("Mantenimiento", "Tareas para mantener los resultados al día.")

    a, b = st.columns(2, gap="medium")
    with a:
        with st.container(key="card_batch"):
            st.markdown("**Analizar a todos los clientes**")
            st.caption("Recalcula el riesgo de cada cliente con los datos más recientes. "
                       "Hazlo cuando se agreguen o actualicen clientes.")
            if st.button("Analizar ahora", type="primary"):
                run_batch()
    with b:
        with st.container(key="card_retrain"):
            st.markdown("**Actualizar el modelo**")
            st.caption("Vuelve a entrenar el modelo. Se hace solo el día 1 de cada mes; "
                       "el nuevo modelo solo se usa si es mejor que el actual.")
            if st.button("Actualizar modelo"):
                try:
                    res = requests.post(f"{API_URL}/api/v1/retrain/trigger", timeout=90)
                    if res.status_code == 200:
                        st.success("Actualización iniciada. Puede tardar varios minutos.")
                    else:
                        st.error("No se pudo iniciar la actualización.")
                except Exception:  # noqa: BLE001
                    connection_error()

    st.write("")
    try:
        data = load_analytics()
    except Exception:  # noqa: BLE001
        return
    active = (data.get("model") or {}).get("active") or {}
    with st.container(key="card_model"):
        st.markdown('<div class="card-title">Calidad del modelo actual</div>', unsafe_allow_html=True)
        if not active:
            st.caption("No hay un modelo activo.")
        else:
            recall = active.get("recall")
            precision = active.get("precision")
            c = st.columns(2)
            if recall is not None:
                c[0].metric("Detecta a los que se van", f"{recall * 100:.0f}%",
                            help="De cada 100 clientes que realmente se van, cuántos identifica el modelo.")
            if precision is not None:
                c[1].metric("Aciertos cuando avisa", f"{precision * 100:.0f}%",
                            help="De cada 100 clientes marcados en riesgo, cuántos realmente se van.")

        history = pd.DataFrame((data.get("model") or {}).get("retrain_history", []))
        if not history.empty:
            with st.expander("Detalles técnicos"):
                st.caption(f"Modelo: {active.get('model_name', '—')} · versión "
                           f"{active.get('model_version', '—')} · F1 {active.get('f1_score') or 0:.3f}")
                st.dataframe(history.sort_values("started_at", ascending=False),
                             width="stretch", hide_index=True)


{
    "Resumen": render_overview,
    "Consultar cliente": render_lookup,
    "Clientes": render_customers,
    "Mantenimiento": render_maintenance,
}[page]()
