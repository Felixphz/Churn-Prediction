"""Página "Dashboard": métricas de clientes, servicios, predicciones y modelo."""
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

CHURN_COLOR = "#E4572E"
OK_COLOR = "#2E86AB"
NEUTRAL = "#8D99AE"
RISK_COLORS = {"Bajo (<30%)": "#4CAF50", "Medio (30-60%)": "#F6AE2D", "Alto (≥60%)": CHURN_COLOR}


@st.cache_data(ttl=300, show_spinner="Cargando métricas…")
def _load(api_url: str) -> Optional[dict]:
    res = requests.get(f"{api_url}/api/v1/dashboard/analytics", timeout=90)
    if res.status_code == 404:
        return None
    res.raise_for_status()
    return res.json()


def _fmt_pct(x):
    return "—" if x is None else f"{x:.1f}%"


def _bar_rate(rows: list[dict], title: str, x_label: str):
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig = go.Figure()
    fig.add_bar(x=df["value"], y=df["customers"], name="Clientes", marker_color=NEUTRAL, yaxis="y")
    if df["churn_rate"].notna().any():
        fig.add_scatter(
            x=df["value"], y=df["churn_rate"], name="% churn predicho",
            mode="lines+markers+text", text=df["churn_rate"].map(_fmt_pct),
            textposition="top center", marker_color=CHURN_COLOR, yaxis="y2",
        )
    fig.update_layout(
        title=title, xaxis_title=x_label, height=360, margin=dict(t=50, b=10),
        yaxis=dict(title="Clientes"),
        yaxis2=dict(title="% churn", overlaying="y", side="right", rangemode="tozero", showgrid=False),
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(fig, use_container_width=True)


def render(api_url: str):
    st.header("📊 Dashboard")
    c1, c2 = st.columns([6, 1])
    if c2.button("🔄 Actualizar"):
        _load.clear()

    try:
        data = _load(api_url)
    except Exception as e:  # noqa: BLE001
        st.error(f"No se pudo conectar con la API: {e}")
        return
    if data is None:
        st.warning(
            "La API todavía no tiene el endpoint `/api/v1/dashboard/analytics`. "
            "Vuelve a desplegar el backend con la última versión."
        )
        return

    cust, preds, model = data["customers"], data["predictions"], data["model"]
    active = model.get("active") or {}

    # ------------------------------------------------------------------ KPIs
    k = st.columns(6)
    k[0].metric("Clientes", f"{cust['total']:,}")
    k[1].metric("Clientes evaluados", f"{preds.get('customers_scored', 0):,}")
    scored = preds.get("customers_scored") or 0
    churn_n = preds.get("predicted_churn", 0)
    k[2].metric("Churn predicho", f"{churn_n:,}",
                f"{churn_n / scored * 100:.1f}% de evaluados" if scored else None,
                delta_color="inverse")
    k[3].metric("Prob. media churn", _fmt_pct((preds.get("avg_probability") or 0) * 100) if preds.get("avg_probability") is not None else "—")
    k[4].metric("Cargo mensual medio", f"${cust.get('avg_monthly_charges', 0):,.2f}")
    k[5].metric("Modelo activo", active.get("model_name", "N/A"),
                f"F1 {active['f1_score']:.3f}" if active.get("f1_score") else None, delta_color="off")

    if scored == 0:
        st.info("Aún no hay predicciones. Ejecuta **Run Batch** en *Pipeline* para puntuar a todos los clientes.")

    tab_cli, tab_srv, tab_pred, tab_model = st.tabs(
        ["👥 Clientes", "📡 Servicios", "🔮 Predicciones", "🧠 Modelo"]
    )

    # -------------------------------------------------------------- Clientes
    with tab_cli:
        seg = cust.get("segments", {})
        m = st.columns(4)
        m[0].metric("Adultos mayores (65+)", _fmt_pct(cust.get("senior_pct")))
        m[1].metric("Antigüedad media", f"{cust.get('avg_tenure', 0)} meses")
        m[2].metric("Servicios por cliente", cust.get("avg_services", 0))
        m[3].metric("Ingreso mensual total", f"${cust.get('total_monthly_revenue', 0):,.0f}")
        st.caption(
            "El dataset (Telco) no trae la edad exacta: la variable de edad es "
            "**SeniorCitizen** (cliente de 65 años o más)."
        )

        a, b = st.columns(2)
        with a:
            _bar_rate(seg.get("Adulto mayor (65+)", []), "Edad: adulto mayor vs. resto", "Adulto mayor (65+)")
        with b:
            _bar_rate(cust.get("tenure_groups", []), "Antigüedad del cliente", "Meses con la compañía")

        a, b = st.columns(2)
        with a:
            _bar_rate(seg.get("Contrato", []), "Tipo de contrato", "Contrato")
        with b:
            _bar_rate(seg.get("Método de pago", []), "Método de pago", "Método")

        a, b, c = st.columns(3)
        for col, name in zip((a, b, c), ("Género", "Pareja", "Dependientes")):
            df = pd.DataFrame(seg.get(name, []))
            if not df.empty:
                fig = px.pie(df, names="value", values="customers", hole=0.55, title=name,
                             color_discrete_sequence=[OK_COLOR, "#A23B72", NEUTRAL])
                fig.update_layout(height=300, margin=dict(t=50, b=0), showlegend=True)
                col.plotly_chart(fig, use_container_width=True)

        sample = pd.DataFrame(cust.get("sample", []))
        if not sample.empty:
            a, b = st.columns(2)
            with a:
                fig = px.histogram(sample, x="tenure", nbins=24, color="contract",
                                   title="Distribución de antigüedad por contrato",
                                   labels={"tenure": "Meses", "contract": "Contrato"})
                fig.update_layout(height=360, bargap=0.05)
                st.plotly_chart(fig, use_container_width=True)
            with b:
                fig = px.box(sample, x="contract", y="monthly_charges", color="senior_citizen",
                             title="Cargo mensual por contrato y edad",
                             labels={"monthly_charges": "Cargo mensual ($)", "contract": "Contrato",
                                     "senior_citizen": "65+"})
                fig.update_layout(height=360)
                st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------- Servicios
    with tab_srv:
        srv = pd.DataFrame(cust.get("services", []))
        seg = cust.get("segments", {})
        a, b = st.columns(2)
        with a:
            if not srv.empty:
                fig = px.bar(srv.sort_values("adoption_pct"), x="adoption_pct", y="service",
                             orientation="h", text=srv.sort_values("adoption_pct")["adoption_pct"].map(_fmt_pct),
                             title="Adopción de servicios (% de clientes)",
                             labels={"adoption_pct": "% clientes", "service": ""},
                             color_discrete_sequence=[OK_COLOR])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        with b:
            internet = pd.DataFrame(seg.get("Internet", []))
            if not internet.empty:
                fig = px.pie(internet, names="value", values="customers", hole=0.55,
                             title="Tipo de servicio de internet")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        if not srv.empty and srv["churn_rate_with"].notna().any():
            long = srv.melt(id_vars="service", value_vars=["churn_rate_with", "churn_rate_without"],
                            var_name="tiene", value_name="churn")
            long["tiene"] = long["tiene"].map({"churn_rate_with": "Con el servicio",
                                               "churn_rate_without": "Sin el servicio"})
            fig = px.bar(long, x="service", y="churn", color="tiene", barmode="group",
                         title="% churn predicho con vs. sin cada servicio",
                         labels={"churn": "% churn", "service": "", "tiene": ""},
                         color_discrete_map={"Con el servicio": OK_COLOR, "Sin el servicio": CHURN_COLOR})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        a, b = st.columns(2)
        with a:
            rows = sorted(cust.get("services_count", []), key=lambda r: int(r["value"]))
            _bar_rate(rows, "Número de servicios contratados", "Servicios")
        with b:
            _bar_rate(seg.get("Internet", []), "Churn por tipo de internet", "Internet")

    # ---------------------------------------------------------- Predicciones
    with tab_pred:
        if not preds.get("total"):
            st.info("Todavía no hay predicciones registradas.")
        else:
            m = st.columns(3)
            m[0].metric("Predicciones totales", f"{preds['total']:,}")
            bt = preds.get("by_type", {})
            m[1].metric("Individuales / Batch", f"{bt.get('on_demand', 0):,} / {bt.get('batch', 0):,}")
            m[2].metric("Modelos usados", ", ".join(preds.get("by_model", {}).keys()) or "—")

            a, b = st.columns(2)
            with a:
                risk = preds.get("risk_levels", {})
                if risk:
                    rdf = pd.DataFrame({"nivel": list(risk), "clientes": list(risk.values())})
                    fig = px.pie(rdf, names="nivel", values="clientes", hole=0.55,
                                 title="Clientes por nivel de riesgo", color="nivel",
                                 color_discrete_map=RISK_COLORS)
                    fig.update_layout(height=360)
                    st.plotly_chart(fig, use_container_width=True)
            with b:
                hist = pd.DataFrame(preds.get("probability_histogram", []))
                if not hist.empty:
                    thr = active.get("threshold") or 0.3
                    fig = px.bar(hist, x="bin", y="count", title="Distribución de probabilidad de churn",
                                 labels={"bin": "Probabilidad", "count": "Clientes"},
                                 color_discrete_sequence=[OK_COLOR])
                    fig.add_vline(x=thr * 10 - 0.5, line_dash="dash", line_color=CHURN_COLOR,
                                  annotation_text=f"umbral {thr:.2f}")
                    fig.update_layout(height=360)
                    st.plotly_chart(fig, use_container_width=True)

            daily = pd.DataFrame(preds.get("daily", []))
            if not daily.empty:
                daily["% churn"] = (daily["churn"] / daily["predictions"] * 100).round(2)
                fig = go.Figure()
                fig.add_bar(x=daily["day"], y=daily["predictions"], name="Predicciones", marker_color=NEUTRAL)
                fig.add_scatter(x=daily["day"], y=daily["% churn"], name="% churn", yaxis="y2",
                                mode="lines+markers", marker_color=CHURN_COLOR)
                fig.update_layout(title="Predicciones por día", height=360,
                                  yaxis=dict(title="Predicciones"),
                                  yaxis2=dict(title="% churn", overlaying="y", side="right", showgrid=False),
                                  legend=dict(orientation="h", y=-0.25))
                st.plotly_chart(fig, use_container_width=True)

            sample = pd.DataFrame(cust.get("sample", [])).dropna(subset=["churn_probability"])
            if not sample.empty:
                fig = px.scatter(sample, x="tenure", y="monthly_charges", color="churn_probability",
                                 color_continuous_scale="RdYlGn_r", opacity=0.7,
                                 title="Riesgo según antigüedad y cargo mensual",
                                 labels={"tenure": "Meses", "monthly_charges": "Cargo mensual ($)",
                                         "churn_probability": "Prob."})
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------------- Modelo
    with tab_model:
        if active:
            m = st.columns(5)
            m[0].metric("F1-score", f"{active['f1_score']:.3f}" if active.get("f1_score") is not None else "—")
            m[1].metric("Recall", f"{active['recall']:.3f}" if active.get("recall") is not None else "—")
            m[2].metric("Precisión", f"{active['precision']:.3f}" if active.get("precision") is not None else "—")
            m[3].metric("AUC-ROC", f"{active['auc_roc']:.3f}" if active.get("auc_roc") is not None else "—")
            m[4].metric("Umbral", f"{active['threshold']:.2f}" if active.get("threshold") is not None else "—")
            st.caption(f"{active.get('model_name')} · versión {active.get('model_version')} · "
                       f"resampling: {active.get('resampling_strategy') or '—'}")
        else:
            st.info("No hay un modelo activo en el registro.")

        a, b = st.columns(2)
        with a:
            feats = pd.DataFrame(preds.get("top_features", []))
            if not feats.empty:
                feats = feats.sort_values("mean_abs_shap")
                fig = px.bar(feats, x="mean_abs_shap", y="feature", orientation="h",
                             title="Importancia de variables (|SHAP| medio)",
                             labels={"mean_abs_shap": "|SHAP| medio", "feature": ""},
                             color_discrete_sequence=["#A23B72"])
                fig.update_layout(height=460)
                st.plotly_chart(fig, use_container_width=True)
        with b:
            reg = pd.DataFrame(model.get("registry", []))
            if not reg.empty:
                reg["modelo"] = reg["model_name"] + " " + reg["model_version"].astype(str)
                long = reg.melt(id_vars="modelo", value_vars=["f1_score", "recall", "precision", "auc_roc"],
                                var_name="métrica", value_name="valor").dropna()
                fig = px.bar(long, x="modelo", y="valor", color="métrica", barmode="group",
                             title="Comparación de modelos registrados", range_y=[0, 1])
                fig.update_layout(height=460)
                st.plotly_chart(fig, use_container_width=True)

        hist = pd.DataFrame(model.get("retrain_history", []))
        if not hist.empty:
            ok = hist[hist["status"].str.lower().isin(["completed", "success", "succeeded"])] if "status" in hist else hist
            ok = ok.dropna(subset=["f1_score"])
            if not ok.empty:
                long = ok.melt(id_vars="started_at", value_vars=["f1_score", "recall", "auc_roc"],
                               var_name="métrica", value_name="valor")
                fig = px.line(long, x="started_at", y="valor", color="métrica", markers=True,
                              title="Evolución en reentrenamientos", range_y=[0, 1],
                              labels={"started_at": "Fecha"})
                fig.update_layout(height=360)
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(hist, use_container_width=True, hide_index=True)
