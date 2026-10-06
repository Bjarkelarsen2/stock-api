import requests
import streamlit as st

API_URL = "https://stock-api-1-jl0r.onrender.com"

st.title("Næste lukkekurs")

symbol = st.text_input("Stock symbol", "TSLA")

if st.button("Predict next close"):
    try:
        r = requests.get(
            f"{API_URL}/predict/live",
            params={"symbol": symbol},
            timeout=30,
        )
    except requests.RequestException as e:
        st.error(f"Kunne ikke nå API'et: {e}")
        st.stop()

    if r.status_code != 200:
        st.error(
            f"API'et svarede {r.status_code} på /predict/live. "
            "Endpointet er nede – brug den manuelle beregning nedenfor."
        )
        st.stop()

    try:
        data = r.json()
    except ValueError:
        st.error("API'et returnerede ikke JSON:")
        st.code(r.text[:500])
        st.stop()

    if "predicted_next_close" not in data:
        st.error("Uventet svar fra API'et:")
        st.json(data)
        st.stop()

    st.metric("Predicted next close", f"{data['predicted_next_close']:.2f}")
    if "naive_forecast" in data:
        st.caption(f"Naiv baseline (= seneste close): {data['naive_forecast']:.2f}")

st.divider()
st.subheader("Manuel beregning")
st.caption("Virker, mens /predict/live er nede – /predict er oppe.")

lag_1 = st.number_input("Seneste lukkekurs (lag_1)", value=440.0, step=1.0)
lag_2 = st.number_input("Lukkekursen før den (lag_2)", value=435.0, step=1.0)

if st.button("Beregn"):
    r = requests.get(
        f"{API_URL}/predict",
        params={"lag_1": lag_1, "lag_2": lag_2},
        timeout=30,
    )
    if not r.ok:
        st.error(f"API'et svarede {r.status_code}")
    else:
        d = r.json()
        st.metric("Predicted next close", f"{d['predicted_next_close']:.2f}")
        st.caption(f"Naiv baseline: {d['naive_forecast']:.2f}")
