import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Data Collector UI", layout="centered")
st.title("📋 Road Data Collection Interface")
st.write("Enter the configuration metrics below to log them into a local dataset.")

LOG_FILE_PATH = "captured_road_data.csv"

with st.form(key="data_collector_form", clear_on_submit=True):

    st.subheader("Numerical Features")
    curvature_in = st.text_input("Road Curvature (Leave blank if missing):", "")
    speed_in = st.text_input("Speed Limit (Leave blank if missing):", "")
    accidents_in = st.text_input(
        "Historically Reported Accidents (Leave blank if missing):", ""
    )

    st.subheader("Categorical Features")
    lighting = st.selectbox(
        "Lighting Conditions:",
        ["daylight", "dim", "night"],
    )
    weather = st.selectbox(
        "Weather Conditions:",
        ["clear", "rainy", "foggy"],
    )

    st.subheader("Boolean Features")
    public_road = st.selectbox("Is it a Public Road?", [1, 0])
    holiday = st.selectbox("Is it a Holiday?", [1, 0])

    submit_button = st.form_submit_button(label="Save Inputs to Dataset")

if submit_button:
    try:
        new_entry = {
            "curvature": (
                float(curvature_in) if curvature_in.strip() else None
            ),
            "speed_limit": float(speed_in) if speed_in.strip() else None,
            "num_reported_accidents": (
                float(accidents_in) if accidents_in.strip() else None
            ),
            "lighting": lighting if lighting != "Unknown" else None,
            "weather": weather if weather != "Unknown" else None,
            "public_road": public_road if public_road != "Unknown" else None,
            "holiday": holiday if holiday != "Unknown" else None,
        }

        df_new = pd.DataFrame([new_entry])

        file_exists = os.path.isfile(LOG_FILE_PATH)
        df_new.to_csv(
            LOG_FILE_PATH, mode="a", index=False, header=not file_exists
        )

        st.success(f"🎉 Row successfully appended to '{LOG_FILE_PATH}'!")
        st.write("### Captured DataFrame Row:")
        st.dataframe(df_new)

    except ValueError:
        st.error(
            "⚠️ Validation Error: Please ensure numerical inputs contain only valid numbers."
        )
    except Exception as e:
        st.error(f"An unexpected storage error occurred: {e}")

if os.path.isfile(LOG_FILE_PATH):
    st.sidebar.subheader("📈 Local Log Preview")
    df_logged = pd.read_csv(LOG_FILE_PATH)
    st.sidebar.write(f"Total entries: {len(df_logged)}")
    st.sidebar.dataframe(df_logged.tail(5))  
