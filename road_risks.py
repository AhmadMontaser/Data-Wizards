import pandas as pd
import numpy as np
import streamlit as st
import joblib
import ollama

st.set_page_config(page_title="Advanced Road Safety Predictor", layout="centered")
st.title("🚗 Advanced Road Safety Predictor")
st.write("Fill out the road characteristics. Gaps are automatically resolved by the internal pipeline.")

@st.cache_resource
def load_model():
    return joblib.load('Enchanted_forest.joblib')

try:
    rf_model = load_model()
except Exception as e:
    st.error(f"Failed to load the model file: {e}")
    st.stop()

with st.form(key='road_safety_form'):
    st.subheader("Numerical Features")
    curvature_in = st.text_input("Road Curvature (Leave blank if missing):", "")
    speed_in = st.text_input("Speed Limit (Leave blank if missing):", "")
    accidents_in = st.text_input("Historically Reported Accidents (Leave blank if missing):", "")

    st.subheader("Categorical Features")
    lighting = st.selectbox("Lighting Conditions:", ["daylight", "dim", "night"])
    weather = st.selectbox("Weather Conditions:", ["clear", "rainy", "foggy"])

    st.subheader("Boolean Features")
    public_road = st.selectbox("Is it a Public Road?", [1, 0])
    holiday = st.selectbox("Is it a Holiday?", [1, 0])

    submit_button = st.form_submit_button(label="Analyze Safety & Get Advice")

if submit_button:
    with st.spinner("Processing features and predicting..."):
        try:
            curvature = float(curvature_in) if curvature_in.strip() else np.nan
            speed_limit = float(speed_in) if speed_in.strip() else np.nan
            num_reported_accidents = float(accidents_in) if accidents_in.strip() else np.nan
            
            new_entry = {
                "curvature": curvature,
                "speed_limit": speed_limit,
                "num_reported_accidents": num_reported_accidents,
                "lighting": lighting,
                "weather": weather,
                "public_road": public_road,
                "holiday": holiday
            }

            feature_names = ['curvature', 'speed_limit', 'num_reported_accidents', 'lighting', 'weather', 'public_road', 'holiday']
            df_input = pd.DataFrame([new_entry], columns=feature_names)
            
            probabilities = rf_model.predict(df_input)
            print(probabilities)
            accident_prob = probabilities[0] * 100  
            
            st.subheader("📊 Accident Probability Analysis")
            if accident_prob > 50:
                st.error(f"High Risk Detected: {accident_prob:.1f}% chance of an accident.")
            elif accident_prob > 20:
                st.warning(f"Moderate Risk Detected: {accident_prob:.1f}% chance of an accident.")
            else:
                st.success(f"Low Risk Detected: {accident_prob:.1f}% chance of an accident.")
                
            prompt = f"""
            A Random Forest model analyzed these road metrics:
            - Speed Limit: {speed_in if speed_in.strip() else 'Unknown'} km/h
            - Road Curvature: {curvature_in if curvature_in.strip() else 'Unknown'}
            - Historical Accidents: {accidents_in if accidents_in.strip() else 'Unknown'}
            - Lighting condition: {lighting}
            - Weather condition: {weather}
            - Public Road status: {public_road}
            - Holiday status: {holiday}
            
            The model computed a {accident_prob:.1f}% risk of an accident based on these parameters.
            
            Is this road safe? and what should I care about?
            """
            
            response = ollama.chat(
                model='llama3.2:1b',
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            st.subheader("🤖 AI Safety Recommendations:")
            st.info(response['message']['content'])
            
        except ValueError:
            st.error("⚠️ Validation Error: Please ensure numerical inputs contain only valid numbers.")
        except Exception as e:
            st.error(f"An error occurred during calculation: {e}")



























# import streamlit as st
# import ollama

# st.set_page_config(page_title="Llama 3.2 Web Interface", layout="centered")
# st.title("💬 Llama 3.2 Web Interface")
# st.write("Type your prompt in the form below to get a response from the local LLM.")

# with st.form(key='llm_form', clear_on_submit=False):
#     user_input = st.text_area("Enter your prompt or question here:", placeholder="e.g., Explain quantum computing in simple terms...")
    
#     submit_button = st.form_submit_button(label='Submit to Model')

# if submit_button:
#     if not user_input.strip():
#         st.warning("Please enter some text before submitting.")
#     else:
#         with st.spinner("Thinking and generating response..."):
#             try:
#                 response = ollama.chat(
#                     model='llama3.2:1b',
#                     messages=[{'role': 'user', 'content': user_input}]
#                 )
                
#                 st.subheader("🤖 Model Response:")
#                 st.info(response['message']['content'])
                
#             except Exception as e:
#                 st.error(f"An error occurred while connecting to Ollama: {e}")
