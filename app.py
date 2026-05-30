# import streamlit as st

# st.title("Power Consumption Prediction ⚡")
# st.write("Welcome to the Ai driven Power Consumption Preddictor.")

# st.header("Input Features")
# st.subheader("Weather Information")
# st.markdown("""
# ### Instructions

# - Fill all fields
# - Click Predict
# """)

# voltage = st.number_input(
#     "Voltage",
#     min_value=0.0,
#     max_value=500.0,
#     value=240.0
# )

# month = st.selectbox(
#     "Month",
#     list(range(1,13))
# )

# # city = st.text_input("City")


# # st.write(voltage)

# prediction = 2.54

# st.metric(
#     label="Predicted Power Consumption",
#     value=f"{prediction:.2f} kW"
# )

# if st.button("Predict"):
#     st.write("Predicting...")

# col1, col2 = st.columns(2)

# with col1:
#     voltage = st.number_input("Voltage")

# with col2:
#     temp = st.number_input("Temperature")


# import streamlit as st

# # Sidebar title
# st.sidebar.title("Select Model")

# # Model selection dropdown
# model = st.sidebar.selectbox(
#     "Choose a model:",
#     ["Random Forest", "Sarimax", "Linear Regression", "XGBoost"]
# )


# import pandas as pd

# df = pd.read_csv("Data/cleaned_data.csv")

# st.dataframe(df.head())


# st.line_chart(df["Global_active_power"])



###################################################3

# import streamlit as st

# st.title("Power Consumption Prediction ⚡")
# st.write("Welcome to the Ai driven Power Consumption Preddictor.")

# import streamlit as st

# # Sidebar title
# st.sidebar.title("Select Model")

# # Model selection dropdown
# model = st.sidebar.selectbox(
#     "Choose a model:",
#     ["Random Forest", "Sarimax", "Linear Regression", "XGBoost"]
# )


# st.header("Inputs :")

# global_reactive_power = st.slider(
#     "Global_reactive_power",
#     min_value=0.0,
#     max_value=5.0,
#     value=0.168850
# )

# voltage = st.slider(
#     "Voltage",
#     min_value=200,
#     max_value=270,
#     value= 220

# )

import streamlit as st
import pickle as pk
import pandas as pd

st.set_page_config(page_title="Power Consumption Predictor", layout="wide")

st.title("Power Consumption Prediction ⚡")
st.write("Welcome to the AI-driven Power Consumption Predictor.")

# --- SIDEBAR MODEL SELECTION ---
st.sidebar.title("Configuration")
model = st.sidebar.selectbox(
    "Choose a predictive model:",
    ["Random Forest", "Sarimax", "Linear Regression", "XGBoost"]
)

st.header("Input Features")
st.write("Adjust the parameters below to generate a power consumption prediction.")

# --- COLUMN LAYOUT FOR INPUTS ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🔌 Electrical Metrics")
    
    global_reactive_power = st.slider(
        "Global Reactive Power",
        min_value=0.0,
        max_value=1.5,
        value=0.123,
        step=0.01,
        help="Grid reactive power. Usually keeps below 1.0"
    )
    
    voltage = st.slider(
        "Voltage (V)",
        min_value=200.0,
        max_value=260.0,
        value=241.0,
        step=0.5
    )

with col2:
    st.subheader("🧼 Sub-Metering (Energy)")
    
    sub_metering_1 = st.slider(
        "Sub Metering 1 (Kitchen)",
        min_value=0.0,
        max_value=50.0,
        value=1.1,
        step=0.1
    )
    
    sub_metering_2 = st.slider(
        "Sub Metering 2 (Laundry)",
        min_value=0.0,
        max_value=50.0,
        value=1.2,
        step=0.1
    )
    
    sub_metering_3 = st.slider(
        "Sub Metering 3 (Climate Control)",
        min_value=0.0,
        max_value=30.0,
        value=6.4,
        step=0.1
    )

with col3:
    st.subheader("🌤️ Weather Metrics")
    
    # Using number_input for precision, but restricting limits to your dataset bounds
    temperature = st.number_input(
        "Temperature (Kelvin)",
        min_value=250.0,
        max_value=320.0,
        value=284.5,
        step=0.1,
        help="284.5 Kelvin is approx. 11.3°C"
    )
    
    wind = st.number_input(
        "Wind Speed",
        min_value=0.0,
        max_value=20.0,
        value=3.4,
        step=0.1
    )
    
    cloud_cover = st.slider(
        "Cloud Cover (Scale 0-1)",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05
    )
    
    precipitation = st.number_input(
        "Precipitation Amount",
        min_value=0.000000,
        max_value=0.000100,
        value=0.000000,
        format="%.6f"
    )
    
    snowfall = st.number_input(
        "Snowfall Amount",
        min_value=0.000000,
        max_value=0.000100,
        value=0.000000,
        format="%.6f"
    )

# --- TIME & DATE SELECTION ---
st.markdown("---")
st.subheader("📅 Timestamp Selection")
time_col1, time_col2, time_col3, time_col4 = st.columns(4)

with time_col1:
    year = st.selectbox("Year", [2026,2027,2028,2029,2030], index=1)
with time_col2:
    month = st.slider("Month", min_value=1, max_value=12, value=6)
with time_col3:
    day = st.slider("Day", min_value=1, max_value=31, value=15)
with time_col4:
    hour = st.slider("Hour of Day (24h)", min_value=0, max_value=23, value=12)

# --- PREDICTION BUTTON ---
st.markdown("---")
if st.button("Generate Prediction 🚀", type="primary"):
    st.success(f"Running predictions using **{model}**...")
    
    # This is a dictionary of the exact features structured to pass into your ML model
    features_dict = {
        "Global_reactive_power": global_reactive_power,
        "Voltage": voltage,
        "Sub_metering_1": sub_metering_1,
        "Sub_metering_2": sub_metering_2,
        "Sub_metering_3": sub_metering_3,
        "Precipitation": precipitation,
        "CloudCover": cloud_cover,
        "Temperature": temperature,
        "Snowfall": snowfall,
        "Wind": wind,
        "Year": year,
        "hour": hour,
        "month": month,
        "Day": day
    }
    input_df = pd.DataFrame(features_dict,index=[0])
    # Index(['Global_reactive_power', 'Voltage', 'Sub_metering_1', 'Sub_metering_2',
    #    'Sub_metering_3', 'Precipitation', 'CloudCover', 'Temperature',
    #    'Snowfall', 'Wind', 'Year', 'hour', 'month', 'Day'],
    #   dtype='object')

#  ["Random Forest", "Sarimax", "Linear Regression", "XGBoost"]
    loaded_model = None
    try:
        if model == "Random Forest":
            with open('models/RFmodel.pkl',"rb") as file:
                loaded_model = pk.load(file)
        elif model == "Sarimax":
            with open('models/Sarimaxmodel.pkl',"rb") as file:
                loaded_model = pk.load(file)
        elif model == "Linear Regression":
            with open('models/Linearmodel.pkl',"rb") as file:
                loaded_model = pk.load(file)
        elif model == "XGBoost":
            with open('models/XGBmodel.pkl',"rb") as file:
                loaded_model = pk.load(file)
    except FileNotFoundError:
        st.error(f"Model file not found inside the 'models/' directory for {model}!")

    if loaded_model is not None:
        
        pred = loaded_model.predict(input_df)
        
        # 5. Display the prediction cleanly using st.metric
        predicted_value = round(float(pred[0]), 4)
        st.metric(label="Predicted Global Active Power", value=f"{predicted_value} kW")
    

    
    # st.write("Processed inputs ready for inference:", features_dict)
    # Your prediction function code goes here:
    # prediction = your_model.predict([list(features_dict.values())])