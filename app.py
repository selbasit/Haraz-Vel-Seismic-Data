
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

st.set_page_config(layout="wide", page_title="Haraz-Vel Velocity Modeling")

@st.cache_data
def load_data():
    df = pd.read_csv('Haraz-vel-final.csv', delim_whitespace=True)
    df = df.dropna().drop_duplicates()
    return df

df = load_data()

features = ['Inline', 'Xline', 'CDP_X', 'CDP_Y', 'Time']
target = 'RMS_V'

if not all(f in df.columns for f in features + [target]):
    st.error("Required columns are missing from the dataset.")
    st.stop()

X = df[features].values
y = df[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

nn = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
nn.compile(optimizer='adam', loss='mse', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
nn.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2,
       callbacks=[early_stop], verbose=0)
y_pred_nn = nn.predict(X_test_scaled).flatten()

st.header("📊 Model Evaluation")
st.metric("Random Forest RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f} m/s")
st.metric("Random Forest R²", f"{r2_score(y_test, y_pred_rf):.2f}")
st.metric("Neural Network RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred_nn)):.2f} m/s")
st.metric("Neural Network R²", f"{r2_score(y_test, y_pred_nn):.2f}")

if st.button('💾 Save Models'):
    joblib.dump({'rf': rf, 'nn': nn, 'scaler': scaler}, 'haraz_vel_models.pkl')
    st.success("Models saved successfully as 'haraz_vel_models.pkl'.")

st.header("🧮 Predict RMS Velocity")
input_data = []
for col in features:
    val = st.number_input(col, value=float(df[col].mean()))
    input_data.append(val)

input_scaled = scaler.transform([input_data])
rf_prediction = rf.predict(input_scaled)[0]
nn_prediction = nn.predict(input_scaled).flatten()[0]

st.success(f"Random Forest Prediction: **{rf_prediction:.2f} m/s**")
st.success(f"Neural Network Prediction: **{nn_prediction:.2f} m/s**")
