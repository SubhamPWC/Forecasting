import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from prophet import Prophet
import torch
import torch.nn as nn
from datetime import timedelta
import io

st.set_page_config(layout="wide")
st.title("🔮 Power Demand Forecast with Weather & Calendar Features")

# 📤 Upload Files
demand_file = st.file_uploader("Upload Demand File", type=["xlsx"])
weather_file = st.file_uploader("Upload Weather File", type=["xlsx"])
calendar_file = st.file_uploader("Upload Calendar File", type=["xlsx"])

# 🔧 Model Selection
model_list = [
     "ARIMA", "SARIMAX", "VAR", "VARMA",
    "RandomForest", "XGBoost", "SVR",
    "RNN", "LSTM", "GRU", "TCN", "Transformer",
    "Prophet", "BSTS"
]
selected_model = st.sidebar.selectbox("Choose Forecasting Model", model_list)

if demand_file and weather_file and calendar_file:
    # 📊 Load Data
    demand_df = pd.read_excel(demand_file)
    weather_df = pd.read_excel(weather_file)
    calendar_df = pd.read_excel(calendar_file)

    # 🧩 Preprocess & Merge
    demand_df['Date'] = pd.to_datetime(demand_df['Date'])
    demand_df['Hour'] = pd.to_datetime(demand_df['Hour'], format='%H:%M:%S').dt.hour
    demand_df['Datetime'] = demand_df['Date'] + pd.to_timedelta(demand_df['Hour'], unit='h')
    weather_df['Datetime'] = pd.to_datetime(weather_df['Date'].astype(str) + ' ' + weather_df['Time'].astype(str))
    calendar_df['Date'] = pd.to_datetime(calendar_df['Date'], errors='coerce')

    merged_df = demand_df.merge(weather_df, on=['State', 'Datetime'], how='left')
    merged_df['Date'] = merged_df['Datetime'].dt.date  # Extract date part from datetime
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])  # Convert to datetime64
    merged_df = merged_df.merge(calendar_df, on=['State', 'Date'], how='left')

    # 🧠 Feature Engineering
    def generate_features(df):
        df = df.sort_values('Datetime')
        df['Hour'] = df['Datetime'].dt.hour
        df['DayOfWeek'] = df['Datetime'].dt.dayofweek
        df['Month'] = df['Datetime'].dt.month
        df['Lag_1'] = df['Demand'].shift(1)
        df['Lag_24'] = df['Demand'].shift(24)
        df['RollingMean_3'] = df['Demand'].rolling(3).mean()
        df['RollingStd_3'] = df['Demand'].rolling(3).std()

        for col in ['Temperature_2m', 'DNI', 'Relative_humidity_2m', 'Dew_point_2m',
                    'Apparent_temperature', 'Rain', 'Cloud_cover']:
            if col in df.columns:
                df[f'{col}_Lag1'] = df[col].shift(1)
                df[f'{col}_Diff'] = df[col].diff()

        df = df.dropna(subset=['Demand'])  # Only drop if Demand is missing
        df = df.fillna(method='ffill').fillna(method='bfill')  # Fill other missing values
        return df

    feature_df = generate_features(merged_df)

    # 📍 Select State
    state = st.selectbox("Select State", feature_df['State'].unique())
    state_df = feature_df[feature_df['State'] == state]

    # 📈 Forecasting Setup
    exclude_cols = ['State', 'Datetime', 'Demand']
    X = state_df.drop(columns=exclude_cols, errors='ignore')
    y = state_df['Demand']

    # ✅ Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    # 🧼 Handle missing values
    X = X.fillna(X.mean())

    # ❗ Check if X is empty
    if X.empty or len(X) == 0:
        st.error("❌ Feature matrix X is empty after cleaning. Please check your data or feature generation.")
        st.stop()

    # ✅ Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 🔀 Train-test split
    split = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]


    # 🔮 Model Training
    if selected_model == "RandomForest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        forecast = model.predict(X_test)
    elif selected_model == "XGBoost":
        model = XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        forecast = model.predict(X_test)
    elif selected_model == "SVR":
        model = SVR(kernel='rbf')
        model.fit(X_train, y_train)
        forecast = model.predict(X_test)
    elif selected_model == "ARIMA":
        model = ARIMA(y_train, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(y_test))
    elif selected_model == "SARIMAX":
        model = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=len(y_test))
    elif selected_model == "VAR":
        df_var = state_df[['Demand']].copy()
        df_var['Lag1'] = df_var['Demand'].shift(1)
        df_var.dropna(inplace=True)
        model = VAR(df_var)
        model_fit = model.fit()
        forecast = model_fit.forecast(model_fit.y, steps=len(y_test))[:, 0]
    elif selected_model == "Prophet":
        prophet_df = state_df[['Datetime', 'Demand']].rename(columns={'Datetime': 'ds', 'Demand': 'y'})
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=len(y_test), freq='H')
        forecast_df = model.predict(future)
        forecast = forecast_df['yhat'].tail(len(y_test)).values
    elif selected_model == "BSTS":
        st.warning("BSTS not implemented. Placeholder only.")
        forecast = np.full_like(y_test, y_train.mean())
    elif selected_model in ["RNN", "LSTM", "GRU", "TCN", "Transformer"]:
        class TimeSeriesModel(nn.Module):
            def __init__(self, model_type):
                super().__init__()
                if model_type == "LSTM":
                    self.rnn = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
                elif model_type == "GRU":
                    self.rnn = nn.GRU(input_size=1, hidden_size=50, batch_first=True)
                elif model_type == "RNN":
                    self.rnn = nn.RNN(input_size=1, hidden_size=50, batch_first=True)
                else:
                    self.rnn = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
                self.fc = nn.Linear(50, 1)

            def forward(self, x):
                out, _ = self.rnn(x)
                out = out[:, -1, :]
                return self.fc(out)

        def create_sequences(data, window=5):
            X, y = [], []
            for i in range(window, len(data)):
                X.append(data[i-window:i])
                y.append(data[i])
            return np.array(X), np.array(y)

        window = 5
        scaled_series = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
        X_seq, y_seq = create_sequences(scaled_series, window)
        split = int(len(X_seq) * 0.8)
        X_train_seq = torch.tensor(X_seq[:split], dtype=torch.float32).unsqueeze(-1)
        y_train_seq = torch.tensor(y_seq[:split], dtype=torch.float32).unsqueeze(-1)
        X_test_seq = torch.tensor(X_seq[split:], dtype=torch.float32).unsqueeze(-1)

        model = TimeSeriesModel(selected_model)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            output = model(X_train_seq)
            loss = criterion(output, y_train_seq)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            forecast_scaled = model(X_test_seq).squeeze().numpy()
        forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
    else:
        st.error("Unsupported model selected.")
        st.stop()

    # 📊 Metrics
    rmse = np.sqrt(mean_squared_error(y_test, forecast))
    st.subheader("📊 Model Performance")
    st.write(f"**RMSE**: {rmse:.2f}")
    st.write(f"**MAE**: {mean_absolute_error(y_test, forecast):.2f}")
    st.write(f"**R² Score**: {r2_score(y_test, forecast):.2f}")


    # 🔮 Predict Next 245 Days (Hourly)
    st.subheader("📅 Predict Next 245 Days (Hourly)")
    last_datetime = state_df['Datetime'].max()
    future_dates = [last_datetime + timedelta(hours=i) for i in range(1, 245 * 24 + 1)]
    future_df = pd.DataFrame({'Datetime': future_dates})
    future_df['State'] = state
    future_df['Hour'] = future_df['Datetime'].dt.hour
    future_df['DayOfWeek'] = future_df['Datetime'].dt.dayofweek
    future_df['Month'] = future_df['Datetime'].dt.month

    # Fill with historical averages (simplified)
    for col in X.columns:
        future_df[col] = X[col].mean()

    # Forecast future demand
    if selected_model in ["RandomForest", "XGBoost", "SVR", "LinearRegression"]:
        future_scaled = scaler.transform(future_df[X.columns])
        future_forecast = model.predict(future_scaled)
    elif selected_model == "Prophet":
        future_prophet = pd.DataFrame({'ds': future_df['Datetime']})
        future_forecast_df = model.predict(future_prophet)
        future_forecast = future_forecast_df['yhat'].values
    elif selected_model in ["ARIMA", "SARIMAX", "VAR"]:
        future_forecast = np.full(len(future_df), forecast.iloc[-1])  # Extend last value
    elif selected_model in ["RNN", "LSTM", "GRU", "TCN", "Transformer"]:
        future_scaled = scaler.transform(future_df[X.columns])
        future_seq = []
        for i in range(len(future_scaled) - window):
            future_seq.append(future_scaled[i:i+window])
        future_seq_tensor = torch.tensor(future_seq, dtype=torch.float32).unsqueeze(-1)
        with torch.no_grad():
            future_scaled_pred = model(future_seq_tensor).squeeze().numpy()
        future_forecast = scaler.inverse_transform(future_scaled_pred.reshape(-1, 1)).flatten()
    else:
        future_forecast = np.full(len(future_df), y.mean())

    future_df['Forecasted_Demand'] = future_forecast

    # 📉 Preview Chart
    st.subheader("📈 Forecast Preview (First 2 Days)")
    st.line_chart(future_df.set_index('Datetime')['Forecasted_Demand'].head(48))

    # 📁 Export CSV
    st.subheader("📁 Export Forecast")
    csv_buffer = io.StringIO()
    future_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv_buffer.getvalue(),
        file_name=f"{state}_245_day_forecast.csv",
        mime="text/csv"

    )















