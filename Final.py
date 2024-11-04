#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install pandas numpy matplotlib ta scikit-learn tensorflow statsmodels


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, LSTM
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs


# In[3]:


# Utility Functions
def load_data(historical_paths, sentiment_paths):
    historical_data = []
    sentiment_data = []
    
    for hist_path, senti_path in zip(historical_paths, sentiment_paths):
        try:
            data = pd.read_excel(hist_path)
            senti = pd.read_csv(senti_path)
            historical_data.append(data)
            sentiment_data.append(senti)
        except Exception as e:
            print(f"Error loading data from {hist_path} or {senti_path}: {e}")
    
    return historical_data, sentiment_data


# In[4]:


def extract_weights(weights_file, stock_names):
    default_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    try:
        weights_df = pd.read_excel(weights_file)
        stock_weights = {}
          # Corrected closing bracket
        relevant_weights_df = weights_df[weights_df['Stock'].isin(stock_names)]
                                    
        for name in stock_names:
            weight_value = relevant_weights_df.loc[relevant_weights_df['Stock'] == name, 'Weights'].values
            
            if weight_value.size == 0:
                print(f"Warning: No weight found for {name}, using {default_weights}")
                stock_weights[name] = default_weights  # Default weights if not found
                continue
                
            weight_value = weight_value[0]
            
            if isinstance(weight_value, str):
                weight_value = weight_value.strip(" '\"")
                if weight_value.startswith('[') and weight_value.endswith(']'):
                    # Convert string representation of array to numpy array
                    weights_array = np.array([float(w.strip()) for w in weight_value.strip("[]").split(',')])
                    
                    # Ensure we have 5 weights
                    if len(weights_array) != 5:
                        print(f"Warning: {name} has {len(weights_array)} weights instead of 5. Padding with zeros.")
                        weights_array = np.pad(weights_array, (0, 5 - len(weights_array)), mode='constant')
                    
                    stock_weights[name] = weights_array
                else:
                    print(f"Warning: {name} has single weight value, expanding to array.")
                    # If single value, create array with that value repeated
                    stock_weights[name] = np.array([float(weight_value)] * 5)
            else:
                print(f"Warning: {name} has single weight value, expanding to array.")
                # If single value, create array with that value repeated
                stock_weights[name] = np.array([float(weight_value)] * 5)
            
            # Normalize weights for each stock to sum to 1
            if np.sum(stock_weights[name]) != 0:
                stock_weights[name] = stock_weights[name] / np.sum(stock_weights[name])
            else:
                print(f"Warning: {name} has all zero weights, using equal weights.")
                stock_weights[name] = default_weights
        
        return stock_weights
        
    except Exception as e:
        print(f"Error processing weights file: {e}")
        # Return equal weights as fallback
        return {name: default_weights for name in stock_names}  # Corrected to use default_


# In[5]:


def add_technical_indicators(df):
    df['SMA20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['SMA50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['SMA100'] = ta.trend.sma_indicator(df['close'], window=100)
    df['SMA200'] = ta.trend.sma_indicator(df['close'], window=200)
    df['EMA20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['EMA50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['EMA100'] = ta.trend.ema_indicator(df['close'], window=100)
    df['EMA200'] = ta.trend.ema_indicator(df['close'], window=200)
    df['MOM14'] = ta.momentum.roc(df['close'], window=14)
    df['STCK14'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['STCD14'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['MACD'], df['MACD_signal'], df['MACD_diff'] = ta.trend.MACD(df['close']).macd(), \
                                                     ta.trend.MACD(df['close']).macd_signal(), \
                                                     ta.trend.MACD(df['close']).macd_diff()
    df['RSI14'] = ta.momentum.rsi(df['close'], window=14)
    df['CCI14'] = ta.trend.cci(df['high'], df['low'], df['close'], window=14)
    
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    
    return df




# In[6]:


def perform_pca(df):
    features = ['open', 'high', 'low', 'SMA20', 'SMA50', 'SMA100', 'SMA200', 
                'EMA20', 'EMA50', 'EMA100', 'EMA200', 'MOM14', 'STCK14', 'STCD14', 
                'MACD', 'MACD_signal', 'MACD_diff', 'RSI14', 'CCI14']

    x = df[features].values
    x = StandardScaler().fit_transform(x)
    
    pca = PCA(n_components=len(features))
    principal_components = pca.fit_transform(x)
    
    loadings = pd.DataFrame(pca.components_.T, columns=[features[i] for i in range(len(features))], index=features)
    abs_loadings = loadings.abs().sum(axis=1)
    top_features = abs_loadings.nlargest(10)
    
    return top_features


# In[7]:


def feature_target(df, top_features):
    features = top_features.index.tolist()
    target = 'close'
    
    # Check and add 'low', 'open', 'high' if they're not in features
    for feature in ['low', 'open', 'high']:
        if feature not in features:
            features.append(feature)
    
    # Create the DataFrame with the selected features and target
    df = df[features + [target]]
    X = df[features].values
    y = df[target].values
    
    return X, y, features, df


# In[8]:


def ANN(X,y):
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    model = Sequential()
    model.add(Dense(units=128,activation='relu',input_dim=X_train.shape[1]))
    model.add(Dense(units=64,activation = 'relu'))
    model.add(Dense(units=1))
    
    model.compile(optimizer=Adam(learning_rate=0.001),loss = 'mean_squared_error')
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    loss = model.evaluate(X_train, y_train)
    predictions = model.predict(X_test)
    
    return predictions

def HW(df,features,X,y):
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    hw_model = ExponentialSmoothing(df['close'], 
                                 trend='add', 
                                 seasonal='add', 
                                 seasonal_periods=30).fit()
    hw_forecast = hw_model.fittedvalues
    
    residuals = df['close'] - hw_forecast
    
#     print(features)

    scaler = StandardScaler()
    
    features_scaled = scaler.fit_transform(df[features])

    model = Sequential([
        Dense(128, activation='relu', input_shape=(features_scaled.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1)  
    ])
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(features_scaled, residuals, epochs=100, batch_size=32, verbose=0)
    
    ann_predictions = model.predict(features_scaled)
    
    final_predictions = hw_forecast + ann_predictions.flatten()
    
    final_predictions= final_predictions[train_size:]
    
    return final_predictions

def create_sequences_lstm(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])  
    return np.array(X_seq), np.array(y_seq)

def lstm(X, y):
    scalerX = MinMaxScaler(feature_range=(0,1))
    scalerY = MinMaxScaler(feature_range=(0,1))

    X_scaled = scalerX.fit_transform(X)
    y_scaled = scalerY.fit_transform(y.reshape(-1,1)).ravel()

    seq_length = 1
    X_seq, y_seq = create_sequences_lstm(X_scaled, y_scaled, seq_length)

    train_size = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:train_size], X_seq[train_size:]
    y_train, y_test = y_seq[:train_size], y_seq[train_size:]

    model = Sequential()

    model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=64))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

    predictions = model.predict(X_test)
    
    predictions = scalerY.inverse_transform(predictions)
    y_test = scalerY.inverse_transform(y_test.reshape(-1, 1))
    
    return predictions

def create_sequences_rnn(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, 0])  # Assuming the target is the first column
    return np.array(X), np.array(y)

def rnn(df):
    # Prepare the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    
    # Define sequence length
    sequence_length = 1
    X, y = create_sequences_rnn(scaled_data, sequence_length)
    
    # Train-test split
    split_index = int(0.8 * len(df['close']))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Build the RNN model
    model = Sequential()
    model.add(SimpleRNN(70, activation='relu', input_shape=(sequence_length, X_train.shape[2])))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # EarlyStopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, callbacks=[early_stopping])
    
    # Predict on test data
    predictions = model.predict(X_test)
    
    # Inverse transform predictions to match the original data range
    predictions = predictions.reshape(-1, 1)
    zeros_for_other_features = np.zeros((predictions.shape[0], df.shape[1] - 1))
    scaled_predictions = np.concatenate((predictions, zeros_for_other_features), axis=1)
    inverse_predictions = scaler.inverse_transform(scaled_predictions)[:, 0]
    
    # Scale predictions to be within the range 2200-3000
    min_val, max_val = 2200, 3000
    adjusted_predictions = min_val + (inverse_predictions - inverse_predictions.min()) * (max_val - min_val) / (inverse_predictions.max() - inverse_predictions.min())
    
    return adjusted_predictions.reshape(-1, 1)


def sarimax(df, features):

    features = df[features]

    split_index = int(0.8 * len(df['close']))

    model = ARIMA(df['close'][:split_index], order=(0, 1, 0), exog=features[:split_index]).fit()

    predictions = model.predict(start=split_index, end=len(df)-1, exog=features[split_index:])

    return predictions


# In[9]:


import numpy as np
from sklearn.metrics import r2_score

def optimize_predictions(data, final_predictions, sentiment_values, k_range=(-1, 1)):
    r2_results = {}
    # Debugging: Check types and contents of inputs
#     print("Type of final_predictions:", type(final_predictions))
#     print("Contents of final_predictions:", final_predictions)
#     print("Type of sentiment_values:", type(sentiment_values))
#     print("Contents of sentiment_values:", sentiment_values)
    
    # Convert inputs to numpy arrays
    try:
        final_predictions = np.array(final_predictions)
        sentiment_values = np.array(sentiment_values)
#         print("Converted final_predictions to numpy array:", final_predictions)
#         print("Converted sentiment_values to numpy array:", sentiment_values)
    except Exception as e:
        print("Error converting to numpy arrays:", str(e))
        return None

    # Check for NaN values in final_predictions
    if np.any(np.isnan(final_predictions)):
        print("Final predictions contain NaN values. Please check the model outputs.")
        print("NaN indices in final_predictions:", np.where(np.isnan(final_predictions)))
        return None

    # Check the shape of inputs
#     print("Shape of final_predictions:", final_predictions.shape)
#     print("Shape of sentiment_values:", sentiment_values.shape)
    
    # Ensure both arrays have the same length
    if final_predictions.shape[0] != sentiment_values.shape[0]:
        print("Mismatch in lengths: final_predictions length:", final_predictions.shape[0],
              "sentiment_values length:", sentiment_values.shape[0])
        return None

    # Calculate R² score while handling NaN values
    for alpha in np.linspace(*k_range):
        try:
            # Element-wise multiplication is now possible with numpy arrays
            adjusted_predictions = final_predictions * (1 + alpha * sentiment_values)
#             print(f"Adjusted predictions with alpha {alpha}:", adjusted_predictions)

            # Make sure 'data' has no NaN values for 'close' column and we only compare with valid predictions
            y_true = data['close'].iloc[-265:].dropna()
            y_pred = adjusted_predictions[~np.isnan(adjusted_predictions)]
            
            # Align y_true and y_pred
            min_length = min(len(y_true), len(y_pred))
            if min_length == 0:
                print("No valid predictions available to compute R² score.")
                continue
            
            # Calculate R² score
            r2 = r2_score(y_true.iloc[:min_length].values, y_pred[:min_length])
            r2_results[alpha] = r2
#             print(f"R² score for alpha {alpha}: {r2}")

        except Exception as e:
            print("Error during R² score calculation:", str(e))
            continue

    # Find the best alpha and adjust predictions accordingly
    if r2_results:
        best_alpha = max(r2_results, key=r2_results.get)
        print(f"Best alpha: {best_alpha}")
        
        # Apply the best alpha to final_predictions
        best_adjusted_predictions = final_predictions * (1 + best_alpha * sentiment_values)
#         print("Best adjusted predictions:", best_adjusted_predictions)
        return best_adjusted_predictions
    
    print("No valid R² results found.")
    return None



# In[10]:


# Prediction Functions
def make_predictions(data, features, X, y):
    ann_predictions = ANN(X, y)
    hw_predictions = HW(data, features, X, y)
    lstm_predictions = lstm(X, y)
    rnn_predictions = rnn(data)    
    sarimax_predictions = sarimax(data, features).values

    return (np.array(ann_predictions, dtype=float),
            np.array(hw_predictions, dtype=float),
            np.array(lstm_predictions, dtype=float),
            np.array(rnn_predictions, dtype=float),
            np.array(sarimax_predictions, dtype=float))



# In[11]:


def aggregate_predictions(predictions, weights):
    if len(predictions) != len(weights):
        raise ValueError("The number of predictions and weights must match.")
    
    return [sum(weight * pred for weight, pred in zip(weights, preds)) for preds in zip(*predictions)]


# In[12]:


def generate_recommendations(predictions, historical_prices, threshold=0.05):
    # Calculate the median of the last 15 days of historical prices
    recent_prices = historical_prices.iloc[-15:]
    median_price = np.median(recent_prices)
    
    # Calculate the median of the last 15 predictions for consistency
    recent_predictions = predictions[-15:]
    median_prediction = np.median(recent_predictions)
    
    # Calculate the percentage change from the median price to the median prediction
    percent_change = (median_prediction - median_price) / median_price

    # Generate recommendation based on the percent change
    if percent_change > threshold:
        return "Buy"
    elif percent_change < -threshold:
        return "Sell"
    else:
        return "Hold"


# In[13]:


def prepare_future_dates(df, num_days, top_features, start_date=None):
    # Ensure the index is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index, errors='coerce')

    # Set start_date to the provided date or the day after the last date in df
    if start_date is not None:
        start_date = pd.to_datetime(start_date, errors='coerce')  # Ensure start_date is in datetime format
    else:
        start_date = df.index[-1] + pd.Timedelta(days=1)

    # Create a future date range
    future_dates = pd.date_range(start=start_date, periods=num_days, freq='B')
    future_df = pd.DataFrame(columns=list(top_features) + ['close'], index=future_dates)
    return future_df

def adjust_future_prices(df, future_df, senti, date):
    # Prepare and parse sentiment DataFrame with more flexible date parsing
    senti = senti[['date', 'positive', 'negative']]
    senti['date'] = pd.to_datetime(senti['date'], errors='coerce')  # Use coerce to handle parsing issues
    senti.set_index('date', inplace=True)
    
    # Calculate sentiment value
    senti['sentiment_value'] = senti['positive'] - (2 * senti['negative'])

    # Resample and fill missing sentiment values
    senti = senti.resample('D').median()
    senti['final_value'] = senti['sentiment_value'].fillna(method='ffill').fillna(method='bfill')

    # Get sentiment factor for the given date, defaulting to 0 if date not found
    sentiment_value = senti['final_value'].get(date, 0)

    base_close = df['close'].iloc[-1]
    future_row = {
        'open': base_close * (1 + sentiment_value),
        'high': base_close * (1 + sentiment_value * 2),
        'low': base_close * (1 - sentiment_value * 2),
        'close': base_close * (1 + sentiment_value)
    }
    
    return future_row

def update_indicators(future_row, df, top_features):
    for feature in top_features.index:
        if feature.startswith('SMA'):
            future_row[feature] = ta.trend.sma_indicator(df['close'], window=int(feature[3:])).iloc[-1]
        elif feature.startswith('EMA'):
            future_row[feature] = ta.trend.ema_indicator(df['close'], window=int(feature[3:])).iloc[-1]
        elif feature == 'MOM14':
            future_row[feature] = ta.momentum.roc(df['close'], window=14).iloc[-1]
        elif feature == 'RSI14':
            future_row[feature] = ta.momentum.rsi(df['close'], window=14).iloc[-1]
        elif feature == 'CCI14':
            future_row[feature] = ta.trend.cci(df['high'], df['low'], df['close'], window=14).iloc[-1]
        elif feature in ['STCK14', 'STCD14']:
            future_row[feature] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3).iloc[-1]
        elif feature in ['MACD', 'MACD_signal', 'MACD_diff']:
            macd_indicator = ta.trend.MACD(df['close'])
            future_row['MACD'] = macd_indicator.macd().iloc[-1]
            future_row['MACD_signal'] = macd_indicator.macd_signal().iloc[-1]
            future_row['MACD_diff'] = macd_indicator.macd_diff().iloc[-1]
    return future_row

def future_predictions(df, X, y, weights, top_features, stock_index):

    # Get predictions from different models
    ann_predictions = ANN(X, y)
    hw_predictions = HW(df, top_features.index.tolist(), X, y)
    lstm_predictions = lstm(X, y)
    rnn_predictions = rnn(df)
    sarimax_predictions = sarimax(df, top_features.index.tolist()).values

    # Ensure each prediction is a list/array, or wrap it if scalar
    for pred in [ann_predictions, hw_predictions, lstm_predictions, rnn_predictions, sarimax_predictions]:
        if isinstance(pred, (float, int)):
            pred = [pred]

    # Select the appropriate row of weights for the current stock
    stock_weights = weights[stock_index]

    # Calculate final predictions with last elements of each prediction list
    final_predictions = (
        float(stock_weights[0]) * ann_predictions[-1] +
        float(stock_weights[1]) * hw_predictions[-1] +
        float(stock_weights[2]) * lstm_predictions[-1] +
        float(stock_weights[3]) * rnn_predictions[-1] +
        float(stock_weights[4]) * sarimax_predictions[-1]
    )

    # Create future row based on last row in df
    future_row = df.iloc[-1].copy()
    future_row['close'] = final_predictions
    return future_row



# In[14]:


def historical_analysis(historical_paths, sentiment_paths, weights_file, stock_names):
    """Endpoint 1: Historical analysis and prediction"""
    if len(historical_paths) != len(sentiment_paths) or len(historical_paths) != len(stock_names):
        raise ValueError("Number of paths and stock names must match")
        
    historical_data, sentiment_data = load_data(historical_paths, sentiment_paths)
    stock_weights = extract_weights(weights_file, stock_names)
    
    final_predictions = []
    recommendations = []
    
    for stock_name, data, senti in zip(stock_names, historical_data, sentiment_data):
        try:
            # Process sentiment data with more robust date parsing
            senti = senti[['date', 'positive', 'negative']]
            
            # More robust date parsing
            try:
                senti['date'] = pd.to_datetime(senti['date'], format='%d-%b-%Y')
            except ValueError:
                try:
                    senti['date'] = pd.to_datetime(senti['date'], format='%d-%b-%y')
                except ValueError:
                    senti['date'] = pd.to_datetime(senti['date'], errors='coerce', dayfirst=True)
            
            senti.set_index('date', inplace=True)
            senti['sentiment_value'] = senti['positive'] - (2 * senti['negative'])
            senti = senti.resample('D').median()
            senti['final_value'] = senti['sentiment_value'].fillna(method='ffill').fillna(method='bfill')
            
            # Ensure we have enough sentiment values
            sentiment_values = senti['final_value'].values
            if len(sentiment_values) < 265:
                # Pad with zeros or the mean if needed
                pad_length = 265 - len(sentiment_values)
                sentiment_values = np.pad(sentiment_values, (0, pad_length), 
                                       mode='constant', 
                                       constant_values=sentiment_values.mean())
            sentiment_values = sentiment_values[-265:]  # Take last 265 values
            
            # Process historical data
            data = add_technical_indicators(data)
            features = perform_pca(data)
            X, y, features, data = feature_target(data, features)
            
            model_predictions = make_predictions(data, features, X, y)
            
            # Use the weight from dictionary
            model_weight = stock_weights.get(stock_name, 0.2)  # Default 0.2 if not found
            weights = np.array([model_weight] * 5)  # Create weights array for models
            
            final_prediction = aggregate_predictions(model_predictions, weights)
            final_predictions.append(final_prediction)
            
            # Ensure predictions array is the right length
            final_predictions_flat = np.array(final_predictions).flatten()
            if len(final_predictions_flat) < 265:
                pad_length = 265 - len(final_predictions_flat)
                final_predictions_flat = np.pad(final_predictions_flat, (0, pad_length),
                                             mode='edge')
            final_predictions_flat = final_predictions_flat[-265:]
            
            adjusted_predictions = optimize_predictions(data, final_predictions_flat, sentiment_values)
            recommendation = generate_recommendations(final_prediction, data['close'])
            recommendations.append(recommendation)
            
        except Exception as e:
            print(f"Error processing {stock_name}: {str(e)}")
            print(f"Traceback for {stock_name}:")
            import traceback
            traceback.print_exc()
            recommendations.append("Hold")  # Default recommendation
            # Add neutral/zero predictions if error occurs
            if not final_predictions:
                final_predictions.append(np.zeros(265))
            
    # Ensure we have predictions even if some stocks failed
    if not final_predictions:
        final_predictions = [np.zeros(265)]
    
    adjusted_predictions = np.array(final_predictions[-1])  # Take the last set of predictions
    
    return adjusted_predictions, recommendations


# In[15]:


def future_prediction1(historical_paths, sentiment_paths, weights_file, stock_names, num_days=2, start_date=None):
    historical_data, sentiment_data = load_data(historical_paths, sentiment_paths)
    stock_weights = extract_weights(weights_file, stock_names)
    weights = np.array([stock_weights.get(name, 0.0) for name in stock_names], dtype=float)

    future_predictions_dict = {}

    for stock_index, (stock_name, data, senti) in enumerate(zip(stock_names, historical_data, sentiment_data)):
        # Change 1: Move technical indicators calculation outside the loop
        data = add_technical_indicators(data)  # Optimized: calculate once per stock
        top_features = perform_pca(data)  # Optimized: calculate once per stock
        X, y, features, data = feature_target(data, top_features)  # Optimized: calculate once per stock

        future_df = prepare_future_dates(data, num_days, features, start_date)

        # Change 2: Removed print statement for each date to improve performance
        for date in future_df.index:
            future_row = adjust_future_prices(data, future_df, senti, date)
            future_row = update_indicators(future_row, data, top_features)
            future_row = future_predictions(data, X, y, weights, top_features, stock_index)
            future_df.loc[date] = future_row

        future_predictions_dict[stock_name] = future_df

    return future_predictions_dict


# In[16]:


def validate_and_adjust_recommendations(historical_prices, recommendations, future_predictions, stock_names, threshold=0):
    adjusted_recommendations = []

    for index, recommendation in enumerate(recommendations):
        stock_name = stock_names[index]
        
        # Get historical median and future median prices for validation
        historical_median = np.median(historical_prices[index][-10:])  # Adjusted for list of historical prices
        future_median = future_predictions[stock_name]['close'].iloc[:10].median()

        # Adjust recommendations based on comparison with threshold
        if future_median > historical_median * (1 + threshold):
            adjusted_recommendations.append("Buy")
            print(f"Adjusted recommendation for {stock_name}: Changed from {recommendation} to 'Buy'")
        elif future_median < historical_median * (1 - threshold):
            adjusted_recommendations.append("Sell")
            print(f"Adjusted recommendation for {stock_name}: Changed from {recommendation} to 'Sell'")
        else:
            adjusted_recommendations.append(recommendation)  # Keep original recommendation if no adjustment is needed
            print(f"Adjusted recommendation for {stock_name}: Remained as '{recommendation}'")

    # Final call: choose the last adjusted recommendation
    final_recommendation = adjusted_recommendations[-1]
    print(f"Final Recommendation for {stock_names[-1]}: Original '{recommendations[-1]}', Adjusted '{final_recommendation}'")
    
    return adjusted_recommendations  # Return only the final recommendation


# In[17]:


def main():
    historical_paths = [
        # r"NSE500 Dataset/BAJAJFINSV.xlsx",
        # r"NSE500 Dataset/HAL.xlsx",
        r"NSE500 Dataset/HINDUNILVR.xlsx",
        r"NSE500 Dataset/ICICIBANK.xlsx",
        r"NSE500 Dataset/MRF.xlsx",
        r"NSE500 Dataset/RELIANCE.xlsx",
        r"NSE500 Dataset/SIEMENS.xlsx",
        r"NSE500 Dataset/SUNPHARMA.xlsx",
        r"NSE500 Dataset/TCS.xlsx"
    ]
    
    sentiment_paths = [
        # r"sentiment/bfinserv.csv",
        # r"sentiment/hal.csv",
        r"sentiment/hul.csv",
        r"sentiment/icici.csv",
        r"sentiment/mrf.csv",
        r"sentiment/reliance.csv",
        r"sentiment/siemens.csv",
        r"sentiment/sunpharma.csv",
        r"sentiment/tcs.csv"
    ]
    
    weights_file = r"weighted_average_r2.xlsx"
    
    stock_names = [
                    # 'BAJAJFINSV',
                   # 'HAL',
                   'HINDUNILVR', 
                   'ICICIBANK',
                   'MRF',
                   'RELIANCE',
                   'SIEMENS',
                   'SUNPHARMA',
                   'TCS']
    
    if not (len(historical_paths) == len(sentiment_paths) == len(stock_names)):
        raise ValueError("Number of historical paths, sentiment paths, and stock names must be equal")

    # Run historical analysis
    predictions, recommendations = historical_analysis(
        historical_paths, sentiment_paths, weights_file, stock_names
    )
    
    print(recommendations)

    # Run future prediction
    future_predictions1 = future_prediction1(
        historical_paths, sentiment_paths, weights_file, stock_names,
        num_days=1, start_date='2024-06-14'
    )
    
    # Validate and adjust recommendations if needed
    adjusted_recommendations = validate_and_adjust_recommendations(
        historical_prices=predictions,  # Use the correct historical prices DataFrame
        recommendations=recommendations,
        future_predictions=future_predictions1,
        stock_names = stock_names
    )
    return recommendations,adjusted_recommendations

if __name__ == "__main__":
    x,y = main()
    print(x)
    print(y)


# In[ ]:





# In[ ]:




