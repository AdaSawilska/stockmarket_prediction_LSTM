import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from keras import optimizers
from keras.layers import Dense, LSTM, Input, Activation
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_finance_data():
    """
    function which loads the stock market data from yahoo finance
    :return: pd.Dataframe
    """
    data = yf.download(tickers='^RUI', start='2012-03-11', end='2022-07-10')
    print(data.head())
    return data


def add_indicators(data, all_ind=False):
    """
    function which adds indicators to stock market data
    :param data: pd.Dataframe
    :param all_ind: bool
    :return: pd.Dataframe
    """
    if not all_ind:
        data['RSI'] = ta.rsi(data.Close, length=15)  # Relative Strength Index
        data['EMAF'] = ta.ema(data.Close, length=20)  # Exponential Moving Average Fast
        data['EMAM'] = ta.ema(data.Close, length=100)  # Exponential Moving Average Medium
        data['EMAS'] = ta.ema(data.Close, length=150)  # Exponential Moving Average Slow
        data['SMA'] = ta.sma(data.Close, length=20)  # Simple Moving Average
        data['PSL'] = ta.psl(data.Close, length=20)  # Psychological Line
        data['ROC'] = ta.roc(data.Close, length=20)  # Rate Of Change
        data['FWMA'] = ta.fwma(data.Close, length=20)  # Fibonacci's Weighted Moving Average
        data['entropy'] = ta.entropy(data.Close, length=30)  # Entropy
        data['zscore'] = ta.zscore(data.Close, length=20)  # ZScore
        macd = ta.macd(data.Close, fast=12, slow=26, signal=9)  # Moving Average Convergence Divergence
        data = data.join(macd)
        bb = ta.bbands(data.Close, length=20, std=2)  # Bollinger Bands
        data = data.join(bb)
    else:
        data.ta.strategy()
    return data


def preprocess_data(data):
    """
    function which clears the stock market data from not essential columns and adds prediction value to stock market data
    :param data: pd.Dataframe
    :return: pd.Dataframe
    """
    data['TargetNextClose'] = data['Adj Close'].shift(-1)  # add column with desired prediction for the next day
    data.dropna(inplace=True)
    data.reset_index(inplace=True)
    data.drop(['Volume', 'Close', 'Date'], axis=1, inplace=True)
    print(data.tail())


def normalize_data(data):
    """
    function which normalizes the stock market data
    :param data: pd.Dataframe
    :return: np.array()
    """
    sc = MinMaxScaler(feature_range=(0, 1))
    data_normalized = sc.fit_transform(data)
    return data_normalized


def create_feature_target_sets(data, backcandles, target_column):
    """
    function which divide the stock market data to feature set and target vector
    :param data: np.array()
    :param backcandles: int, number of past time steps to include in the feature set for each sample.
    :param target_column: int, index of the column to predict
    :return:
    """
    print(data.shape[0])
    # Create the feature set X with all the columns except the last one
    X = np.array([data[i - backcandles:i, :target_column].copy() for i in range(backcandles, len(data))])
    print(X.shape)
    # Extract and reshape the target column to be a column vector
    y = np.array(data[backcandles:, target_column]).reshape(-1, 1)
    print(y.shape)
    return X, y

def split_data(X, y, split_ratio):
    """
    function which splits the Dataframe stock market data into train and test sets
    :param X: pd.Dataframe or np.array()
    :param y: pd.Dataframe or np.array()
    :param split_ratio: int
    :return: X_train: pd.Dataframe, X_test: pd.Dataframe, y_train: pd.Dataframe, y_test: pd.Dataframe
    """
    # split data into train test sets
    splitlimit = int(len(X) * split_ratio)
    print(f"Length of sets: \ntrain: {splitlimit} \ntest: {len(X) - splitlimit}")
    X_train, X_test = X[:splitlimit], X[splitlimit:]
    y_train, y_test = y[:splitlimit], y[splitlimit:]
    return X_train, X_test, y_train, y_test


def feature_correlation(data):
    """
    function which calculates the correlation between stock market data features
    :param data: pd.DataFrame
    """
    # Calculate the correlation matrix
    corr_matrix = data.corr()

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()


def recursive_feature_elimination(X, y, features_num):
    """
    function which recursively eliminates features using RFE
    :param X: pd.Dataframe
    :param y: pd.Dataframe
    :param features_num: int, the number of chosen features
    :return:
    """
    model = RandomForestRegressor()
    rfe = RFE(model, n_features_to_select=features_num)  # Adjust the number of features to select
    rfe.fit(X, y)

    # Get the selected features
    selected_features = X.columns[rfe.support_]
    print(selected_features)
    return selected_features


def build_LSTM_prediction_model(backcandles, num_features):
    """
    function which set parameters for the LSTM model
    :param backcandles: int
    :param num_features: int
    :return: Model
    """
    np.random.seed(10)

    lstm_input = Input(shape=(backcandles, num_features), name='lstm_input')
    inputs = LSTM(150, name='first_layer')(lstm_input)
    inputs = Dense(1, name='dense_layer')(inputs)
    output = Activation('linear', name='output')(inputs)
    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam()
    model.compile(optimizer=adam, loss='mse')

    return model


def accuracy(y_test, y_pred):
    """
    function which calculates the accuracy of the model
    :param y_test: np.array()
    :param y_pred: np.array()
    :return: int, R² Score
    """
    # Calculate accuracy metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'Root Mean Squared Error: {rmse}')
    print(f'R² Score: {r2}')
    return r2


def plot_prediction(y_test, y_pred, backcandles):
    """
    function plotting the prediction result and real stock market data
    :param y_test: np.array()
    :param y_pred: np.array()
    :param backcandles: int
    """
    # Plot the results
    plt.figure(figsize=(16, 8))
    plt.plot(y_test, color='black', label='Test')
    plt.plot(y_pred, color='green', label='Prediction')
    plt.legend()
    plt.title('Stock Price Prediction for backcandles {i}'.format(i=backcandles))
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.show()


if __name__ == '__main__':
    print('Stock Market Prediction - script running....')
    # Prepare data
    f_data = load_finance_data()
    f_data = add_indicators(f_data)
    preprocess_data(f_data)

    # Feature importance check
    split_ratio = 0.8
    X_train, X_test, y_train, y_test = split_data(f_data.iloc[:, :-1], f_data.iloc[:, -1], split_ratio=split_ratio)
    feature_correlation(X_train)
    selected_features = recursive_feature_elimination(X_train, y_train, 9)

    # LSTM prediction model
    feature_list = list(selected_features)
    print(feature_list)
    feature_list.append('TargetNextClose')
    columns = pd.Index(feature_list)
    array_data = normalize_data(f_data[columns])
    accuracy_r2 = []
    backcandles = [40, 70]  # number of past time steps to include in the feature set for each sample
    for i in backcandles:
        X, y = create_feature_target_sets(array_data, backcandles=i, target_column=-1)
        X_train, X_test, y_train, y_test = split_data(X, y, split_ratio=split_ratio)
        LSTM_model = build_LSTM_prediction_model(i, X_train.shape[2])

        # train model on training data
        LSTM_model.fit(x=X_train, y=y_train, batch_size=10, epochs=50, shuffle=True, validation_split=0.2)
        y_pred = LSTM_model.predict(X_test)  # Predict on test data
        accuracy_r2.append(accuracy(y_test, y_pred))  # Calculate the accuracy of prediction
        plot_prediction(y_test, y_pred, i)

    print(f"Number of past time steps: {backcandles}")
    print(f"R2 score: {accuracy_r2}")
    print('done')
