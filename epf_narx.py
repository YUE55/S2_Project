import numpy as np
from calendar import weekday
from time import time as t

def forecast_narx(DATA):
    import keras
    from keras.layers import Input, Dense
    from keras.models import Model
    from keras.callbacks import EarlyStopping
    # DATA: 8-column matrix (0-date, 1-hour, 2-price, 3-load forecast, 4-Sat, 5-Sun, 6-Mon 7-dummy, 8-p_min)
    # select data to be used
    # print(DATA[-1, :])
    price = DATA[:-1, 2]             # For day d (d-1, ...)
    price_min = DATA[:-1, 7]         # For day d
    Dummies = DATA[1:, 4:7]          # Dummies for day d+1
    loadr = DATA[1:, 3]              # Load for day d+1

    # Take logarithms
    price = np.log(price)
    mc = np.mean(price)
    price -= mc                      # Remove mean(price) to
    price_min = np.log(price_min)
    price_min -= np.mean(price_min)  # Remove mean(price)
    loadr = np.log(loadr)
    y = price[7:]                    # For day d, d-1, ...
    # Define explanatory variables for calibration
    X = np.vstack([price[6:-1], price[5:-2], price[:-7], price_min[6:-1],
                   loadr[6:-1], Dummies[6:-1, 0], Dummies[6:-1, 1], Dummies[6:-1, 2]]).T
    # Define explanatory variables for day d+1
    X_fut = np.hstack([price[-1], price[-2], price[-7], price_min[-1],
                       loadr[-1], Dummies[-1, 0], Dummies[-1, 1], Dummies[-1, 2]])
    # Define Neural Network model
    inputs = Input(shape=(X.shape[1], ))                  # Input layer
    hidden = Dense(units=5, activation='sigmoid')(inputs) # Hidden layer (5 neurons; GM = 20)
    outputs = Dense(units=1, activation='linear')(hidden) # Output layer
    model = keras.Model(inputs=inputs, outputs=outputs)
    callbacks = []
    model.compile(loss='MAE', optimizer='ADAM')           # Compile model
    model.fit(X, y, batch_size=64, epochs=500, verbose=0, # Fit to data
              validation_split=0, shuffle=False, callbacks=callbacks)
    prog = model.predict(np.array(X_fut, ndmin=2))        # Compute a step-ahead forecast

    return np.exp(prog + mc)                     # Convert to price level

def epf_narx(data, Ndays, startd, endd):
    # DATA:   4-column matrix (date, hour, price, load forecast)
    # RESULT: 4-column matrix (date, hour, price, forecasted price)
    first_day = str(int(data[0, 0]))
    # Weekday of starting day: 0 - Monday, .. 5 - Saturday, 6 - Sunday
    first_day = datetime(int(first_day[0:4]), int(first_day[4:6]), int(first_day[6:8])).weekday()
    N = len(data) // 24
    data = np.hstack([data, np.zeros((N*24, 4))]) # Append 'data' matrix with daily dummies & p_min
    for j in range(N):
        if first_day % 7 == 5:
            data[24*j:24*(j+1), 4] = 1 # Saturday dummy in 5th (index 4) column 0-23.....
        elif first_day % 7 == 6:
            data[24*j:24*(j+1), 5] = 1 # Sunday dummy in 6th column 0-23.....
        elif first_day % 7 == 0:
            data[24*j:24*(j+1), 6] = 1 # Monday dummy in 7th column 0-23.....
        first_day += 1
        data[24*j:24*(j+1), 7] = np.min(data[24*j:24*(j+1), 2]) # p_min in 8th column 0-23.....
    result = np.zeros((Ndays * 24, 4)) # Initialize `result` matrix
    result[:, :3] = data[endd*24:(endd + Ndays) * 24, :3] # column: 1 - date, 2 - hour, 3 - price
    for j in range(Ndays):     # for 354 days (0-353)
        for hour in range(24): # compute 1-day ahead forecasts for each hour
            data_h = data[hour::24, :] # slice every 24 elements to get certain hour (0,2,3,...,23)
            # Compute forecasts for the hour
            ts = t()
            result[j * 24 + hour, 3] = forecast_narx(data_h[startd + j:endd + j + 1, :]) # keep the window rolling 1 day
            print(f'{j}\t{hour}\t{t() - ts}') # display completed predictions and time spent
    return result
