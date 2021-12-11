import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def LSTMAIF(df, porcent_train = 75, indice=''):
    # creating dataframe
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df)), columns=['Close'])

    for i in range(0, len(data)):
        new_data['Close'][i] = data['Close'][i]

    # setting index
    new_data.index = new_data.Close
    new_data.drop('Close', axis=1, inplace=True)

    # creating train and test sets
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    dataset = df.iloc[:, 1:2].values
    dataset = dataset.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    train_size = int(len(dataset) * (porcent_train / 100))
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(trainX.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)
    # predicting  values, using past 50 from the train data
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    numpy.round(trainPredict, 2)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    numpy.round(testPredict, 2)
    testY = scaler.inverse_transform([testY])
    # Predicted data
    dataPredict = np.empty_like(dataset)
    dataPredict[look_back:len(trainPredict) + look_back, :] = trainPredict
    dataPredict[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
    dataPredict[0] = dataPredict[1]
    dataPredict[len(dataPredict) - 1] = dataPredict[len(dataPredict) - 2]
    dataPredict[len(trainPredict) + (look_back * 2) - 1] = dataPredict[len(trainPredict) + (look_back * 2) - 2]
    dataPredict[len(trainPredict) + (look_back * 2)] = dataPredict[len(trainPredict) + (look_back * 2) + 1]
    df_aux1 = pd.DataFrame(dataPredict)
    df_Predict = pd.concat([df['Date'], df_aux1], axis=1)
    df_Predict.columns = ['Date', 'Predict']
    df_Predict.set_index('Date', inplace=True)
    return df_Predict, trainPredict, testPredict
    # # shift train predictions for plotting
    # trainPredictPlot = np.empty_like(dataset)
    # trainPredictPlot[:, :] = np.nan
    # trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    # # shift test predictions for plotting
    # testPredictPlot = np.empty_like(dataset)
    # testPredictPlot[:, :] = np.nan
    # testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
    # # plot baseline and predictions
    # plt.plot(scaler.inverse_transform(dataset))
    # plt.plot(trainPredictPlot)
    # plt.plot(testPredictPlot)
    # plt.show()
    # filename = indice + 'Predict_' + str(porcent_train) + '.csv'
    # df_Predict.to_csv(filename)
    # plt.plot(df_Predict['Predict'])
    # plt.show()
    # # calculate root mean squared error
    # trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    # print('Train Score: %.2f RMSE' % (trainScore))
    # testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    # print('Test Score: %.2f RMSE' % (testScore))
