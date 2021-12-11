import pandas as pd
from lstmAIF import LSTMAIF
porcent_train = 75
dates = '_01_01_2007_30_11_2021'
mayor_indices = pd.read_csv('Major World Market Indices.csv')
df_mayor_indices = pd.DataFrame(mayor_indices)

for i in df_mayor_indices.index:
    indice = df_mayor_indices['Index'][i]
    filename = 'DatosIndices/' + indice.replace('/', '_') + dates + '.csv'
    print(filename)
    df = pd.read_csv(filename)
    df_Predict, trainPredict, testPredict = LSTMAIF(df, porcent_train, indice)
    filePredict = 'Predicciones/' + indice.replace('/', '_') + '_Predict_' + str(porcent_train) + '.csv'
    df_Predict.to_csv(filePredict)

print('Hi')