from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dropout, LSTM, Dense

def cria_modelo(historico, parametros):
    modelo = Sequential()
    modelo.add(LSTM(units=100, return_sequences=True, input_shape=(historico, parametros)))
    modelo.add(Dropout(0.3))
    modelo.add(LSTM(units=50, return_sequences=True))
    modelo.add(Dropout(0.3))

    modelo.add(LSTM(units=50))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(units=1))
    modelo.compile(optimizer='adam', loss='mean_squared_error')
    return modelo

def treina(modelo, dados_treino, dados_esperados, epochs=100):
    es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=10, verbose=1)
    rlr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1)
    mcp = ModelCheckpoint(filepath = 'pesos.h5', monitor='loss', save_best_only = True, verbose=1)

    #treina o modelo
    modelo.fit(dados_treino, dados_esperados, batch_size=32, callbacks=[es, rlr, mcp], epochs=epochs)
    return modelo