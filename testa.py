from prepara import carrega, prepara
from treina import cria_modelo
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import joblib
def testa(modelo, caminho, hist, parametros, datas):
    if isinstance(modelo, str):
        temp = cria_modelo(hist, parametros)
        temp.load_weights(modelo)
        modelo = temp
    dados = carrega(caminho)
    dados_teste, dados_esperados = prepara(dados, 'Preço', hist)
    previstos = modelo.predict(dados_teste)
    dados = np.array(dados)
    precos_reais = []

    #armazena preço real e substitui no vetor pelo preço previsto
    for i in range(hist, len(dados)):
        precos_reais.append(dados[i,0])
        dados[i, 0] = previstos[i - hist]
    #ignora as primeiras linhas que n possuem dados suficientes
    #para a previsão
    dados = dados[hist:len(dados), :]

    caminho = 'normalizador.joblib.pkl'
    normalizador = joblib.load(caminho)

    #transforma os valores normalizados em valores reais
    dados_transformados = normalizador.inverse_transform(dados)

    #seleciona apenas a coluna do preço
    precos_previstos = dados_transformados[:, 0]
    fig, ax = plt.subplots()
    ax.plot(datas, precos_reais, color='blue', label='Preço real')
    ax.plot(datas, precos_previstos, color='red', label='Preço previsto')

    #formata data
    myFmt = dates.DateFormatter("%d/%m/%Y %H:%M")
    ax.xaxis.set_major_formatter(myFmt)
    plt.gcf().autofmt_xdate()
    plt.title('Previsão do preço do bitcoin')
    plt.xlabel('Data')
    plt.ylabel('Preço')
    plt.legend()
    plt.show()