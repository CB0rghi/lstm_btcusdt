import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
from datetime import datetime

def carrega(caminho):
    dados = pd.read_csv(caminho)
    dados.sort_values(by=['Data'])
    del dados['Data']
    return dados

def extrai_data(caminho, nome_col):
    dados = pd.read_csv(caminho)
    col_pos = dados.columns.get_loc(nome_col)
    datas = dados.iloc[:, col_pos: col_pos + 1].values
    lista_data = []
    for i in range(0, len(datas)):
        data = datas[i, 0]
        partes = data.split('/')

        dia = partes[0]
        #ajusta os valores quando necessário(com 0 à esq.)
        if(len(dia) == 1):
            dia = '0' + dia
        mes = partes[1]
        if(len(mes) == 1):
            mes ='0' + mes

        data_completa = dia + '/' + mes + '/' + partes[2]
        #converte a string em objeto data
        data = datetime.strptime(data_completa, '%d/%m/%Y %H:%M:%S')
        lista_data.append(data)
    return lista_data

def normaliza(dados):
    caminho = 'normalizador.joblib.pkl'
    if not os.path.isfile(caminho):
       #caso ainda não exista
       #inicializa-o com a função MinMaxScaler
       normalizador = MinMaxScaler(feature_range=(0,1))
       #utiliza a função fit_transform para moldar o normalizador
       dados_normalizados = normalizador.fit_transform(dados)
       joblib.dump(normalizador, caminho)
    else:
       #se já existe, carrega-o em memória e transforma
       #os dados com apenas transform
       normalizador = joblib.load(caminho)
       dados_normalizados = normalizador.transform(dados)

    return dados_normalizados

def prepara(dados, coluna_esperada, historico):
    entrada_normalizada = normaliza(dados)
    #guarda os dados esperados em memória
    pos_esperado = dados.columns.get_loc(coluna_esperada)
    dados_esperados = entrada_normalizada[:, pos_esperado: pos_esperado+1]
    #ignora os primeiros dados(que não possuem historico)
    dados_esperados = dados_esperados[historico:, :]
    #aqui montarei o vetor de entrada da LSTM
    #onde cada posição contém os parâmetros normalizados
    #dos minutos anteriores à i(que simbolizará o minuto atual)
    #isso será feito para todos os minutos possíveis
    dados_treino = []
    for i in range(historico, len(entrada_normalizada)):
        dados_treino.append(entrada_normalizada[i - historico:i, :])

    #transforma todos em arrays com numpy
    dados_treino = np.array(dados_treino)
    dados_esperados = np.array(dados_esperados)

    return [dados_treino, dados_esperados]

