from prepara import carrega, extrai_data, prepara
from treina import cria_modelo, treina
from testa import testa

def treina_modelo():
    caminho = 'dados/btcusdt_treino.csv'
    historico = 60
    dados = carrega(caminho)
    dados_treino, dados_esperados = prepara(dados, 'Preço', historico)
    modelo = cria_modelo(historico, dados_treino.shape[2])
    modelo = treina(modelo, dados_treino, dados_esperados)

def testa_modelo():
    caminho = 'dados/btcusdt_teste.csv'
    historico = 60

    dados = carrega(caminho)
    dados_teste = prepara(dados, 'Preço', historico)[0]
    modelo = 'pesos.h5'
    datas = extrai_data(caminho, 'Data')
    datas = datas[historico: len(datas)]
    testa(modelo, caminho, historico, dados_teste.shape[2], datas)

treina_modelo()
testa_modelo()