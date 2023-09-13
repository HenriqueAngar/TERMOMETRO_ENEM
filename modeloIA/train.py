from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense
from numpy.random.mtrand import random
from sklearn import preprocessing as prep
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras

### IMPORTA E UNIFICA O CONTEÚDO ####
data = pd.read_csv('enem.csv', sep=';')
names = ['NU_INSCRICAO', 'TP_NACIONALIDADE', 'TP_ST_CONCLUSAO', 'TP_ENSINO']
data = data.drop(names, axis=1)


##### PRÉ - PROCESSAMENTO DOS DADOS #####

# CODIFICA CATEGORIAS

encoder = prep.OrdinalEncoder()

sexo = data['TP_SEXO'].values
data['TP_SEXO'] = encoder.fit_transform(sexo.reshape(-1, 1))

sexo = data['NO_MUNICIPIO_ESC'].values
data['NO_MUNICIPIO_ESC'] = encoder.fit_transform(sexo.reshape(-1, 1))

sexo = data['SG_UF_ESC'].values
data['SG_UF_ESC'] = encoder.fit_transform(sexo.reshape(-1, 1))


# CONVERTE PARA NULO VALORES EQUIVALENTES A "NÃO SEI" E SUBSTITUI PARA A MODA

moda = data['Q001'].mode()[0]
data['Q001'] = data['Q001'].replace('H', np.nan)
data['Q001'].fillna(moda, inplace=True)

moda = data['Q002'].mode()[0]
data['Q002'] = data['Q002'].replace('H', np.nan)
data['Q002'].fillna(moda, inplace=True)

moda = data['Q003'].mode()[0]
data['Q003'] = data['Q003'].replace('F', np.nan)
data['Q003'].fillna(moda, inplace=True)

moda = data['Q004'].mode()[0]
data['Q004'] = data['Q004'].replace('F', np.nan)
data['Q004'].fillna(moda, inplace=True)


# CONVERTE VALORES CHAR PARA INT COM VALOR POR ORDEM CRESCENTE NO ALFABETO

quest1 = []
quest2 = []
quest3 = []
quest4 = []
alphabet = 'abcdefghijklmnopqrstuvxyz'
notas = ['NU_NOTA_CN', 'NU_NOTA_CH',
         'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']


def numberit(letter):

    letter = letter.lower()

    if letter in alphabet:
        return ord(letter) - 96
    else:
        return 0


for i in range(1, 38):

    if i != 5:
        if i < 10:
            name1 = 'Q00'+str(i)
            name2 = 'QN00'+str(i)
            quest1.append(name2)

        else:
            name1 = 'Q0'+str(i)
            name2 = 'QN0'+str(i)

            if i <= 18:
                quest2.append(name2)
            elif i <= 28:
                quest3.append(name2)
            else:
                quest4.append(name2)

        data[name2] = data[name1].apply(numberit)
        data = data.drop([name1], axis=1)

    else:
        quest1.append('QN005')
        data['QN005'] = data['Q005']
        data = data.drop('Q005', axis=1)

# CORRIGE VALORES MAIORES QUE 100

for nota in notas:
    data.loc[data[nota] > 1000, nota] = data.loc[data[nota] > 1000, nota] / 10

# ELIMINTA NOTAS 0 NA REDACAO
data = data.loc[data['NU_NOTA_REDACAO'] != 0]

data.dropna(subset=notas, inplace=True)


# CORRELAÇÃO COM NOTA POR PERFIL DEMOGRÁFICO
combo = quest1 + quest2 + quest3 + quest4
am1 = data.drop(combo, axis=1)
am1.corr('pearson')


# CORRELAÇÃO COM NOTA POR PERGUNTAS QUESTIONARIO
names = notas+quest4
am2 = data[names]
am2.corr('pearson')


### GERA AS INFORMAÇÕES  PRÉ - PROCESSADAS ####

# ELIMINA COLUNAS DE BAIXA CORRELAÇÃO
names = quest1+quest2+quest3+quest4+['TP_LOCALIZACAO_ESC',  'NO_MUNICIPIO_ESC',
                                     'TP_ESTADO_CIVIL', 'TP_FAIXA_ETARIA', 'TP_ESTADO_CIVIL', 'SG_UF_ESC']+notas
info = data.drop(names, axis=1)

# AGRUPA COLUNAS DO QUESTIONÁRIO POR CRITÉRIO E RELEVÂNCIA DA COLUNA
info['RENDA'] = data['QN006']
info['PAIS'] = data['QN001'] + data['QN002'] + data['QN003'] + data['QN004']
info['CASA'] = data['QN007']*0.3 + data['QN008']*0.3 + data['QN009']*1 + data['QN010']*1 + \
    data['QN013']*0.5 + data['QN014']*0.5 + data['QN015'] * \
    0.3 + data['QN016']*1 + data['QN018']*0.75
info['INFO'] = data['QN019']*1 + data['QN022'] * \
    0.75 + data['QN024']*1 + data['QN025']*0.5
info['FOCO'] = data['QN027']*0.75 + data['QN028']*1.5 + \
    data['QN030']*0.75 + data['QN035']*0.5 + data['QN036']*1


# ESCALA AS INFORMAÇÃO DAS COLUNAS AGRUPADAS DE 0 A 1
colunas = ['RENDA', 'PAIS', 'CASA', 'INFO', 'FOCO']
for coluna in colunas:

    min = info[coluna].min()
    max = info[coluna].max()

    info[coluna] = (info[coluna].values - min) / (max-min)

for nota in notas:
    info[nota] = data[nota]

info['NOTA'] = data[notas].mean(axis=1)

# coleta dados de amostra pré-normalização
teste = info.sample(20)


# Plota as correlações para as novas entradas
info.corr('pearson')


# PLOTA A DISTRIBUIÇÃO DAS NOTAS PRÉ - TRANSFORMAÇÃO


# Calcular média e desvio padrão dos dados de tarifa
media = info['NOTA'].mean()
desvio_padrao = info['NOTA'].std()

# Ajustar uma distribuição normal aos dados
distribuicao_normal = norm(loc=media, scale=desvio_padrao)

# Criar um espaço de valores para plotagem
valores = np.linspace(info['NOTA'].min(), info['NOTA'].max(), 1000)

# Calcular a densidade de probabilidade para cada valor
densidade_probabilidade = distribuicao_normal.pdf(valores)

# Criar um gráfico de distribuição
plt.figure(figsize=(10, 6))
plt.hist(info['NOTA'], bins=20, density=True,
         alpha=0.6, color='b', label='NOTAS')
plt.plot(valores, densidade_probabilidade, 'r-',
         lw=2, label='Distribuição Normal')
plt.title('Distribuição de Médias com Curva Normal')
plt.xlabel('NOTAS')
plt.ylabel('Densidade de Probabilidade')
plt.legend()
plt.grid(True)
plt.show()


##### NORMALIZAÇÃO DA AMOSTRA #####

for nota in notas:

    lower = info[nota].quantile(0.15)
    upper = info[nota].quantile(0.96)
    info = info[(info[nota] >= lower) & (info[nota] <= upper)]

info['NOTA'] = data[notas].mean(axis=1)
notas.append('NOTA')


# PLOTA A DISTRIBUIÇÃO DAS NOTAS PÓS - TRANSFORMAÇÃO


# Calcular média e desvio padrão dos dados de tarifa
media = info['NOTA'].mean()
desvio_padrao = info['NOTA'].std()

# Ajustar uma distribuição normal aos dados
distribuicao_normal = norm(loc=media, scale=desvio_padrao)

# Criar um espaço de valores para plotagem
valores = np.linspace(info['NOTA'].min(), info['NOTA'].max(), 1000)

# Calcular a densidade de probabilidade para cada valor
densidade_probabilidade = distribuicao_normal.pdf(valores)

# Criar um gráfico de distribuição
plt.figure(figsize=(10, 6))
plt.hist(info['NOTA'], bins=20, density=True,
         alpha=0.6, color='b', label='NOTAS')
plt.plot(valores, densidade_probabilidade, 'r-',
         lw=2, label='Distribuição Normal')
plt.title('Distribuição de Médias com Curva Normal')
plt.xlabel('NOTAS')
plt.ylabel('Densidade de Probabilidade')
plt.legend()
plt.grid(True)
plt.show()


#### CONFIGURAÇÕES DE TREINAMENTO#####

x = info.drop(notas, axis=1)
y = info[notas]
y = y.drop('NOTA', axis=1)

scaler = StandardScaler()
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.3, random_state=73)

model = keras.Sequential()
model.add(Dense(18, input_dim=9, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(5, activation='linear'))

# Compilando o modelo

model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['mae']
)

history = model.fit(
    x_train, y_train,
    epochs=50, batch_size=100,
    verbose=2,
    validation_data=(x_test, y_test)
)


# PLOTAGEM DE PERDAS

plt.figure(figsize=(24, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Histórico de Loss')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.ylim(3750, 5550)
plt.legend


# PLOTAGEM DE ERROS

plt.figure(figsize=(24, 6))
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('ERRO ABSOLUTO DE NOTAS')
plt.xlabel('Épocas')
plt.ylabel('Mae')
plt.legend()
plt.ylim(40, 100)
plt.show


# APRESENTA DADOS A PARTIR DA AMOSTRA
notas_reais = teste[notas]
teste = teste.drop(notas, axis=1)
notas_modelo = model.predict(teste)
notas_modelo = pd.DataFrame(notas_modelo)

i = 0
medias = []
resultado = []
resultado = pd.DataFrame(resultado)

notas = ['NU_NOTA_CN', 'NU_NOTA_CH',
         'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']
for nota in notas:

    resultado[nota+'_R'] = notas_reais[nota].values
    resultado[nota+'_M'] = notas_modelo.iloc[:, i].values
    resultado[nota+'_M'] = resultado[nota+'_M'].round()

    if i == 0:
        resultado['MEAN_R'] = resultado[nota+'_R']
        resultado['MEAN_M'] = resultado[nota+'_M']

    else:
        resultado['MEAN_R'] = resultado['MEAN_R'] + resultado[nota+'_R']
        resultado['MEAN_M'] = resultado['MEAN_M'] + resultado[nota+'_M']

    i += 1

resultado['MEAN_R'] = resultado['MEAN_R'].values/5
resultado['MEAN_M'] = resultado['MEAN_M'].values/5
resultado['ERRO'] = resultado['MEAN_R'] - resultado['MEAN_M']

resultado['MEAN_R'] = resultado['MEAN_R'].values.round()
resultado['MEAN_M'] = resultado['MEAN_M'].values.round()
resultado['ERRO'] = resultado['ERRO'].values.round()

names = ['MEAN_R', 'MEAN_M', 'ERRO']
geral = resultado[names]
resultado = resultado.drop(names, axis=1)
