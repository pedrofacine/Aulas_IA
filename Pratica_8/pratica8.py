import sklearn
from sklearn.datasets import load_breast_cancer
import pandas as pd

#1)
dataset = load_breast_cancer()
x = dataset['data']
y = dataset['target']

matrizDados = dataset.data.shape
alvo = dataset.target.shape
nomeAtributos = dataset.feature_names
nomesClasses = dataset.target_names
descricao = dataset.DESCR

df = pd.DataFrame(dataset.data, columns=nomeAtributos)

df['target'] = y

'''
D)
    a)O objetivo é diagnosticar se um tumor de mama é maligno ou benigno, utilizando como base
    diversas caracteristicas obtidas das células do paciente durante um exame de biópsia.
    
    b) Classificação, pois o target é composto por 0 ou 1.
    
    c) Classes 'maligno' que representa se o tumor é maligno e a 'Benigno', representa se é benigno
    
    d) Cada amostra, ou seja, cada linha do dataframe) representa os dados de um único exame/paciente.
'''
#2)
print(df.head())
print("------Tipos de dados-----")
print(df.dtypes)
print("-----estatisticas------")
print(df.describe())

'''
E)
    a) Sim, quando executa o comando, podemos ver que todos os atributos preditores
     originais do dataset são do tipo float64.
     
    b) Porque algoritmos baseados em cálculo de distância ou variância são sensíveis à
       magnitude dos números. Se as escalas forem muito diferentes, as variáveis com valores maiores vão dominar
       o modelo e receber um peso injusto, prejudicando o aprendizado.
       
    c) Essas métricas resumem o comportamento das variaveis, indicam o valor central (média), o grau de espalhamento
     dos dados (desvio padrão), os limites reais (mínimo e máximo).
    
    d) Não, a variável alvo é categórica (0 e 1), não se calcula média ou desvio padrão
'''
#3)
df['target'].unique()

df['target'].value_counts()

'''
b) É supervisionado porque os dados de treinamento já possuem os rótulos corretos (sabemos se é benigno ou maligno).
   É classificação porque o objetivo é prever categorias (0 ou 1), e não valores contínuos.
   
C)
    a) O objetivo é prever uma classe, categoria ou rótulo discreto para uma amostra.
    
    b) O objetivo é prever um valor numérico contínuo e infinito
    
    c) Porque a variável alvo não é um número contínuo. Ela possui apenas dois estados (0,1)
    
    d) Classificação: Prever se um e-mail recebido é spam ou não spam.
       Regressão: Prever o valor de venda de um imóvel com base na sua metragem e localização.

'''

#4)
print("---cont de valores ausentes---")
print(df.isnull().sum().sum())

'''
C)
    a) Não, não possui dados faltantes
    
    b) Eliminar reduz a quantidade de dados descartando a linha inteira. Substituir mantém
       o volume de dados ao preencher a lacuna com uma estimativa matemática.
    
    c) Quando a variável possui muitos valores extremos, pois eles puxam e distorcem o valor da média.
    
    d) Porque a mediana não é afetada por outliers, sendo uma medida mais segura para dados com distribuição assimétrica
    
    e) Perda de informações úteis, o que pode prejudicar o aprendizado e reduzir a precisão do modelo.

'''

#5)
