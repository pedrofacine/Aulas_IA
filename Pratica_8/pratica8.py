import sklearn
from sklearn.datasets import load_breast_cancer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

atributos_preditores = df.drop(columns=['target'])
matriz_corr = atributos_preditores.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(matriz_corr, annot=False, cmap='coolwarm')
plt.title("mapa de calor de correlação")
plt.show()

'''
E)
    a) Quando uma aumenta, a outra também aumenta ou diminui proporcionalmente).

    b) Não, duas variaveis podem crescer juntas por coincidência ou por causa de um terceiro fator oculto, sem que uma seja
     a causa direta da outra.
    
    c) Porque eles entregam a mesma informação matemática para o modelo de aprendizado, aumentando o custo
     computacional sem agregar novo poder de previsão.
    
    d) A correlação tradicional mede apenas relações lineares. Ela pode falhar em identificar padrões complexos
     ou relacionamentos não lineares entre os dados.

'''
#6)

colunas_para_remover = ['mean perimeter', 'mean area']

df_selecionado = df.drop(columns=colunas_para_remover)

print(f"quantidade de atributos originais: {df.shape[1]}")
print(f"quantidade de atributos após seleção: {df_selecionado.shape[1]}")

'''
    d) removi o perímetro e a área porque eles representam a mesma grandeza que o raio da célula.
    Manter apenas um deles elimina a redundância sem causar perda significativa de informação para o modelo.

e)
    a) É o processo de escolher um subconjunto com as melhores variáveis originais e descartar o resto.

    b) Atributos irrelevantes não têm relação com a variável alvo. Atributos redundantes até ajudam, mas entregam 
    a mesma informação que outra variável que já está no dataset.
    
    c) Porque as colunas não sofrem alterações matemáticas estruturais; elas são apenas filtradas, mantendo sua
     unidade de medida e interpretação originais.
    
    d) Treinamento mais rápido, menor custo computacional, redução do risco de overfitting e modelos mais fáceis de interpretar.
'''

#7)

X = df.drop(columns=['target'])

scaler = StandardScaler()
X_padronizado = scaler.fit_transform(X)

df_padronizado = pd.DataFrame(X_padronizado, columns=X.columns)

print("--- dados padronizados")
print(df_padronizado.head())

print("\n--- comportamento após a transformação ---")
print("media de cada coluna:\n", df_padronizado.mean().head())
print("\ndesvio Padrão de cada coluna:\n", df_padronizado.std().head())

'''
a)Variáveis com números absolutos maiores apresentarão uma variância muito maior
 do que variáveis pequenas. Sem ajuste, o algoritmo achará que a área é mais importante apenas porque seus números são maiores.

c) A média de todos os atributos passa a ser exatamente zero e a variância passa a ser um.

d)
    a) É transformar matematicamente todas as colunas para que fiquem na mesma escala, eliminando distorções de magnitude.
    
    b) Porque o PCA procura capturar as direções de maior variância nos dados. Se não padronizarmos, o PCA dará peso injusto às variáveis com valores maiores.
    
    c) Não. Algoritmos baseados em árvores não calculam distâncias nem variâncias, eles apenas criam regras lógicas
     de corte (ex: "se valor > 10"), logo, são imunes à escala.
    
    d) Dados originais possuem unidade de medida real. Dados padronizados são adimensionais e representam apenas
     a que distância aquele valor está da média.
'''
#8)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_padronizado)

df_pca = pd.DataFrame(data=X_pca, columns=['Componente 1', 'Componente 2'])

print("--- nova representação de dados")
print(df_pca.head())

variancia = pca.explained_variance_ratio_
print(f"componente 1 explica: {variancia[0]*100:.2f}% dos dados originais")
print(f"componente 2 explica: {variancia[1]*100:.2f}% dos dados originais")
print(f"yotal de informação retida: {sum(variancia)*100:.2f}%")

print("\n--- comparação de dimensionalidade")
print(f"dimensionalidade Original: {df_padronizado.shape[1]} atributos")
print(f"Dimensionalidade Transformada: {df_pca.shape[1]} componentes")

'''
E)
    a) É um algoritmo de transformação matematica que reduz a quantidade de colunas de um dataset criando novos
    eixos que maximizam a retenção de variância
    
    b) Porque a seleção apenas escolhe colunas originais e descarta o resto. A extração mistura todos os atributos
     originais para criar variáveis totalmente novas.
     
    c) É a porcentagem de informação útil que os novos componentes principais conseguiram reter após a redução.
    
    d) Ganha: Redução da dimensionalidade e eliminação de variáveis correlacionadas.
       Perde: A interpretabilidade humana. Os componentes são combinações matemáticas abstratas, então você não consegue mais dizer qual coluna é o "raio" ou a "área".
'''

#9)
print("--- comparação final das abordagens")
print(f"dataset via seleção: {df_selecionado.shape[1]} colunas")
print(f"dataset via PCA: {df_pca.shape[1]} componentes")

'''
    A seleção apenas filtrou as colunas, mantendo a interpretabilidade. O PCA pegou todas as
    30 colunas originais e as espremeu em apenas 2 novos eixos, perdendo o significado humano em troca de máxima compressão de dados
    
C)
    a)A seleção apenas escolhe um subconjunto das variáveis originais e descarta o resto. A extração (PCA) cria variáveis inteiramente
     novas a partir da combinação matemática das originais.
    
    b) Na seleção de atributos
    
    c)Na extração de atributos, pca
    
    d) Quando a interpretabilidade é essencial. Por exemplo, na área da saúde, os médicos precisam saber
     exatamente qual característica real da célula motivou a decisão do modelo
    
    e) Quando o foco é apenas o desempenho preditivo ou a visualização gráfica (2D/3D) de datasets gigantescos, e entender o significado físico
     de cada nova coluna não é importante

'''



























