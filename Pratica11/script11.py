import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

print("QUESTÃO 1 - Divisão Simples Treino/Teste")

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

clf1 = DecisionTreeClassifier(random_state=42)
clf1.fit(X_train, y_train)

y_pred1 = clf1.predict(X_test)

cm1 = confusion_matrix(y_test, y_pred1)
acc1    = accuracy_score(y_test, y_pred1)
prec1   = precision_score(y_test, y_pred1)
rec1    = recall_score(y_test, y_pred1)
f1_1    = f1_score(y_test, y_pred1)

print(f"\nmatriz de confusão:\n{cm1}")
print(f"\nacuracia : {acc1:.4f}")
print(f"precisão : {prec1:.4f}")
print(f"recall   : {rec1:.4f}")
print(f"F1Score : {f1_1:.4f}")

print("QUESTÃO 2")

X_trainval, X_test2, y_trainval, y_test2 = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

X_train2, X_val2, y_train2, y_val2 = train_test_split(
    X_trainval, y_trainval,
    test_size=0.30,
    random_state=42,
    stratify=y_trainval
)

print(f"\ntamanhos dos conjuntos:")
print(f"treino: {len(X_train2)} amostras ({len(X_train2)/len(X)*100:.1f}%)")
print(f"validação: {len(X_val2)} amostras ({len(X_val2)/len(X)*100:.1f}%)")
print(f"teste     : {len(X_test2)} amostras ({len(X_test2)/len(X)*100:.1f}%)")

clf2 = DecisionTreeClassifier(random_state=42)
clf2.fit(X_train2, y_train2)

y_pred_val2 = clf2.predict(X_val2)
print(f"\ndesempenho no conjunto de validacao")
print(f"acuracia : {accuracy_score(y_val2, y_pred_val2):.4f}")
print(f"precisão : {precision_score(y_val2, y_pred_val2):.4f}")
print(f"recall   : {recall_score(y_val2, y_pred_val2):.4f}")
print(f"F1Score : {f1_score(y_val2, y_pred_val2):.4f}")

y_pred2 = clf2.predict(X_test2)

cm2   = confusion_matrix(y_test2, y_pred2)
acc2  = accuracy_score(y_test2, y_pred2)
prec2 = precision_score(y_test2, y_pred2)
rec2  = recall_score(y_test2, y_pred2)
f1_2  = f1_score(y_test2, y_pred2)

print(f"\ndesempenho no conjunto de teste")
print(f"matriz de confusão:\n{cm2}")
print(f"\nacurácia : {acc2:.4f}")
print(f"precisão : {prec2:.4f}")
print(f"recall   : {rec2:.4f}")
print(f"F1Score : {f1_2:.4f}")

print("QUESTÃO 3")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []
cms_kfold = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    X_tr, X_ts = X[train_idx], X[test_idx]
    y_tr, y_ts = y[train_idx], y[test_idx]

    clf_k = DecisionTreeClassifier(random_state=42)
    clf_k.fit(X_tr, y_tr)
    y_pred_k = clf_k.predict(X_ts)

    cm_k   = confusion_matrix(y_ts, y_pred_k)
    acc_k  = accuracy_score(y_ts, y_pred_k)
    prec_k = precision_score(y_ts, y_pred_k)
    rec_k  = recall_score(y_ts, y_pred_k)
    f1_k   = f1_score(y_ts, y_pred_k)

    fold_results.append({
        "Fold": fold,
        "Acurácia": acc_k,
        "Precisão": prec_k,
        "Recall": rec_k,
        "F1-Score": f1_k
    })
    cms_kfold.append(cm_k)

    print(f"\nfold {fold}:")
    print(f"matriz de confusão:\n{cm_k}")
    print(f"acurácia : {acc_k:.4f} / precisão : {prec_k:.4f} | "
          f"Recall : {rec_k:.4f} / F1Score : {f1_k:.4f}")

df_folds = pd.DataFrame(fold_results).set_index("Fold")
means = df_folds.mean()

print("\ntabela de resultados por fold ---")
print(df_folds.to_string())
print("\nmédia das métricas")
print(f"acurácia : {means['Acurácia']:.4f}")
print(f"precisão : {means['Precisão']:.4f}")
print(f"recall   : {means['Recall']:.4f}")
print(f"F1Score : {means['F1-Score']:.4f}")

acc3  = means['Acurácia']
prec3 = means['Precisão']
rec3  = means['Recall']
f1_3  = means['F1-Score']

print("QUESTÃO 4")

comparativo = pd.DataFrame({
    "Estratégia": ["Divisão Simples", "Holdout", "K-Fold (média)"],
    "Acurácia":   [acc1, acc2, acc3],
    "Precisão":   [prec1, prec2, prec3],
    "Recall":     [rec1, rec2, rec3],
    "F1-Score":   [f1_1, f1_2, f1_3]
}).set_index("Estratégia")
print("\n--- Tabela Comparativa ---")
print(comparativo.to_string(float_format="{:.4f}".format))

print("\n--- Análise das Matrizes de Confusão ---")
print(f"\nQ1 (Divisão Simples):\n{cm1}")
print(f"\nQ2 (Holdout - conjunto de teste):\n{cm2}")
print(f"\nQ3 (K-Fold - soma das matrizes de cada fold):\n{sum(cms_kfold)}")

print("\n--- Tabela Comparativa ---")
print(comparativo.to_string(float_format="{:.4f}".format))
 
print("\n--- Análise das Matrizes de Confusão ---")
print(f"\nQ1 (Divisão Simples):\n{cm1}")
print(f"\nQ2 (Holdout - conjunto de teste):\n{cm2}")
print(f"\nQ3 (K-Fold - soma das matrizes de cada fold):\n{sum(cms_kfold)}")


"""
a) As três abordagens produziram métricas similares (~91-94%), mas a divisão
   simples pode ser otimista por depender de uma única partição aleatória.
 
b) A matriz do K-Fold (soma dos folds) cobre todos os dados e é mais
   representativa; as de Q1 e Q2 refletem apenas uma divisão específica.
 
c) A validação evita que o teste vaze informações para o ajuste do modelo,
   tornando a avaliação final mais honesta e reduzindo o risco de overfitting.
 
d) Para este dataset (569 amostras), o StratifiedKFold (K=5) é a estratégia
   mais adequada: usa todos os dados, mantém a proporção das classes e
   fornece estimativas mais estáveis do desempenho real do modelo.

"""