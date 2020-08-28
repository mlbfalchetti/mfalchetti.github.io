---
layout: post
title:  "Função Softmax"
date:   2020-08-26 14:00:00 -0300
categories: deep-learning activation-function
---
### Probabilidades de múltiplas possíveis classes

<p style="text-align: justify;">
Quando você está implementando uma classificação em <i>Deep Learning</i> com múltiplas possíveis classes (mais classes que uma classificação binária) você pode utilizar a generalização da regressão logística chamada <i>regressão softmax</i>.
</p>

<p style="text-align: justify;">
Como exemplo de bioinformática, você pode querer montar um modelo que categorize amostras de câncer de mama em seus subtipos determinados de suas expressões de mRNA, 1 - subtipo Basal-<i>like</i>, 2 - Claudin-low, 3 - Luminal-<i>like</i> type A, 4 - Luminal-<i>like</i> type B e 5 - Normal-<i>like</i>. Neste caso você deve definir a última camada da rede como possuindo 5 nós, um para cada classe e deve esperar que em cada nó você receba as probabilidades de classes, para cada amostra, algo como mostrado abaixo. <b>É importante notar que o valor total das probabilidades é 1.</b> 
</p>

```
#-- Amostra #1
#-- Classe              #-- Probabilidade de classe
Basal-like              0.850
Claudin-low             0.025
Luminal-like type A     0.050
Luminal-like type B     0.050
Normal-like             0.025
Total                   1
```

Na última camada da rede neural, como sempre, você irá calcular a função linear (Z<sup>[L]</sup> = W<sup>[L]</sup> * A<sup>[L - 1]</sup> + B<sup>[L]</sup>) e então irá empregar a função de ativação (A) <i>Softmax</i>. A fórmula é:

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<p>
  $$a^{[L]} = {e^{Z^{L}} \over {\sum_{i=i}^5 e^{Z^{L}}}}.$$
</p>

### Dissecando 

<p style="text-align: justify;">
O que a fórmula faz é colocar os resultados da função linear (Z<sup>[L]</sup>) no lugar do valor expoente do logaritmo de base <i>e</i>, ou logaritmo <i>natural</i>, gerando os valores e ^ Z<sup>[L]</sup>, e por fim os divide pela soma de todos esses valores (&sum; e ^ Z<sup>[L]</sup>), nos dando assim uma proporção de 0 a 1. Simples e efetivo. 
</p>

Vejamos com exemplo:

Dados:
```
#-- Classe              #-- Z^{L}  
Basal-like              5.0
Claudin-low             2.0
Luminal-like type A     0.2
Luminal-like type B     0.2 
Normal-like             0.5
Total                   7.9
```

Passo 01:
```
#-- Classe              #-- Z^{L}     #-- e^{Z^{L}}
Basal-like              5.0           148.5
Claudin-low             2.0           7.3
Luminal-like type A     0.2           1.2
Luminal-like type B     0.2           1.2
Normal-like             0.5           1.6
Total                   7.9           159.8
```

Passo 02:
```
#-- Classe              #-- Z^{L}     #-- e^{Z^{L}}     #-- e^{Z^{L}} / e^{Z^{L}}
Basal-like              5.0           148.5             ~0.929
Claudin-low             2.0           7.3               ~0.045
Luminal-like type A     0.2           1.2               ~0.007
Luminal-like type B     0.2           1.2               ~0.007
Normal-like             0.5           1.6               ~0.010
Total                   7.9           159.8             0.998 ou ~1
```

Aqui, essa amostra foi predita como subtipo molecular "Basal-<i>like</i>" com uma <i>probabilidade de classe</i> de 0.929. 
