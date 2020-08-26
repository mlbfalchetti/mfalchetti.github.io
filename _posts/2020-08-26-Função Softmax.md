---
layout: post
title:  "Função Softmax"
date:   2020-08-26 14:00:00 -0300
categories: jekyll update
---
*Deep Learning*

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

Na última camada da rede neural, como sempre, você irá calcular a função linear (Z<sup>[L]</sup> = W<sup>[L]</sup> * A<sup>[L - 1]</sup> + B<sup>[L]</sup>) e então irá empregar a função de ativação (A) <i>Softmax</i>. Primeiramente você deve calcular:

```
# Em R 
t = e ^ Z # aqui Z terá 5 valores, um para cada classe -nó- da última camada da rede neural, ou seja, e ^ Z[da última camada L]
```

e então você irá normalizar o vetor t [5 nós, 1 camada] para que ele totalize 1, conforme mostrado acima na probablidade de classe esperada. Para isso:

```
# Em R
A = t / sum(t)
```

Por exemplo

```

```

$ \sum_{\forall i}{x_i^{2}} $

