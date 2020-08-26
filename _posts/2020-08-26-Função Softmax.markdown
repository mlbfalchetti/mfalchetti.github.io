---
layout: post
title:  "Scripts UNIX - parte 01"
date:   2020-08-22 14:00:00 -0300
categories: jekyll update
---
{: style="text-align: justify" }

Quando você está implementando uma classificação em *Deep Learning* com múltiplas possíveis classes (mais classes que uma classificação binária) você pode utilizar a generalização da regressão logística chamada *regressão softmax*. 

Como exemplo de bioinformática, você pode querer montar um modelo que categorize amostras de câncer de mama em seus subtipos determinados de suas expressões de mRNA, 1 - subtipo Basal-*like*, 2 - Claudin-low, 3 - Luminal-*like* type A, 4 - Luminal-*like* type B e 5 - Normal-*like*. Neste caso você deve definir a última camada da rede como possuindo 5 nós, um para cada classe e deve esperar que em cada nó você receba as probabilidades de classes, para cada amostra.

´´´
Classe                  Probabilidade de classe
Basal-*like*            0.850
Claudin-low             0.025
Luminal-*like* type A   0.050
Luminal-*like* type B   0.050
Normal-*like*           0.025
**SOMA**                    1.000
