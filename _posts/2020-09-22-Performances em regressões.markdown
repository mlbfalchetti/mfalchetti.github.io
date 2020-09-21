---
layout: post
title:  "Performances em regressões"
date:   2020-09-22 14:00:00 -0300
categories: performance-measure regression
---
### Decorar ou aprender

<p style="text-align: justify;">


Toda tarefa de aprendizagem de máquina deve possuir um medidor de performance, dessa forma você saberá se seu modelo está cometendo muitos erros. Quando você está realizando uma tarefa de regressão, seja ela linear, logística, múltipla etc, a medida típica de performance é o RMSE ou Root Mean Square Error. Ele dá uma ideia do erro que o sistema tipicamente faz em suas predições, dando maior peso a erros maiores. 
  A fórmula matemática para computar o RMSE é: 
  
  Apesar de que geralmente o RMSE é o preferido como medida de performance para tarefas de regressão, em alguns contextos vocÊ alvez prefira utilizar outra função. Por exemplo, se houver muitas amostras outliers você preferirá optar pelo Mean Absolute Error (MAE, também chamado de average absolute deviation).
  
 Ambos RMSE e MAE são formas de medir a distância entre dois vetores: o vetor de predições e o vetor de valores alvo. Várias medidas de distância, <i>ou normas</i>, são possíveis:
 
 - Computar a raiz da soma dos quadrados (RMSE) corresponde a norma Euclideana: Essa é a noção de distância que você é familiarizado. É também chamada de norma L2, notada ||.||2 (ou apenas ||.||) 
 - Computar a soma dos valores absolutos (MAE) corresponde a norma L1, notada ||.||1. Ela é algumas vezes chamada de norma Manhattan porque a medida de distância entre dois pontos em uma cidade onde você só pode viajar através de blocos ortogonais da cidade (quadras).
 - Mais genericamente norma lk de um vetor v contendo n elementos é definido como ||v||k = (fórmula). l0 dá o número de elementos não-zero no vetor e linf dá o valor máximo absoluto no vetor.
 - Quanto maior for o indice da norma, mais ela se focará em valores grandes e negligenciará valores pequenos. É por isso que o RMSE é mais sensível a outliers do que MAE. Mas quando os outliers são raros (como em uma curva em formato de sino), o RMSE performa muito bem e é geralmente preferido. 


</p>


<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<p style="text-align: justify;">
  $$RMSE (X, h) = \sqrt{{1 \over m} {\sum_{i=1}^m (h(x^{(i)}) - y^{(i)})^2} }.$$
</p>

<p style="text-align: justify;">
  $$MAE (X, h) = {1 \over m} \sum_{i=1}^m|h (x^{(i)}) - y^{(i)}|.$$
</p>

