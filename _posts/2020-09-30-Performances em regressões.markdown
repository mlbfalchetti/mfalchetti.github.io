---
layout: post
title:  "RMSE ou MAE para performances?"
date:   2020-09-30 20:00:00 -0300
categories: performance-measure
---
### RMSE ou MAE?

<p style="text-align: justify;">
As tarefas de aprendizagem de máquina devem sempre possuir um medidor de performance, para que você consiga avaliar se os modelos estão comentendo altas taxas de acertos ou erros. 
</p>

<p style="text-align: justify;">
Quando você está elaborando um modelo de regressão, seja ela linear, logística, redes elásticas etc, a típica medida de performance é o RMSE, do inglês: <i>Root Mean Square Error</i>, ou do tupiniquim: Raiz do Erro Quadrático Médio. Ela dá uma ideia dos erros do sistema, dando um maior peso a erros maiores. 
</p>

<p style="text-align: justify;">
A fórmula para computar o RMSE é: 
</p>

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<p style="text-align: justify;">
  $$RMSE (X, h) = \sqrt{\\{1 \over m} {\sum_{i=1}^m (h(x^{(i)}) - y^{(i)})^2}\\}.$$
</p>

<p style="text-align: justify;">
Porém-contudo-todavia, em alguns cenários é possível que outra fórmula funcione melhor. É o caso de amostras com muitos <i>outliers</i>: valores extremos, muito discrepantes dos demais. Em casos como este você deve optar pelo MAE, do inglês: <i>Mean Absolute Error</i> (também conhecido como <i>Average Absolute Deviation</i>, ou em nosso tupiniquim: Erro Absoluto Médio.
</p>

<p style="text-align: justify;">
A fórmula para computar o MAE é: 
</p>

<p style="text-align: justify;">
  $$MAE (X, h) = {1 \over m} \sum_{i=1}^m|h (x^{(i)}) - y^{(i)}|.$$
</p>

<p style="text-align: justify;">
Ambos, o RMSE e o MAE, medem <b>as distâncias entre dois vetores</b>: o vetor dos valores observados (preditos) e o vetor dos valores alvos. Algumas medidas de distância, ou <i>normas</i>, são possíveis: 
</p>

<p style="text-align: justify;">
- Calcular a Raiz do Erro Quadrático Médio corresponde a norma Euclidean, também denominada <i>norma L2</i>, notada ||.||<sub>2</sub> (ou apenas ||.||)
</p>

<p style="text-align: justify;">
- Calcular o Erro Absoluto Médio corresponde a norma Manhattan, também denominada <i>norma L1</i>, notada ||.||<sub>1</sub>. O termo Manhattan é utilizado porque essa medida é a distância entre dois ou mais pontos seguindo blocos ortogonais, <b>ou seja (ufa!)</b>, como querer ir do ponto A ao ponto B em Manhattan a pé, andando pelas quadras (vai reto, vira, vai reto, vira).
</p>

<p style="text-align: justify;">
Psiu: a distância Euclidean, por sua vez, é o mesmo estar em Manhattan, querer chegar ao ponto B e ter o poder de atravessar as paredes (no melhor estilo Kitty Pryde) e dessa forma poder andar diagonalmente. 
</p>

<p style="text-align: justify;">
Para ser mais genérico, a norma <i>L<sub>k</sub></i> de um vetor chamado <i>v</i> é definido como:
</p>

<p style="text-align: justify;">
  $$||v||_k = (|v_0|^k + |v_1|^k + ... + |v_n|^k)^{1 \over k}.$$
</p>

<p style="text-align: justify;">
Por fim, a regra é: quanto maior for a norma, mais ela será focada em valores "grandes" (como <i>outliers</i>) e irá assim negligenciar os valores "menores", os próximos da média. Por isso então que o RMSE (norma L2) é mais sensível a <i>outliers</i> do que o MAE (norma L1).
</p>

<p style="text-align: justify;">
Interessante, né?
</p>


