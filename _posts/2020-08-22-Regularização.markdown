---
layout: post
title:  "Regularização"
date:   2020-08-26 14:00:00 -0300
categories: deep-learning regularization
---

<p style="text-align: justify;">
Para lembrar, quando as análises da rede neural apresentam resultados ruins de acurácia no grupo de <i>treino</i> (<i>train set error</i>) você está usando um modelo com alto viés, <i>high bias</i>, porém quando apresentam resultados ruins no grupo de <i>dev</i> ou <i>teste</i> (<i>dev/teste set error</i>) você está usando um modelo com alta variância, <i>high variance</i>, e esta está <i>over fitting</i> os dados. 
</p>

<p style="text-align: justify;">
Caso você possua um modelo de <i>high variance</i>, você pode tentar diminuir o <i>over fitting</i> 1 - buscando mais dados para o treino ou 2 - operando técnicas de regularização. Como você pode imaginar, as vezes não é possível adquirir mais dados, <i>p.e.</i> obter novas amostras pode ser caro, pode não se ter mais acesso a fonte desse dado e, sendo assim, a regularização acaba sendo a solução ideal. 
</p>

<p style="text-align: justify;">
Mas o que é regularização em rede neural?
</p>

<p style="text-align: justify;">
Para isso, vale recordar que nós sempre queremos <b>diminuir o valor do <i>custo</i></b> da rede neural, ou seja, nós sempre queremos diminuir as diferenças entre os valores obtidos e os esperados. Por exemplo, no caso de uma regressão logística a função desse custo, ou <i>cost function</i> (J) é definida como: 
</p>

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<p style="text-align: justify;">
  $$J(w,b) = {1 \over m} \sum_{i=1}^m L(\hat{y} - y).$$
</p>

<p style="text-align: justify;">
Sendo <i>w</i> e <i>b</i> os parâmetros da regressão logística (<i>weight</i> e <i>bias</i>, respectivamente), onde aqui <i>w</i> representa um vetor de valores e <i>b</i> um valor único. A fórmula resulta a soma das diferenças entre os valores obtidos e os esperados e a divide pelo número de amostras. Dessa forma, temos a média das diferenças ou das <i>losses</i>.
</p>

<p style="text-align: justify;">
  $$J(w,b) = {1 \over m} \sum_{i=1}^m L(\hat{y} - y) \color{red} + {\lambda \over 2m} ||w||_2^2.$$
</p>
