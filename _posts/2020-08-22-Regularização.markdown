---
layout: post
title:  "Regularização"
date:   2020-08-30 14:00:00 -0300
categories: deep-learning regularization
---
###

<p style="text-align: justify;">
Para lembrar, quando as análises da rede neural apresentam resultados ruins de acurácia no grupo de <i>treino</i> (<i>train set error</i>) você está usando um modelo com alto viés, <i>high bias</i>, porém quando apresentam resultados ruins no grupo de <i>dev</i> ou <i>teste</i> (<i>dev/teste set error</i>) você está usando um modelo com alta variância, <i>high variance</i>, e esta está <i>over fitting</i> os dados. 
</p>

<p style="text-align: justify;">
Caso você possua um modelo de <i>high variance</i>, você pode tentar diminuir o <i>over fitting</i> 1 - buscando mais dados para o treino ou 2 - operando técnicas de regularização. Como você pode imaginar, as vezes não é possível adquirir mais dados, <i>p.e.</i> obter novas amostras pode ser caro, pode não se ter mais acesso a fonte desse dado e, sendo assim, a regularização acaba sendo a solução ideal. 
</p>

### Como "regularizar" (Regularização L1 e L2)?

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
Para <b>regularização L1</B> na regressão logística se adiciona:
</p>

<p style="text-align: justify;">
  $$+ {\lambda \over 2m} ||w||_1.$$
</p>

<p style="text-align: justify;">
Tendo assim:
</p>

<p style="text-align: justify;">
  $$J(w,b) = {1 \over m} \sum_{i=1}^m L(\hat{y} - y) \color{red} + {\lambda \over 2m} ||w||_1.$$
</p>

<p style="text-align: justify;">
Para <b>regularização L2</B> na regressão logística se adiciona:
</p>

<p style="text-align: justify;">
  $$+ {\lambda \over 2m} ||w||_2^2.$$
</p>

<p style="text-align: justify;">
Tendo assim:
</p>

<p style="text-align: justify;">
  $$J(w,b) = {1 \over m} \sum_{i=1}^m L(\hat{y} - y) \color{red} + {\lambda \over 2m} ||w||_2^2.$$
</p>

<p style="text-align: justify;">
&lambda; é o <i>parâmetro de regularização</i> e é um hiperparâmetro ajustado de forma empirica; <i>2m</i> é apenas um fator de escala. Assim sendo, o que diferencia a regularização L1 da L2 é o ||<i>w</i>||<sub>1</sub>, ou <i>norma</i> L1 do parâmetro <i>w</i>, na regularização L1 e o ||<i>w</i>||<sub>2</sub>², ou quadrado da <i>norma</i> L2 do parâmetro <i>w</i>, na regularização L2. Essas diferenças, bem como seus resultados, determinam quando L1 e L2 são geralmente utilizados. <b>A ausência do quadrado na norma L1 faz com que o vetor <i>w</i>) acabe sendo "esparso", ou seja, cheio de zeros (0). Isso pode ser utilizado na compressão de modelos, porque quanto mais parâmetros forem zeros, será preciso menos memória para armazená-los. 
</p>






<p style="text-align: justify;">
Sendo:
</p>

<p style="text-align: justify;">
  $$||x||_2^2 = \sum_{j=1}^{n_x}w_j^2 = w^Tw .$$
</p>

### Por que regularizar só o parâmetro <i>w</i>?

<p style="text-align: justify;">
Geralmente se omite as regularizações em <i>b</i> por este ser um valor único enquanto o parâmetro <i>w</i> é, mais uma vez geralmente, um vetor com uma elevada dimensionalidade, ou seja, com muitos parâmetros da rede neural. Sendo assim, quase todos os parâmetros estão em <i>w</i> e adicionar a regularização em <i>b</i> não faz uma grande diferença. Claro, você pode adicionar, se quiser.
</p>


<p style="text-align: justify;">
Aqui, <b>a ausência do quadrado da norma euclideana do vetor <i>w</i></b> faz com que ele (vetor <i>w</i>) acabe sendo "esparso", ou seja, acabe sendo um vetor cheio de zeros. Algumas pessoas utilizam isso para comprimir modelos, porque quanto mais parâmetros forem zeros, será preciso menos memória para armazená-los.
</p>
