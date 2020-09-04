---
layout: post
title:  "Regularização  L1 e L2"
date:   2020-08-30 14:00:00 -0300
categories: deep-learning regularization
---
### Decorar ou aprender

<p style="text-align: justify;">
Para lembrar, quando as análises da rede neural apresentam resultados ruins de acurácia no grupo de <i>treino</i> (<i>train set error</i>) você está usando um modelo com alto viés, <i>high bias</i>, porém quando apresentam resultados ruins no grupo de <i>dev</i> ou <i>teste</i> (<i>dev/test set error</i>) você está usando um modelo com alta variância, <i>high variance</i>, e esta está <i>over fitting</i> os dados. 
</p>

<p style="text-align: justify;">
<i>Overfitting: Quando o modelo não <i>generaliza</i> bem, ou seja, o modelo possui alta acurácia para amostras que foi treinado, mas não outras, os chamados "dados reais". Isso porque o modelo está "justo" demais ao treino, ele "gravou bem as características apenas destas amostras, abrangendo as características "úteis" à classificação/aprendizado e as "inúteis" como os ruídos/<i>noises</i> destas.</i>  
</p>

<p style="text-align: justify;">
Caso você possua um modelo de <i>high variance</i>, você pode tentar diminuir o <i>over fitting</i> 1 - buscando mais dados para o treino ou 2 - operando técnicas de regularização. Como você pode imaginar, as vezes não é possível adquirir mais dados, <i>p.e.</i> obter novas amostras pode ser caro, pode não se ter mais acesso a fonte desse dado e, sendo assim, a regularização acaba sendo a solução ideal. 
</p>

<p style="text-align: justify;">
Regularizar é basicamente penalizar os valores de <i>w</i> (os pesos), diminuindo-os. Dessa forma um modelo criado ainda identifica os parâmetros aprendidos, porém "dá menos importância" para eles. <i>Não leva eles tão a sério</i>, que era o que fazia ele atuar bem só em amostras que foi treinado. 
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
&lambda; é o <i>parâmetro de regularização</i> e é um hiperparâmetro ajustado de forma empirica; <i>2m</i> é apenas um fator de escala. <b>Assim sendo, o que diferencia a regularização L1 da L2 é o ||<i>w</i>||<sub>1</sub>, ou <i>norma</i> L1 do parâmetro <i>w</i>, na regularização L1 e o ||<i>w</i>||<sub>2</sub>², ou quadrado da <i>norma</i> L2 do parâmetro <i>w</i>, na regularização L2</b>. Essas diferenças, bem como seus resultados, determinam quando L1 e L2 são geralmente utilizados. <b>A ausência do quadrado na norma L1 faz com que o vetor <i>w</i> acabe sendo "esparso"</b>, ou seja, cheio de zeros (0). Isso pode ser utilizado na compressão de modelos, porque quanto mais parâmetros forem zeros, menor é o total de memória necessária para armazená-los. 
</p>

### Como a Regularização L1 e L2 previnem o *overfitting*?

<p style="text-align: justify;">
O valor &lambda; influenciará os valores de <i>w</i>, das matrizes dos pesos. Uma vez que &lambda; for 1 - um valor muito grande, os valores de <i>w</i> serão muito próximos a zero, e sendo assim os impactos dessas <i>hidden units</i>, ou neurônios, estarão sendo (quase) zerados. Dessa forma você acaba tendo uma rede com "menos parâmetros", "menor", simplificada. Se &lambda; for 2 - um valor muito baixo, os efeitos da regularização sobre os pesos é pequeno. O objetivo então é determinar um valor de &lambda; que produza modelos intermediários, não muito simples ou complexos, que tenha aprendido o essencial dos dados. 
</p>

<p style="text-align: justify;">
Outro importante ponto é que a Regularização L1 e L2 pode tornar os resultados de funções de ativação (como sigmóide ou <i>tanh</i>) análogos a uma função linear. Para lembrar, em cada camada da rede neural você calcula a função linear (Z<sup>[L]</sup> = W<sup>[L]</sup> * A<sup>[L - 1]</sup> + B<sup>[L]</sup>) e a função de ativação. Caso W represente valores próximos a zero (após regularização), Z<sup>[L]</sup> representará valores também próximos a zero, e como aqui representado, o centro da função de ativação tanh, por exemplo, é algo próximo a uma função linear (os valores de <i>x</i> próximos a zero).  Uma rede com camadas de função linear não alcança padrões muito complexos, fazendo modelos com muito <i>overfitting</i>. Interessante, né?
</p>

<p class="aligncenter">
  <img src="/imagens/tanh_x.png" alt="imagem" style="width:350px;height:350px;">
</p>
<style>
.aligncenter {
    text-align: center;
}
</style>

### Curiosidades: Por que regularizar só o parâmetro <i>w</i>?

<p style="text-align: justify;">
Geralmente se omite as regularizações em <i>b</i> por este ser um valor único enquanto o parâmetro <i>w</i> é, mais uma vez geralmente, um vetor com uma elevada dimensionalidade, ou seja, com muitos parâmetros da rede neural. Sendo assim, quase todos os parâmetros estão em <i>w</i> e adicionar a regularização em <i>b</i> não faz uma grande diferença. Claro, você pode adicionar, se quiser.
</p>

