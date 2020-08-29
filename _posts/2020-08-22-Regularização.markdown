---
layout: post
title:  "Regularização"
date:   2020-08-26 14:00:00 -0300
categories: deep-learning regularization
---

<p style="text-align: justify;">
Para lembrar, quando as análises da rede neural apresentam resultados ruins de acurácia no grupo de <i>treino</i> (<i>train set error</i>) você está usando um modelo com alto viés, <i>high bias</i>, porém quando os resultados ruins no grupo de <i>dev</i> ou <i>teste</i> (<i>dev/teste set error</i>) você está usando um modelo com alta variância, <i>high variance</i>, e esta está <i>over fitting</i> os dados. 
</p>

<p style="text-align: justify;">
Caso você possua um modelo de <i>high variance</i>, você pode tentar diminuir o <i>over fitting</i> 1 - buscando mais dados para o treino ou 2 - operando técnicas de regularização.
</p>
