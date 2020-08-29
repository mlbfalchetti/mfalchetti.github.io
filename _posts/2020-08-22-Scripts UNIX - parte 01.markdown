---
layout: post
title:  "Scripts UNIX - parte 01"
date:   2020-08-22 14:00:00 -0300
categories: basics UNIX
---
### Enter e já foi

<p style="text-align: justify;">
Olá, aqui falarei um pouco sobre scripts de UNIX para bash shell. 
</p>

<p style="text-align: justify;">
Um script é um arquivo que permite a construção de esquemas de execução complexos. Esses arquivos geralmente contém um texto curto, de 10 a 100 linhas, com uma série de comandos bash e é convencional que contenham a extensão .sh, sendo rodados em shell chamando:
</p>

```sh 
bash script.sh
```

<p style="text-align: justify;">
Entretanto, caso 1 - eles contenham na primeira linha do script o chamado <i>shebang</i> (ou <i>hashpound</i>) 
</p>

```sh
#!/bin/bash # p.e. para scripts com códigos em bash
ou 
#!/usr/bin/python3 # p.e. para scripts com códigos em python3
```

<p style="text-align: justify;">
e 2 - tenham sido marcados como arquivos executáveis (chmod, "<i>change mode</i>" e +x, "<i>executable</i>")
</p>

```sh
chmod +x script.sh
```

<p style="text-align: justify;">
eles podem então serem chamados de modo direto 
</p>

```sh
script.sh
``` 

### Mas, por que? 

<p style="text-align: justify;">
Utilizando um script você pode agrupar funcões em um arquivo, p.e. é possível montar um script capaz de realizar todas as etapas de uma boa análise de expressão diferencial de um experimento de RNA-Seq, com cada linha comentada para futuros estudos, onde você pode explicar as razões, as justificativas e as decisões tomadas pelo caminho. 
</p>

<p style="text-align: justify;">
PS: Comentários são feitos com o "#", indicando que o que há após esse símbolo não deve ser executado.
</p>
