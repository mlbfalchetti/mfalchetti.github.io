---
layout: post
title:  "Scripts UNIX - parte 01"
date:   2020-08-22 14:00:00 -0300
categories: jekyll update
---

{: style="text-align: justify" }

## Enter e já foi

Olá, aqui falarei um pouco sobre scripts de UNIX para bash shell. 

Um script é um arquivo que permite a construção de esquemas de execução complexos. Esses arquivos geralmente contém um texto curto, de 10 a 100 linhas, com uma série de comandos bash e é convencional que contenham a extensão .sh, sendo rodados em shell chamando:

```sh 
bash script.sh
```

Entretanto, caso i) eles contenham na primeira linha do script o chamado *shebang* (ou *hashpound*) 

```sh
#!/bin/bash # p.e. para scripts com códigos em bash
ou 
#!/usr/bin/python3 # p.e. para scripts com códigos em python3
```

e ii) tenham sido marcados como arquivos executáveis (chmod, "*change mode*" e +x, "*executable*")

```sh
chmod +x script.sh
```

eles podem então serem chamados de modo direto 

```sh
script.sh
``` 

## Mas, por que? 

Utilizando um script você pode agrupar funcões em um arquivo, p.e. é possível montar um script capaz de realizar todas as etapas de uma boa análise de expressão diferencial de um experimento de RNA-Seq, com cada linha comentada para futuros estudos, onde você pode explicar as razões, as justificativas e as decisões tomadas pelo caminho. 

PS: Comentários são feitos com o "#", indicando que o que há após esse símbolo não deve ser executado.

