#-- CHROME

wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb

sudo apt-get install ./google-chrome-stable_current_amd64.deb

#-- ANACONDA

sudo apt-get update

sudo apt-get install curl
cd /tmp
curl –O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh

bash Anaconda3-2020.02-Linux-x86_64.sh

after r install
sudo apt-get update -y
sudo apt-get install -y libxml2-dev
sudo apt-get install -y libssl-dev
sudo apt-get install -y libcurl4-openssl-dev



#--
sudo apt-get install catdoc
xls2csv -c/, file_name.xls
xls2csv -c/, file_name.xls > file_name.csv
man xls2csv


#--
AWK
linguagem para manipulação de arquivos, flexibilidade, performance em arquivos grandes comparado com shell, 
linguagem orientada a linha, cada linha será processada para o que tem dentro da {} 

awk '{print $0}'
ls -l | awk '{print $5}'

vi arquivo.awk

{
  print $0
}
{
  print "bloco repetido", $0
}
awk -f arquivo.awk arquivo_que_quero_executar.txt

/PAULO/ {comandos}
$1 == 455 {comandos}

BEGIN e END
BEGIN {comandos}
END {comandos}

vi fitro_expr.awk
/PAULO/{
  print $0
}
awk -f filtro_expr.awk arquivo_que_quero_executar.txt

vi fitro_expr.awk
/o/{
  print $0
}
awk -f filtro_expr.awk arquivo_que_quero_executar.txt

vi fitro_expr.awk
/^.2/{
  print $0
}
awk -f filtro_expr.awk arquivo_que_quero_executar.txt
(segundo caracter é 2)

vi fitro_expr.awk
$1 == 840 || $1 == 454{
  print $0
}
awk -f filtro_expr.awk arquivo_que_quero_executar.txt
(ou)

vi x.awk
BEGIN{
  print "CODIGO NOME TELEFONE"
}
{
  print $0
}
awk -f filtro_expr.awk arquivo_que_quero_executar.txt

vi x.awk
{
  print $0
}
END{
  print "FIM DO ARQUIVO"
}
awk -f filtro_expr.awk arquivo_que_quero_executar.txt
