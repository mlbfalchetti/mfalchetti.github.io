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
