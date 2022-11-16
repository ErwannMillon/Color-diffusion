# sudo yum update -y
# sudo yum groupinstall "Development Tools" -y
# sudo yum erase openssl-devel -y
# sudo yum install openssl11 openssl11-devel  libffi-devel bzip2-devel wget -y
# wget https://www.python.org/ftp/python/3.10.4/Python-3.10.4.tgz
# tar -xf Python-3.10.4.tgz
# cd Python-3.10.4/
# ./configure --enable-optimizations
# make -j $(nproc)
# sudo make altinstall
sudo apt-get install zsh
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
apt-get update
export PATH="/home/ubuntu/.local/bin:$PATH"
wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
echo 'alias python=python3'>> ~/.zshrc
echo 'alias pip=pip3'>> ~/.zshrc
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip3 install -r requirements.txt


