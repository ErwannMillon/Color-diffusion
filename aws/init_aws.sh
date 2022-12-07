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
sudo yum install zsh
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
echo 'alias python=python3'>> ~/.zshrc
echo 'alias pip=pip3'>> ~/.zshrc
git clone https://github.com/ErwannMillon/dotfiles.git
cat dotfiles/vimrc >> ~/.vimrc
pip3 install -r requirements.txt


