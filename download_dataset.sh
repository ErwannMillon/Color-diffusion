#run from root of repo
#requires gdown installed (pip install gdown, or use conda if the pip install fails)
pip install gdown
mkdir celeba
gdown 11NjbMNrOD0LlgsNLqGr1WA3VE4TQIT4g
unzip celeba.zip -d ./celeba
mv celeba/img_align_celeba ./
rm -rf celeba celeba.zip
