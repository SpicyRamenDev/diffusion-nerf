apt-get update
apt-get upgrade -y
apt-get install -y libx11-6
apt-get install -y python-opengl
apt-get install -y python3-opengl
apt-get install -y libglfw3-dev
apt-get install -y libgles2-mesa-dev
apt-get install -y libfreetype-dev
apt-get install -y libglib2.0-0
apt-get install -y software-properties-common
add-apt-repository -y ppa:kisak/kisak-mesa
apt-get update
apt-get upgrade -y
pip install ipywidgets --no-deps
pip install diffusers transformers accelerate
pip install opencv-python==4.5.5.64
pip install pillow --upgrade
apt-get install -y python3-opengl
apt-get install -y libglfw3-dev libgles2-mesa-dev
pip install pyrender
# pip install triton==2.0.0.dev20221120
# pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu117/torch_sparse-0.6.15+pt113cu117-cp38-cp38-linux_x86_64.whl
# pip install xformers
# pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
# pip install xformers-0.0.15.dev0+affe4da.d20221212-cp38-cp38-linux_x86_64.whl
