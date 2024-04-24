# Music-Analysis With SA


### 1. Login to cluster:
ssh netid@login.cs.duke.edu

### 2. Activate GPU with Interactive Session:
srun -p compsci-gpu --gres=gpu:1 --pty bash -i

### 3. Install Conda Environment(only install once)
#### Download Miniconda
https://docs.anaconda.com/free/miniconda/index.html
Remeber to pick the "Linux" one.

#### Create New Working Environment
conda create -n myenv –file package-list.txt

### 4. Activate Your Conda Env before Training
conda activate myenv

### 5. Train:
python train.py

### 6. Experiments:

If you want to change the weight of the loss:
python train.py --> CLASS_WEIGHT = [first layer wight, second layer weight]

### 7. Plot with PaCMAP
python plot.py
