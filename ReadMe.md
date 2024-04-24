For Sck-Music-Analysis

Activate GPU with interactive session:
srun -p compsci-gpu --gres=gpu:1 --pty bash -i

Install Conda Environment
conda create -n myenv –file package-list.txt

Activate Your Conda Env before training
conda activate myenv

Train:
python train.py
If you want to change the weight of the loss:
CLASS_WEIGHT = [first cluster wight, second cluster weight]

Plot with PaCMAP
