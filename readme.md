# To train
```
python3 train.py --config-name control_ADE20K
```

## Training with nohup
Using nohup prevents the training process from dying when disconnecting from the server
```
nohup python3 train.py --config-name control_ADE20K &> ADE20K.log &
```