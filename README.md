# Dacon_Super_Resolution
## Team : 꽃미남트리오
- Oh_kingtaek, 제로원, 장원석그는가히전설이라말할수있다
- private score : 24.25905(7th)
## 개발 환경
---
- CPU: Intel® Xeon® E5-2698 v4 @2.20GHz

- GPU: NVIDIA V100(32GB) 4EA
## Library import & create Environment
---
```
python 3.9.5
pytorch 1.11.0
cuda 11.3
```
```
git clone https://github.com/ohkingtaek/Dacon_Super_Resolution.git
cd Dacon_SR
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```
## Data Prepration
---
```
python create_lmdb.py
```

## Commands for train & test
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/SwinIR/train_SwinIR_SRx4_dacon.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/test.py -opt options/train/SwinIR/test_SwinIR_x4.yml --launcher pytorch
```