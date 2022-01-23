## Aggregating Global Features into Local Vision Transformer(MOA-Transformer)

### Requirements
   python = 3.7 <br />
   pytorch >= 0.4 <br />
   Cuda = 10.2 <br />
   timm = 0.3.2 <br />
   apex <br />
   
### Datapreperation
```
ImageNet
└───Train
   └───Class1
       │   image111.jpg
       │   image112.jpg
       │ ...
   └───Class2
        │   image113.jpg
        │   image114.jpg
        │ ...
   └───....   
└───Val
   └───Class1
       │   image115.jpg
       │   image116.jpg
       │ ...
   └───Class2
       │   image117.jpg
       │   image118.jpg
       │ ...
   └───....  
```

### Usage

1. Train : 

```
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  mainup.py \
--cfg configs/MOA_tiny_patch4_window14_224.yaml --data-path <imagenet-path> --batch-size 128
```
2. Evaluate:

```
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 mainup.py --eval \
--cfg configs/MOA_tiny_patch4_window14_224.yaml --resume <checkpoint> --data-path <imagenet-path> 
```        

    





