#------------------------------------------------------------------------------------------------------------------------------------------------cifar10
#lsd training with different alpha and beta, model_num = 3
python train/train_lsd.py       --model_num 3 --ld_coeff 4.0 --alpha 0.5 --beta 0.0 --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 0
python train/train_lsd.py       --model_num 3 --ld_coeff 4.0 --alpha 1.0 --beta 0.0 --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 1
python train/train_lsd.py       --model_num 3 --ld_coeff 4.0 --alpha 2.0 --beta 0.0 --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 0
python train/train_lsd.py       --model_num 3 --ld_coeff 4.0 --alpha 4.0 --beta 0.0 --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 1

python train/train_lsd.py       --model_num 3 --ld_coeff 4.0 --alpha 0.5 --beta 0.5 --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 0
python train/train_lsd.py       --model_num 3 --ld_coeff 4.0 --alpha 1.0 --beta 0.5 --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 1
python train/train_lsd.py       --model_num 3 --ld_coeff 4.0 --alpha 2.0 --beta 0.5 --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 0
python train/train_lsd.py       --model_num 3 --ld_coeff 4.0 --alpha 4.0 --beta 0.5 --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 1

python train/train_lsd.py       --model_num 3 --ld_coeff 4.0 --alpha 0.5 --beta 1.0 --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 0
python train/train_lsd.py       --model_num 3 --ld_coeff 4.0 --alpha 1.0 --beta 1.0 --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 1
python train/train_lsd.py       --model_num 3 --ld_coeff 4.0 --alpha 2.0 --beta 1.0 --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 0
python train/train_lsd.py       --model_num 3 --ld_coeff 4.0 --alpha 4.0 --beta 1.0 --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 1

python train/train_lsd.py       --model_num 3 --ld_coeff 4.0 --alpha 0.5 --beta 2.0 --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 0
python train/train_lsd.py       --model_num 3 --ld_coeff 4.0 --alpha 1.0 --beta 2.0 --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 1
python train/train_lsd.py       --model_num 3 --ld_coeff 4.0 --alpha 2.0 --beta 2.0 --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 0
python train/train_lsd.py       --model_num 3 --ld_coeff 4.0 --alpha 4.0 --beta 2.0 --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 1

python train/train_lsd.py       --model_num 3 --ld_coeff 4.0 --alpha 0.5 --beta 4.0 --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 0
python train/train_lsd.py       --model_num 3 --ld_coeff 4.0 --alpha 1.0 --beta 4.0 --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 1
python train/train_lsd.py       --model_num 3 --ld_coeff 4.0 --alpha 2.0 --beta 4.0 --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 0
python train/train_lsd.py       --model_num 3 --ld_coeff 4.0 --alpha 4.0 --beta 4.0 --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 1

# training of Ensembles
python train/train_baseline.py  --model_num 3 --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 0
python train/train_adp.py       --model_num 3 --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 1
python train/train_gal.py       --model_num 3 --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 0
python train/train_dverge.py    --model_num 3 --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 1
python train/train_trs.py       --model_num 3 --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 0
python train/train_lsd.py       --model_num 3 --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 1

#------------------------------------------------------------------------------------------------------------------------------------------------mnist
#lsd training with different alpha and beta, model_num = 3
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 0.5 --beta 0.0 --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 0
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 1.0 --beta 0.0 --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 1
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 2.0 --beta 0.0 --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 0
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 4.0 --beta 0.0 --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 1

python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 0.5 --beta 0.5 --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 0
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 1.0 --beta 0.5 --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 1
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 2.0 --beta 0.5 --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 0
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 4.0 --beta 0.5 --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 1

python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 0.5 --beta 1.0 --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 0
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 1.0 --beta 1.0 --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 1
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 2.0 --beta 1.0 --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 0
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 4.0 --beta 1.0 --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 1

python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 0.5 --beta 2.0 --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 0
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 1.0 --beta 2.0 --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 1
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 2.0 --beta 2.0 --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 0
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 4.0 --beta 2.0 --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 1

python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 0.5 --beta 4.0 --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 0
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 1.0 --beta 4.0 --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 1
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 2.0 --beta 4.0 --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 0
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 4.0 --beta 4.0 --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 1

# training of Ensembles
python train/train_baseline.py  --model_num 3 --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 0 
python train/train_adp.py       --model_num 3 --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 1
python train/train_gal.py       --model_num 3 --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 0
python train/train_dverge.py    --model_num 3 --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 1
python train/train_trs.py       --model_num 3 --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 0
python train/train_lsd.py       --model_num 3 --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 1

#------------------------------------------------------------------------------------------------------------------------------------------------fmnist
#lsd training with different alpha and beta, model_num = 3
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 0.5 --beta 0.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 1.0 --beta 0.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 2.0 --beta 0.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 4.0 --beta 0.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 
                                                                                                          30                                  0 
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 0.5 --beta 0.5 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 1.0 --beta 0.5 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 2.0 --beta 0.5 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 4.0 --beta 0.5 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 
                                                                                                          30                                  0 
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 0.5 --beta 1.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 1 
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 1.0 --beta 1.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 1 
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 2.0 --beta 1.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 1 
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 4.0 --beta 1.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 1 
                                                                                                          30                                  0 
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 0.5 --beta 2.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 1.0 --beta 2.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 2.0 --beta 2.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 4.0 --beta 2.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 
                                                                                                          30                                  0 
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 0.5 --beta 4.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 1.0 --beta 4.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 2.0 --beta 4.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 
python train/train_lsd.py       --model_num 3 --ld_coeff 3.0 --alpha 4.0 --beta 4.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0

# training of Ensembles
python train/train_baseline.py  --model_num 3 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 2
python train/train_adp.py       --model_num 3 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 2
python train/train_gal.py       --model_num 3 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 2
python train/train_dverge.py    --model_num 3 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 2
python train/train_trs.py       --model_num 3 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 2
python train/train_lsd.py       --model_num 3 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 2

# training of Ensembles
python train/train_baseline.py  --model_num 5 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 2
python train/train_adp.py       --model_num 5 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 2
python train/train_gal.py       --model_num 5 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 2
python train/train_dverge.py    --model_num 5 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 2
python train/train_trs.py       --model_num 5 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 2
python train/train_lsd.py       --model_num 5 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 2

python train/train_lsd.py       --model_num 5 --ld_coeff 3.0 --alpha 0.5 --beta 0.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 
python train/train_lsd.py       --model_num 5 --ld_coeff 3.0 --alpha 1.0 --beta 0.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 1 
python train/train_lsd.py       --model_num 5 --ld_coeff 3.0 --alpha 2.0 --beta 0.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 1 
python train/train_lsd.py       --model_num 5 --ld_coeff 3.0 --alpha 4.0 --beta 0.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 1 

python train/train_lsd.py       --model_num 5 --ld_coeff 3.0 --alpha 0.5 --beta 0.5 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 1
python train/train_lsd.py       --model_num 5 --ld_coeff 3.0 --alpha 1.0 --beta 0.5 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 1 
python train/train_lsd.py       --model_num 5 --ld_coeff 3.0 --alpha 2.0 --beta 0.5 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 1 
python train/train_lsd.py       --model_num 5 --ld_coeff 3.0 --alpha 4.0 --beta 0.5 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 1 

python train/train_lsd.py       --model_num 5 --ld_coeff 3.0 --alpha 0.5 --beta 1.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 1
python train/train_lsd.py       --model_num 5 --ld_coeff 3.0 --alpha 1.0 --beta 1.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 1 
python train/train_lsd.py       --model_num 5 --ld_coeff 3.0 --alpha 2.0 --beta 1.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 1 
python train/train_lsd.py       --model_num 5 --ld_coeff 3.0 --alpha 4.0 --beta 1.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 1 

python train/train_lsd.py       --model_num 5 --ld_coeff 3.0 --alpha 0.5 --beta 2.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 1
python train/train_lsd.py       --model_num 5 --ld_coeff 3.0 --alpha 1.0 --beta 2.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 1 
python train/train_lsd.py       --model_num 5 --ld_coeff 3.0 --alpha 2.0 --beta 2.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 1 
python train/train_lsd.py       --model_num 5 --ld_coeff 3.0 --alpha 4.0 --beta 2.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 1 

python train/train_lsd.py       --model_num 5 --ld_coeff 3.0 --alpha 0.5 --beta 4.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0
python train/train_lsd.py       --model_num 5 --ld_coeff 3.0 --alpha 1.0 --beta 4.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 
python train/train_lsd.py       --model_num 5 --ld_coeff 3.0 --alpha 2.0 --beta 4.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 
python train/train_lsd.py       --model_num 5 --ld_coeff 3.0 --alpha 4.0 --beta 4.0 --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0 
