#-------------------------------------------------------------------------------------------------------------------------------cifar10
# split_test_index
python eval/split_test_index.py --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 0

# surrogate
python eval/surrogate.py --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 1

# para_compare
python eval/wbox_para_compare.py --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 1
python eval/bbox_para_compare.py --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 1

#evaluation of black-box robustness
python eval/bbox.py --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 1

#evaluation of white-box robustness
python eval/wbox.py --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 1

#evaluation of transferability
python eval/transferability.py --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 0 

#natural_acc
python eval/natural_acc.py --arch ResNet --epochs 200 --dataset cifar10 --depth 20 --gpu 1
#-------------------------------------------------------------------------------------------------------------------------------mnist
# split_test_index
python eval/split_test_index.py --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 1

#surrogate
python eval/surrogate.py --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 0

# para_compare
python eval/wbox_para_compare.py --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 0 
python eval/bbox_para_compare.py --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 0

#evaluation of black-box robustness
python eval/bbox.py --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 0

#evaluation of white-box robustness
python eval/wbox.py --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 1

#evaluation of transferability
python eval/transferability.py --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 0

#natural_acc
python eval/natural_acc.py --arch LeNet --epochs 20 --dataset mnist --depth 5 --gpu 0
#-------------------------------------------------------------------------------------------------------------------------------fmnist
# split_test_index
python eval/split_test_index.py --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0

#surrogate
python eval/surrogate.py --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0

# para_compare
python eval/wbox_para_compare.py --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 1
python eval/bbox_para_compare.py --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 1

#evaluation of black-box robustness
python eval/bbox.py --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 1

#evaluation of white-box robustness
python eval/wbox.py --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 1

#evaluation of transferability
python eval/transferability.py --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 1

#natural_acc
python eval/natural_acc.py --arch LeNet --epochs 30 --dataset fmnist --depth 5 --gpu 0
