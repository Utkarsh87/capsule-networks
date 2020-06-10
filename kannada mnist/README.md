This repository contains scripts for experimenting on the ![Kannada MNIST and DIG-10K MNIST datasets](https://towardsdatascience.com/a-new-handwritten-digits-dataset-in-ml-town-kannada-mnist-69df0f2d1456).
Original paper: https://arxiv.org/pdf/1908.01242.pdf
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Progress:


|Version| Script | Epochs | Validation accuracy | Validation loss | Test set accuracy | DIG set accuracy | Comments
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | 
| 1.0.1 (CapsuleNet Kannada MNIST) | Basic capsule network | 12 | 99.5% | 0.00450 | 97.84% | 81.12% | - |
| 1.0.2 (CapsuleNet Kannada MNIST with Augmentation) | Capsule network with image data augmentation | 20 | 99.64% | 0.00365 | 98.27% | 86.54% | better generalisation |

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


Model architecture in use(1.0.2):
![Model architecture in use:](https://github.com/Utkarsh87/Capsule-Networks/blob/master/kannada mnsit/images/model.png)
