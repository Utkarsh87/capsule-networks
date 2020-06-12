This repository contains scripts for experimenting on the ![Kannada MNIST and DIG-10K MNIST datasets](https://towardsdatascience.com/a-new-handwritten-digits-dataset-in-ml-town-kannada-mnist-69df0f2d1456).

![Original paper](https://arxiv.org/pdf/1908.01242.pdf)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Progress:


|    Version    |     Epochs    | Val accuracy  |    Val loss   | Test accuracy |  DIG accuracy |    Changes    |     Comments  |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
|     1.0.1     |       12      |     99.50%    |    0.00450    |     97.84%    |     81.12%    |       -       |       -       |
|     1.0.2     |       20      |     99.64%    |    0.00365    |     98.27%    |     86.54%    | data augmentation | better generalisation; recon loss still a pain point |
|     1.0.3     |       20      |     99.66%    |    0.00358    |     98.13%    |     85.13%    | deconv decoder instead of dense network | improvement on recon loss, promising, train for more epochs |
|     1.0.4     |       30      |     99.65%    |    0.00347    |     98.44%    |     84.93%    | more epochs, change val split to 0.1 | best test acc, drastic decrease in dig acc, maybe try other augmentations |

NOTE: performance metrics reported are top-1 metrics

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


Model architecture in use(1.0.2):
![Model architecture in use:](https://github.com/Utkarsh87/Capsule-Networks/blob/master/kannada%20mnist/images/model.png)
