The scripts in this folder implement the novel <b>Capsule Network</b> architecture to the experiment on the ![Kannada MNIST and DIG-10K MNIST datasets](https://towardsdatascience.com/a-new-handwritten-digits-dataset-in-ml-town-kannada-mnist-69df0f2d1456).

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Progress is logged below:


|    Version    |     Epochs    | Val accuracy  |    Val loss   | Test accuracy |  DIG accuracy |    Changes    |     Comments  |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
|     1.0.1     |       12      |     99.50%    |    0.00450    |     97.84%    |     81.12%    |       -       |  Basic script |
|     1.0.2     |       20      |     99.64%    |    0.00365    |     98.27%    |     86.54%    | data augmentation | better generalisation; recon loss still a pain point |
|     1.0.3     |       20      |     99.66%    |    0.00358    |     98.13%    |     85.13%    | deconv decoder deployed instead of dense network | improvement on recon loss, promising, train for more epochs, will be sticking with deconv recon network hereon |
|     1.0.4     |       30      |     99.65%    |    0.00347    |     98.44%    |     84.93%    | more epochs, change val split to 0.1 | best test acc, drastic decrease in dig acc, maybe try other augmentations |
|     1.0.5     |       35      |     99.60%    |    0.00386    |     98.44%    |     91.23%    | nadam optimizer, more data augmentation | matches best test acc and by far the best dig acc, still underfitting, train for more epochs. Val loss lesser than train loss due to the high data augmentation making it tougher to learn features |
|     1.0.6     |       70      |     99.68%, 99.67%    |    0.00350, 0.00380    |     98.46%, 98.50%    |     91.11%, 91.03%    |      trained previous architecture for more epochs     |  probably reached the threshold performance using this architecture, the <b>best-weights6.h5</b> file holds the model weights with the <b>better dig set performance</b>, the <b>trained_model6.h5</b> file holds the weights for the <b>best test set performance yet</b> |
|     1.0.7     |       70      |     99.64%    |    0.0116    |     98.52%    |     91.09%    | improved viz, increased the weight of the recon loss which led to much better reconstructions of the input images, introduced manipulation of recon images tweaking the dimensions of the output vectors of digit capsule layer, introduced modularity | model probably overfit after 50 epochs, achieves the best test set accuracy yet, weights for the same are in <b>best.h5</b>, the increased weight of the recon loss is the reason for the abnormally high val loss but the individual losses attain values as expected |


NOTE: performance metrics reported are top-1 metrics

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Model architecture(uses a Transposed convolutional network as a reconstruction network)(1.0.5):

Encoder(Convolution + Primary Capsule + Digit Capsule layers)
![Encoder(1.0.5)](https://github.com/Utkarsh87/Capsule-Networks/blob/master/kannada%20mnist/images/model2.png)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Decoder/Reconstruction(Transposed convolutional network)

![Decoder/Reconstruction(1.0.5)](https://github.com/Utkarsh87/Capsule-Networks/blob/master/kannada%20mnist/images/decoder.png)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


Model architecture with dense layers as reconstruction/decoder network(old || 1.0.2):
![Model architecture in use(1.0.2)](https://github.com/Utkarsh87/Capsule-Networks/blob/master/kannada%20mnist/images/model.png)


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

