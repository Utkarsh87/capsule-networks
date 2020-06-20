<b>Results:</b><br>

|    Script(ipynb file)    | Kannada MNIST test set accuracy |  DIG-10k test set accuracy | Digit manipulation results(folder name) |
| --------------------------------------  | :----------------------: | :----------------------: | ------------------------------ |
|    capsnet with aug    |   98.52%  |     91.09%   |   manipulate(with augmentation)   |
|     capsnet without aug      |   97.98%       |      81.73%       |  manipulate(without augmentation) |


<b>Comments:</b><br>
The DIG-10K set has images that are randomly zoomed-in/out, cropped at certain places and all sorts of such variations<br>
The image data augmentation process helps the model become robust to some of those changes and hence the DIG set accuracy is significantly better when the data is augmented first.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The scripts reside in the <b>scripts</b> folder with the same names as mentioned above.<br>
Use the weights for the 2 scripts as instructed in the <b>README inside the weights folder.</b><br>

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

<b>Model architecture</b>(uses a Capsule network for learning the features and a Transposed convolutional network as a reconstruction and regularisation network)(1.0.5):

<b>Encoder</b>(Convolution + Primary Capsule + Digit Capsule layers)
![Encoder(1.0.5)](https://github.com/Utkarsh87/Capsule-Networks/blob/master/kannada%20mnist/images/capsnet.png)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

<b>Decoder/Reconstruction</b>(Transposed convolutional network)

![Decoder/Reconstruction(1.0.5)](https://github.com/Utkarsh87/Capsule-Networks/blob/master/kannada%20mnist/images/decoder.png)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
