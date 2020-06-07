Using the Dynamic Routing ALgorithm proposed in the paper as is:
![Using the Dynamic Routing ALgorithm proposed in the paper as is:](https://github.com/Utkarsh87/Capsule-Networks/blob/master/mnist%20experiments/images/Dynamic%20Routing.PNG)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Current progress:

|Version| Conv layers before primary caps | Kernels | Kernel sizes | Epochs | Routing iterations | Val accuracy | Val loss | Changes | Comments |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 1.0.1  | 1  |  256 |  9  |  10  |  3   |  99.33%   |  0.0076  | - | first result |
| 1.0.2  | 2  |  256 |  9  |  15  |  3   |  99.42%   |  0.0052  | increased conv layer before primary caps | not much help, not the motive |
| 1.0.3  | 1  |  256 |  9  |  10  |  3   |  99.24%   |  0.0081  | increased digit caps dimensions(16->32) | training time almost 2x, good trend, promising |
| 1.0.4  | 1  |  256 |  9  |  10  |  1   |  99.41%   |  0.0080  | kept routing iterations limited to 1 | slower than expected, higher accuracy than expected |
| 1.0.5  | 1  |  256 |  9  |  15  |  3   |  99.45%   |  0.0059  | routing iterations bumped back up to 3 | best result yet, still underfitting, try more epochs |
 

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Model architecture in use(in majority of the versions):
![Model architecture in use:](https://github.com/Utkarsh87/Capsule-Networks/blob/master/mnist%20experiments/images/model1.png)
