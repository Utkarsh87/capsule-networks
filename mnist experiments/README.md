Using the Dynamic Routing ALgorithm proposed in the paper as is:
![Using the Dynamic Routing ALgorithm proposed in the paper as is:](https://github.com/Utkarsh87/Capsule-Networks/blob/master/mnist%20experiments/images/Dynamic%20Routing.PNG)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Current progress:

| Conv layers  | Kernels | Kernel sizes | Epochs | Routing iterations | Val accuracy | Val loss | Comments |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 1  |  256 |  9  |  10  |  3   |  99.33%   |  0.0076  | first result |
| 2  |  256 |  9  |  15  |  3   |  99.42%   |  0.0052  | increased conv layer, not much help, not the motive |
 

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Model architecture in use:(for trials 1 and 3, trial 2 had 2 conv layers in succession before feeding it to primary caps)
![Model architecture in use:](https://github.com/Utkarsh87/Capsule-Networks/blob/master/mnist%20experiments/images/model1.png)
