### Overview

The implement of the data valuation algorithms presented in  Section 3"Data Valuation".



### Datasets

All datasets are placed in `/dataset`. Except for the `a9a` dataset that can be download from [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a9a), other four datasets `MNIST`, `USPS`, `CIFAR-10`, `STL-10` are downloaded using `torchvision` library.



### Running

`/main/main_evalEffective.py`:  to get the accuracy lines and the corresponding effectiveness scores.

`/main/main_evalRobust.py`: to get shapley values.

`/main/main_margiContrib.py`: to evaluate the marginal contribution of pre-shared data and public dataset.



All outputs are stored as `.xlsx` format in  `/outputs` 
