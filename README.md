# Overfitting_in_DNN
A comparative study of methods(including noise, penalty term and dropout method) to reduce overfitting in Deep Neural Network.

## Data
[MNIST](http://yann.lecun.com/exdb/mnist/) with 60000 train set and and 10000 test set is used. The data is sampled to  get a subset of size 6000 so that we can compare overfitting on different data size.

## Prerequisite
Please install MATLAB deeplearning toolbox and download MNIST data.

## Description
- 2-layer SAE + Softmax DNN is trained
- Net is trained with a greedy algorithm with trains net layer by layer(hinton,2006)

## Reference
[1] Hinton, Geoffrey E., Simon Osindero, and Yee-Whye Teh. "A fast learning algorithm for deep belief nets." Neural computation 18.7 (2006): 1527-1554.  
[2]	LeCun, Y., Cortes, C., & Burges, C. J. (1998). The MNIST database of handwritten digits
