# SIMM
This repository provides the MATLAB implementation for the following article:

B.-B. Jia, T. Huang, M.-L. Zhang. A Similarity-based Approach for Multi-Dimensional Classification, 2026.

"SIMM_train.m" and "SIMM_test.m" are the train and test functions of this algorithm and "ReadMe.m" is an examplar file on how the SIMM_train and SIMM_test program could be used.

## Dependencies
This implementation calls the pre-compiled MEX binaries (`predict.mexw64` and `train.mexw64`) from the **LIBLINEAR** library to perform L1-regularized logistic regression. The LIBLINEAR library is available at https://www.csie.ntu.edu.tw/~cjlin/liblinear/. For users on non-Windows platforms, please visit the official LIBLINEAR website to obtain the source code and compile the binaries for your specific operating system.
