function [ sim_info_vec, KNN_test_idx] = sim_info_gen( dist_idx, y_train )
%sim_info_gen generates the similarity-based information for multi-class data set
%
%	Syntax
%
%       [ sim_info_vec, KNN_test_idx] = sim_info_gen( dist_idx, y_train )
%
%	Description
%
%   sim_info_gen takes,
%       dist_idx	- An pxK array, the KNN index for the ith test instance is stored in dist_idx(i,:)
%       y_train     - An mxq array, the ith class vector of training instance is stored in y_train(i,:)
%   and returns,
%       sim_info_vec	- KNN statistics of train set
%       KNN_test_idx	- KNN index of test set


    %% default parameters setting
    if nargin<2
        error('Not enough input parameters!');
    end

    %% get parameters of data sets
    C = unique(y_train);
    num_class = length(C);%number of class labels
    num_testing = size(dist_idx,1);%number of testing examples
    %% Obtain KNN statistics for testing instance
    sim_info_vec = zeros(num_testing,num_class);
    for ii=1:num_testing
        index_ii = dist_idx(ii,:);
        y_train_ii = y_train(index_ii);
        for iclass=1:num_class
            sim_info_vec(ii,iclass) = sum(y_train_ii==C(iclass));
        end
    end
    KNN_test_idx = dist_idx;
end