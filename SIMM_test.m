function [ Eval,y_predict ] = SIMM_test( Model, X_train, y_train, X_test, y_test )
%SIMM_test implements the test phase of the SIMM approach as described in [1]
%Type 'help SIMM_test' under Matlab prompt for more detailed information about SIMM_test
%
%	Syntax
%
%       [ Eval,y_predict ] = SIMM_test( Model, X_train, y_train, X_test, y_test )
%
%	Description
%
%   SIMM_test takes,
%       Model       - A struct that stores the outputs of SIMM_train (some necessary model parameters)
%       X_train     - An mxd array, the ith instance of training instance is stored in X_train(i,:)
%       y_train     - An mxq array, the ith class vector of training instance is stored in y_train(i,:)
%       X_test      - An pxd array, the ith instance of testing instance is stored in X_test(i,:)
%       y_test      - An pxq array, the ith class vector of testing instance is stored in y_test(i,:)
%   and returns,
%       Eval	    - A struct where 
%						Eval.HS correpsonds to the hamming score on testing data as described in [1]
%						Eval.EM correpsonds to the exact match on testing data as described in [1]
%						Eval.SEM correpsonds to the sub-exact match on testing data as described in [1]
%       y_predict	- An pxq array, the predicted class matrix for test instance matrix X_test
%
%  [1] B.-B. Jia, T. Huang, M.-L. Zhang. A Similarity-based Approach for Multi-Dimensional Classification, 2026.
%
%See also SIMM_train and sim_info_gen.

    %model parameters
    model_t = Model.model_t;%model parameters for transforming categorical label space into numeric one
    model_f = Model.model_f;%model parameters for synergizing feature-based similarity
    model_l = Model.model_l;%model parameters for synergizing label-based similarity
    model_c = Model.model_c;%model parameters for combining the feature-based and label-based similarity
    C_y_pair = Model.C_y_pair;%just for convenience, can be derived from y_train
    NumK = Model.NumK;%just for convenience
    y_predict_tr_pair_cell = Model.y_predict_tr_pair_cell;%pairwise powerset prediction for training samples, just for convenience, can be derived from model_t, model_f, model_l and model_c

    %obtain parameters of data sets
    num_training = size(X_train,1);%number of training examples
    num_dim = size(y_train,2);%number of dimensions(class variables)
    num_testing = size(X_test,1);%number of testing examples
%     C_per_dim = cell(num_dim,1);%class labels in each dimension
%     num_per_dim = zeros(num_dim,1);%number of class labels in each dimension
%     for dd=1:num_dim
%         temp = y_train(:,dd);
%         C_per_dim{dd} = unique(temp);
%         num_per_dim(dd) = length(C_per_dim{dd});
%     end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%TESTING PHASE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %(1)feature & label similarity for test samples
    index_f_te = knnsearch(X_train,X_test,'K',NumK);
    sim_info_vec_f_te = cell(num_dim,1);
    sim_info_vec_l_te = cell(num_dim,1);
    for dd=1:num_dim
        sim_info_vec_f_te{dd} = sim_info_gen(index_f_te, y_train(:,dd));
        [~, ~, p_tr] = predict(ones(num_training,1), sparse(X_train), model_t{dd}, '-q');
        [~, ~, p_te] = predict(ones(num_testing,1), sparse(X_test), model_t{dd}, '-q');
        index_l_te = knnsearch(p_tr,p_te,'K',NumK);
        sim_info_vec_l_te{dd} = sim_info_gen(index_l_te, y_train(:,dd));
    end
    
    %(2)obtain and then rearrange the prediction for test samples
    y_predict_te_single_cell = cell(num_dim,1);
    for dd1=1:num_dim-1
        sim_info_vec_f_te_dd1 = sim_info_vec_f_te{dd1};
        sim_info_vec_l_te_dd1 = sim_info_vec_l_te{dd1};
        for dd2=dd1+1:num_dim
            sim_info_vec_f_te_dd2 = sim_info_vec_f_te{dd2};
            sim_info_vec_l_te_dd2 = sim_info_vec_l_te{dd2};
            sim_info_vec_f_te_pair = [sim_info_vec_f_te_dd1, sim_info_vec_f_te_dd2];
            sim_info_vec_l_te_pair = [sim_info_vec_l_te_dd1, sim_info_vec_l_te_dd2];
            [~, ~, outputs_f_te_pair] = predict(ones(num_testing,1), sparse(sim_info_vec_f_te_pair), model_f{dd1,dd2}, '-q');
            [~, ~, outputs_l_te_pair] = predict(ones(num_testing,1), sparse(sim_info_vec_l_te_pair), model_l{dd1,dd2}, '-q');
            sim_info_vec_c_te_pair = [outputs_f_te_pair,outputs_l_te_pair];
            y_predict_te_pair = predict(ones(num_testing,1),sparse(sim_info_vec_c_te_pair), model_c{dd1,dd2},'-q');%-b 1 
            y_predict_te_single = C_y_pair{dd1,dd2}(y_predict_te_pair,:);
            y_predict_te_single_cell{dd1} = [y_predict_te_single_cell{dd1},y_predict_te_single(:,1)];
            y_predict_te_single_cell{dd2} = [y_predict_te_single_cell{dd2},y_predict_te_single(:,2)];
        end
    end
    
    %(3)KNN accuracy
    KNN_test_idx = index_f_te;%for KNN empical accuracy selection
    acc_global_cell = cell(num_dim,1);%used when KNN accuracy is the same
    acc_knn_cell = cell(num_dim,1);%KNN accuracy for each dim. and each test sample
    for dd1=1:num_dim-1
        for dd2=dd1+1:num_dim
            %solve the global accuracy which will be used when KNN accuracy is the same
            y_predict_tr_pair = y_predict_tr_pair_cell{dd1,dd2};
            y_predict_tr_single = C_y_pair{dd1,dd2}(y_predict_tr_pair,:);
            acc_global_cell{dd1} = [acc_global_cell{dd1},...
                sum(y_predict_tr_single(:,1)==y_train(:,dd1))];%global accuarcy
            acc_global_cell{dd2} = [acc_global_cell{dd2},...
                sum(y_predict_tr_single(:,2)==y_train(:,dd2))];%global accuarcy
            %solve the KNN accuracy
            y_test_KNN_true_1 = zeros(num_testing,NumK);
            y_test_KNN_predict_1 = zeros(num_testing,NumK);
            y_test_KNN_true_2 = zeros(num_testing,NumK);
            y_test_KNN_predict_2 = zeros(num_testing,NumK);
            for itest=1:num_testing
                y_test_KNN_predict_1(itest,:) = y_predict_tr_single(KNN_test_idx(itest,:),1)';
                y_test_KNN_true_1(itest,:) = y_train(KNN_test_idx(itest,:),dd1)';
                y_test_KNN_predict_2(itest,:) = y_predict_tr_single(KNN_test_idx(itest,:),2)';
                y_test_KNN_true_2(itest,:) = y_train(KNN_test_idx(itest,:),dd2)';
            end
            acc_knn_cell{dd1} = [acc_knn_cell{dd1},...
                sum(y_test_KNN_true_1==y_test_KNN_predict_1,2)];%KNN accuarcy 
            acc_knn_cell{dd2} = [acc_knn_cell{dd2},...
                sum(y_test_KNN_true_2==y_test_KNN_predict_2,2)];%KNN accuarcy 
        end
    end
    
    %(4)obtain the final prediction
    y_predict = zeros(size(y_test));
    for dd=1:num_dim
        acc_knn = acc_knn_cell{dd};
        acc_global = acc_global_cell{dd};
        y_predict_te_single_dd = y_predict_te_single_cell{dd};
        for itest=1:num_testing
            [tempmax,tempidx] = max(acc_knn(itest,:));
            index_max = (acc_knn(itest,:)==tempmax);
            num_max = sum(index_max);
            if num_max==1%only one maximum
                index = tempidx;
            else%more than one classifier with maximum KNN accuracy
                index_set = 1:num_dim-1;
                index_candidate = index_set(index_max);
                [~,tempmaxidx] = max(acc_global(index_max));
                index = index_candidate(tempmaxidx);%use the maximum global accuracy instead
            end
            y_predict(itest,dd) = y_predict_te_single_dd(itest,index);
        end
    end
    Eval.HS = sum(sum(y_predict==y_test))/(size(y_test,1)*size(y_test,2));
    Eval.EM = sum(sum((y_predict==y_test),2)==size(y_test,2))/size(y_test,1);
    Eval.SEM = sum(sum((y_predict==y_test),2)>=(size(y_test,2)-1))/size(y_test,1);
end