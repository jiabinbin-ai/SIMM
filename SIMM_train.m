function Model = SIMM_train( X_train, y_train, NumK )
%SIMM_train implements the training phase of the SIMM approach as described in [1]
%Type 'help SIMM_train' under Matlab prompt for more detailed information about SIMM_train
%
%	Syntax
%
%       model = SIMM_train( X_train, y_train, NumK )
%
%	Description
%
%   SIMM_train takes,
%       X_train     - An mxd array, the ith instance of training instance is stored in X_train(i,:)
%       y_train     - An mxq array, the ith class vector of training instance is stored in y_train(i,:)
%       NumK        - Number of most similar examples considered (default 10)
%   and returns,
%       Model       - A struct where 
%                       Model.model_t stores the parameters of the models that obtain the transformed version of each class label
%						Model.model_f stores the parameters of the models that synergize feature-based similarity
%						Model.model_l stores the parameters of the models that synergize label-based similarity
%						Model.model_c stores the parameters of the models that combine the feature-based and label-based similarity
%                       Model.C_y_pair stores the pairwise powerset transformation parameters
%
%  [1] B.-B. Jia, T. Huang, M.-L. Zhang. A Similarity-based Approach for Multi-Dimensional Classification, 2026.
%
%See also SIMM_test and sim_info_gen.

    if nargin<3
        NumK = 10;
    end

    %obtain parameters of data sets
    num_training = size(X_train,1);%number of training examples
    num_dim = size(y_train,2);%number of dimensions(class variables)
%     C_per_dim = cell(num_dim,1);%class labels in each dimension
%     num_per_dim = zeros(num_dim,1);%number of class labels in each dimension
%     for dd=1:num_dim
%         temp = y_train(:,dd);
%         C_per_dim{dd} = unique(temp);
%         num_per_dim(dd) = length(C_per_dim{dd});
%     end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%TRAINING PHASE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %(1)feature similarity
	%SIMM only requires identifying the k most similar examples from training set, 
	%this means that the results are invariant to the bandwidth parameter in Eq.(1) and Eq.(11). 
	%Therefore, we simply use the built-in function knnsearch instead. 
    index_k_plus_1 = knnsearch(X_train,X_train,'K',NumK+1);
    index_f_tr = index_k_plus_1(:,2:end);%exclude the instance itself in training set
    sim_info_vec_f_tr = cell(num_dim,1);%similarity information vector in feature space for training samples
    for dd=1:num_dim
        sim_info_vec_f_tr{dd} = sim_info_gen(index_f_tr, y_train(:,dd));
    end
    %(2)label similarity
    model_t = cell(num_dim,1);
    sim_info_vec_l_tr = cell(num_dim,1);%similarity information vector in label space for training samples
    for dd=1:num_dim
        model_t{dd} = train(y_train(:,dd),sparse(X_train),'-s 6 -B 1 -q');
        [~, ~, p_tr] = predict(ones(num_training,1), sparse(X_train), model_t{dd}, '-q');
        index_k_plus_1 = knnsearch(p_tr,p_tr,'K',NumK+1);
		index_l_tr_dd = index_k_plus_1(:,2:end);%exclude the instance itself in training set
        sim_info_vec_l_tr{dd} = sim_info_gen(index_l_tr_dd, y_train(:,dd));
    end
    %(3)prediction model induction
    %(3-1)synergize the feature-based similarity information
    C_y_pair = cell(num_dim,num_dim);
    y_pair_cp = cell(num_dim,num_dim);
    model_f = cell(num_dim,num_dim);
    for dd1=1:num_dim-1
        sim_info_vec_f_tr_dd1 = sim_info_vec_f_tr{dd1};
        for dd2=dd1+1:num_dim
            sim_info_vec_f_tr_dd2 = sim_info_vec_f_tr{dd2};
            [C_y_pair{dd1,dd2},~,y_pair_cp{dd1,dd2}] = unique(y_train(:,[dd1,dd2]),'rows');
            sim_info_vec_f_tr_pair = [sim_info_vec_f_tr_dd1, sim_info_vec_f_tr_dd2];
            model_f{dd1,dd2} = train(y_pair_cp{dd1,dd2},sparse(sim_info_vec_f_tr_pair),'-s 6 -B 1 -q');
        end
    end
    %(3-2)synergize the label-based similarity information
    model_l = cell(num_dim,num_dim);
    for dd1=1:num_dim-1
        sim_info_vec_l_tr_dd1 = sim_info_vec_l_tr{dd1};
        for dd2=dd1+1:num_dim
            sim_info_vec_l_tr_dd2 = sim_info_vec_l_tr{dd2};
            sim_info_vec_l_tr_pair = [sim_info_vec_l_tr_dd1, sim_info_vec_l_tr_dd2];
            model_l{dd1,dd2} = train(y_pair_cp{dd1,dd2},sparse(sim_info_vec_l_tr_pair),'-s 6 -B 1 -q');
        end
    end
    %(3-3)combine the feature-based and label-based similarity information
    model_c = cell(num_dim,num_dim);
    y_predict_tr_pair_cell = cell(num_dim,num_dim);
    for dd1=1:num_dim-1
        sim_info_vec_f_tr_dd1 = sim_info_vec_f_tr{dd1};
        sim_info_vec_l_tr_dd1 = sim_info_vec_l_tr{dd1};
        for dd2=dd1+1:num_dim
            sim_info_vec_f_tr_dd2 = sim_info_vec_f_tr{dd2};
            sim_info_vec_l_tr_dd2 = sim_info_vec_l_tr{dd2};
            sim_info_vec_f_tr_pair = [sim_info_vec_f_tr_dd1, sim_info_vec_f_tr_dd2];
            sim_info_vec_l_tr_pair = [sim_info_vec_l_tr_dd1, sim_info_vec_l_tr_dd2];
            [~, ~, outputs_f_tr_pair] = predict(ones(num_training,1), sparse(sim_info_vec_f_tr_pair), model_f{dd1,dd2}, '-q');
            [~, ~, outputs_l_tr_pair] = predict(ones(num_training,1), sparse(sim_info_vec_l_tr_pair), model_l{dd1,dd2}, '-q');
            sim_info_vec_c_tr_pair = [outputs_f_tr_pair,outputs_l_tr_pair];
            model_c{dd1,dd2} = train(y_pair_cp{dd1,dd2},sparse(sim_info_vec_c_tr_pair),'-s 6 -B 1 -q');
            y_predict_tr_pair_cell{dd1,dd2} = predict(ones(num_training,1),sparse(sim_info_vec_c_tr_pair), model_c{dd1,dd2},'-q');%-b 1 
        end
    end
    
    %return necessary model parameters for test phase
    Model.model_t = model_t;%model parameters for transforming categorical label space into numeric one
    Model.model_f = model_f;%model parameters for synergizing feature-based similarity
    Model.model_l = model_l;%model parameters for synergizing label-based similarity
    Model.model_c = model_c;%model parameters for combining the feature-based and label-based similarity
    Model.C_y_pair = C_y_pair;%just to make it easier to use for SIMM_test
    Model.NumK = NumK;%just to make it easier to use for SIMM_test
    Model.y_predict_tr_pair_cell = y_predict_tr_pair_cell;%pairwise powerset prediction for training samples
end