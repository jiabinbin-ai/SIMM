%This is an examplar file on how the SIMM algorithm could be used
%Type 'help SIMM_train' and 'help SIMM_test' under Matlab prompt for more detailed information
clear;clc;close all;fclose('all');
%more data sets are publicly available at: 
%https://palm.seu.edu.cn/zhangml/Resources.htm#MDC_data
load_data = load('Enb.mat');%load the MDC data
X_load = load_data.data.norm;
y_load = load_data.target;
data_name = load_data.data_name;

%create log file
all_fid = 1;%for standard output, the screen
% title_str = ['log_SIMM_',data_name];
% temp_str = [title_str,'.txt'];
% all_fid = fopen(temp_str,'w');
    
%hyper-parameters
NumK = 10;
    
%main
start_clock = clock;
numFolds = 10;
HS = zeros(numFolds,1);
EM = zeros(numFolds,1);
SEM = zeros(numFolds,1);
for numFold=1:numFolds
    temp_str = ['Fold-(', num2str(numFold),'/',num2str(numFolds),') begins...\n'];
    fprintf(all_fid,temp_str);
    %split dataset into training set and testing set
    X_train = X_load(load_data.idx_folds{numFold}.train,:);
    y_train = y_load(load_data.idx_folds{numFold}.train,:);
    X_test = X_load(load_data.idx_folds{numFold}.test,:);
    y_test = y_load(load_data.idx_folds{numFold}.test,:);
    %train & test
    SIMM_Model = SIMM_train(X_train, y_train, NumK);
    Eval = SIMM_test(SIMM_Model, X_train, y_train, X_test, y_test);
    %store the performance metric values
    HS(numFold) = Eval.HS;
    EM(numFold) = Eval.EM;
    SEM(numFold) = Eval.SEM;
end
%disp experimental results
temp_str = ['HS   : ', num2str(mean(HS),'%4.3f'),' ¡À ',num2str(std(HS),'%4.3f'),'\n'];
fprintf(all_fid,temp_str);

temp_str = ['EM   : ', num2str(mean(EM),'%4.3f'),' ¡À ',num2str(std(EM),'%4.3f'),'\n'];
fprintf(all_fid,temp_str);

temp_str = ['SEM  : ', num2str(mean(SEM),'%4.3f'),' ¡À ',num2str(std(SEM),'%4.3f'),'\n'];
fprintf(all_fid,temp_str);

finish_clock = clock;
time_cost = etime(finish_clock,start_clock);
temp_str = ['The time cost is about ', num2str(time_cost,'%4.3f'),' seconds.\n'];
fprintf(all_fid,temp_str);