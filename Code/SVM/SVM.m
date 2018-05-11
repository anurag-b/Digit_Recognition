clc;
clear all;
addpath('/home/dark_knight/libsvm-3.22/matlab');
%load images: row is feature, column is observation
train_imgs = loadMNISTImages('train-images.idx3-ubyte');
train_label = loadMNISTLabels('train-labels.idx1-ubyte');
test_imgs = loadMNISTImages('t10k-images.idx3-ubyte');
test_label = loadMNISTLabels('t10k-labels.idx1-ubyte');

%Do dimensionality reduction here
prompt = 'Select the reduction Algorithm\n 1. PCA\n 2. LDA\n';
dimensionality_reduction = input(prompt);
disp('Dimensionality reduction started');
if dimensionality_reduction == 1
    prompt = 'Enter the percentage of reduced features\n';
    red_per = input(prompt);
    [train_new test_new] = PCA(train_imgs,test_imgs,red_per);
elseif dimensionality_reduction == 2
    prompt = 'Enter the number of reduced features <= C-1\n';
    red_per = input(prompt);
    [train_new test_new] = LDA(train_imgs,train_label,test_imgs,test_label,red_per);
else
    disp('invalid reduction selection');
    return;
end
disp('Dimensionality reduction ended');

% ============================================= %
%               SVM starts here                 %
% ============================================= %
%train & test
kernel = 3; %libsvm, 1:linear, 2:polynomial, 3:rbf, default setting
labels = unique(train_label);
num_label = length(labels);
model = cell(num_label,1);
disp('SVM Training Started');
if kernel == 1
    model = svmtrain(train_label, train_new', '-s 0 -t 0 -b 1 -q -g 0.01 -c 2');
elseif kernel == 2
    model = svmtrain(train_label, train_new', '-s 0 -t 1 -b 1 -q -g 0.01 -c 2');
elseif kernel == 3
    model = svmtrain(train_label, train_new', '-s 0 -t 2 -b 1 -q -g 0.01 -c 2');
else
    disp('invalid kernel selection');
    return;
end
disp('SVM Training Ended');
disp('SVM Testing Started');
[y_predict,accuracy,prob_estimates]=svmpredict(test_label, test_new', model,'-q');
disp('SVM Testing Ended');
fprintf('Accuracy : %0.2f %%.\n' ,accuracy(1));