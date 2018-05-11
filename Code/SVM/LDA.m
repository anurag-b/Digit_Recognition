function [x_train_new,x_test_new] = LDA(train_imgs,train_label,test_imgs,test_label,red_per)

[feature pose_train] = size(train_imgs);
[feature pose_test] = size(test_imgs);

labels = unique(train_label);
num_label = length(labels);

% ============================================= %
%   LDA starts here(Dimensionality Reduction)   %
% ============================================= %

%Calculate the mean of all the data
mean_all_data = mean(train_imgs,2);
%Classwise mean
for i = 1:num_label
    z= (train_label==labels(i));
    mean_classwise(:,i) = mean(train_imgs(:,z),2);
end

%Calulate the within class scatter
% s_i = (x_i - mu) * (x_i - mu)'
% Compute Total within class scatter Sw i.e. Sw = S1 + S2 + S3 + ... + Sc
s_w = zeros(feature,feature);
x_train_hat = train_imgs;
%Get x_hat i.e. x - mean_classwise
for i = 1:num_label
    z= (train_label==labels(i));
    x_train_hat(:,z) = train_imgs(:,z) - mean_classwise(:,i);
    x = x_train_hat(:,z);
    s_i = zeros(feature,feature);
    for j = 1:size(x,2)
        temp = x(:,j) * x(:,j)';
        s_i = s_i + temp;
    end
    s_w = s_w + s_i;
end

delta = 0.1;  %Sw singularity
s_w = s_w + delta*eye(feature);

%Calculate the between class scatter
%%s_b = summation(n * (mu_k - mu) * (mu_k - mu)')
s_b = zeros(feature, feature);
for i = 1:num_label
    s_b(:,:) =  s_b(:,:) + pose_train * (mean_classwise(:,i) - mean_all_data(:,:)) * (mean_classwise(:,i) - mean_all_data(:,:))';
end

%Get the w matrix i.e. w = Sb/Sw
w = inv(s_w)*s_b;
%K = ceil(red_per/100 * feature);
[evec, eval] = eigs(w,red_per);
w_lda = evec;

%new training data
x_train_new(:,:) = w_lda' * train_imgs(:,:);

%new test data
x_test_new(:,:) = w_lda' * test_imgs(:,:);

end

