function [x_train_new,x_test_new] = PCA(train_imgs,test_imgs,red_per)

[feature pose_train] = size(train_imgs);
[feature pose_test class] = size(test_imgs);
label = 10;

% ============================================= %
%   PCA starts here(Dimensionality Reduction)   %
% ============================================= %
%Calculate the mean of all the data
mean_pca = mean(train_imgs,2);

%Get x_hat i.e. x- mean
x_train_hat(:,:) = train_imgs(:,:) - mean_pca;

%Calculate the Covariance of the data
sigma_pca(:,:) = cov(x_train_hat');

%Get the reduced feature
%Find the eigenvalues of the covariance matrix
[U V W] = svd(sigma_pca);
red = ceil(red_per/100 * feature);
%Get the k largest eigen values to get number of reduced features
U_new  = U(:,1:red);

%new training data
x_train_new(:,:) = U_new' * x_train_hat(:,:);

%Normalise the test data using data mean
x_test_hat(:,:) = test_imgs(:,:) - mean_pca;

%new test data
x_test_new(:,:) = U_new' * x_test_hat(:,:);

%end