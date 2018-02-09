function bayesian_regression(X,Y,small_sigma_squared,eta_sqaured)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name - bayesian_regression
% Creation Date - 3rd Nov 2014
% Author: Soumya Banerjee
% Website: https://sites.google.com/site/neelsoumya/
%
% Description:
%   Function to do Bayesian regression
%   inspired by video on bayesian linear regression
%   https://www.youtube.com/watch?v=qz2U8coNwV4
%   by mathematicalmonk on youtube
%
% Input:
%       X - matrix of predictors
%       Y - vector of responses
%       small_sigma_squared - standard deviation^2 (variance) for covariance matrix for Y
%       eta_sqaured - standard deviation^2 (variance) for covariance matrix for beta (regressors)
%
% Output:
%       1) Vector of inferred regressors/parameters
%       2) Histograms of inferred regressors/parameters
%       3) Monte Carlo trace plots
%
% Assumptions -
%
% Example usage:
%   X = randn(100,5)
%   r = [0;2;0;-3;0] % only two nonzero coefficients
%   Y = X*r + randn(100,1)*.1 % small added noise
%   small_sigma_squared = 0.01
%   eta_sqaured = 0.01
%   bayesian_regression(X,Y,small_sigma_squared,eta_sqaured)
%
% License - BSD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


tic;
iNumMeasurements = size(X,1);%行
iNumRegressors   = size(X,2);%列

big_sigma = small_sigma_squared * eye(iNumMeasurements); %a = 0.01 * eye(5);
big_omega = eta_sqaured * eye(iNumRegressors); %b = 0.01 * eye(5);beta的协方差矩阵的标准差的平方（方差）

disp('covariance matrix and mean vector of posterior distribution')
lambda = inv(X' * inv(big_sigma) * X + inv(big_omega))%big_omega是先验的精度矩阵，big_sigma是似然函数的精度矩阵
mu     = lambda * X' * inv(big_sigma) * Y

% mvnrnd(MU,SIGMA)
% mvnrnd(mu,lambda)
iNumIter = 10000; % number of samples
for iCount=1:iNumIter
    w_vector_array(iCount,:) = [mvnrnd(mu,lambda)];%mvnrnd生成多维正态分布
end
size(w_vector_array)
disp('inferred parameter vector (mean)')
if iNumRegressors == 1
    
    [mean(w_vector_array(1:end,1))]
    y=mu*X;
    iNumBins = 100;
    subplot(1,1,1)
    hist(w_vector_array(1:end,1),iNumBins)
    figure
    plot(X,Y,'.')
    hold on
    plot(X,y)
end
if iNumRegressors == 5
% [mean(w_vector_array(1:end,1)) mean(w_vector_array(1:end,2)) ...
%     mean(w_vector_array(1:end,3))  mean(w_vector_array(1:end,4)) ...
%     mean(w_vector_array(1:end,5)) ]              %每一列求均值

    iNumBins = 100;
    figID = figure;
    subplot(2,3,1)
    hist(w_vector_array(1:end,1),iNumBins)
    hold on
    subplot(2,3,2)
    hist(w_vector_array(1:end,2),iNumBins)
    hold on
    subplot(2,3,3)
    hist(w_vector_array(1:end,3),iNumBins)
    hold on
    subplot(2,3,4)
    hist(w_vector_array(1:end,4),iNumBins)
    hold on
    subplot(2,3,5)
    hist(w_vector_array(1:end,5),iNumBins)

    hold off
    print(figID, '-djpeg', sprintf('bayesregression_parameters_hist%s.jpg', date));
    figID_2 = figure
    plot(w_vector_array(1:end,1))
    xlabel('Monte Carlo samples'); ylabel('Posterior of one regressor parameter')
    print(figID_2, '-djpeg', sprintf('mcmctrace_%s.jpg', date));
    end

toc;