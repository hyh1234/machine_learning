function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes k
% inputSize - the size N of the input vector 输入向量x的长度
% lambda - weight decay parameter
% data - the N x M(len_data) input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);%10行inputSize列
numCases = size(data, 2);%把矩阵X的列数赋给numCases,size(data,1)矩阵X的行数

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;
thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.
weight_dacay=0.0;
for i=1:1:numClasses
    for j=1:1:inputSize
        weight_dacay= weight_dacay+theta(i,j)*theta(i,j);
    end
end
M = bsxfun(@minus,theta*data,max(theta*data, [], 1));
M=exp(M);
p= bsxfun(@rdivide, M, sum(M));
cost=-1.0/numCases*(groundTruth(:)'*log(p(:)))+lambda/2.0*weight_dacay;%cost is a variable, not a matrix
thetagrad= -1/numCases*(groundTruth-p)*data'+lambda*theta;

  
% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

