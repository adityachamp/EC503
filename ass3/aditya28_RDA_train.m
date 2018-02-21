function [RDAmodel]= aditya28_RDA_train(X_train, Y_train,gamma, numofClass)
%
% Training RDA
%
% EC 503 Learning from Data
% Gaussian Discriminant Analysis
%
% Assuming D = dimension of data
% Inputs :
% X_train : training data matrix, each row is a training data point
% Y_train : training labels for rows of X_train
% numofClass : number of classes 
%
% Assuming that the classes are labeled  from 1 to numofClass
% Output :
% RDAmodel : the parameters of RDA classifier which has the following fields
% RDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% RDAmodel.Sigmapooled : D * D  covariance matrix
% RDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i

% Write your code here:

[samples,D]=size(X_train);
Mu=zeros(numofClass,D);
Sigma_LDA=zeros(D,D);
Sigmapooled=zeros(D,D);
Pi=zeros(numofClass,1);

for i=1:numofClass 
    index=find(Y_train==(i-1));
    Mu(i,:)=mean(X_train(index,:));
    Pi(i)=length(index)/samples; 
end
Sigma_LDA(:,:) = cov(X_train);

Sigmapooled= (gamma* diag(diag(Sigma_LDA))) + ((1-gamma)*Sigma_LDA);


RDAmodel = struct('Mu',Mu,'Sigmapooled',Sigmapooled,'Pi',Pi);

end
















