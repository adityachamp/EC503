function [QDAmodel]= aditya28_QDA_train(X_train, Y_train, numofClass);
% Training QDA
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
% QDAmodel : the parameters of QDA classifier which has the following fields
% QDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% QDAmodel.Sigma : D * D * numofClass array, Sigma(:,:,i) = covariance matrix of class i
% QDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i

% Write your code here:

%declaring all the parameters
[samples,D]=size(X_train);
Mu=zeros(numofClass,D);
Sigma=zeros(D,D,numofClass);
Pi=zeros(numofClass,1);

%calculating al the parameters
for i=1:numofClass
   index_array=find(Y_train==i);
   Mu(i,:)=mean(X_train(index_array,:));
   Pi(i)=length(index_array)/samples;
   Sigma(:,:,i) = cov(X_train(index_array,:)); 
end

%creating structure that stires all values
QDAmodel = struct('Mu',Mu,'Sigma',Sigma,'Pi',Pi);

end


