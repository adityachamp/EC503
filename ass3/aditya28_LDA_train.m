%load('data_iris');
function [LDAmodel] = aditya28_LDA_train(X_train, Y_train, numofClass);
% Training LDA
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
% LDAmodel : the parameters of LDA classifier which has the following fields
% LDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% LDAmodel.Sigmapooled : D * D  covariance matrix
% LDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i
%

% Write your codes here:

%declaring varialbles for the LDA equation
[samples,D]=size(X_train);
Mu=zeros(numofClass,D);
Sigmapooled=zeros(D,D);
Pi=zeros(numofClass,1);

%Calculating parameters for LDA
for i=1:numofClass
   index=find(Y_train==i);
   Mu(i,:)=mean(X_train(index,:));
   Pi(i)=length(index)/samples; 
end

%calculating common covariance matrix for LDA
Sigmapooled(:,:) = cov(X_train);

%creating a structure and storing values
LDAmodel = struct('Mu',Mu,'Sigmapooled',Sigmapooled,'Pi',Pi);

end
