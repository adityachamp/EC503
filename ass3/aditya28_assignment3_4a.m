clear all;
clc;
K = load('data_knnSimulation')
% W = [Xtrain ytrain]

%obtaining the plot
gscatter(K.Xtrain(:,1),K.Xtrain(:,2),K.ytrain,'rgb')
xlabel('feature1')
ylabel('feature2')
title('knn feature1 vs feature2')