clear all;
clc;
K = load('data_knnSimulation');
dist = zeros(200,2);
k1_classmatrix = zeros(96,96);
k5_classmatrix = zeros(96,96);
ind_i = 1;
ind_j = 1;
for i = -3.5:0.1:6 
    ind_j = 1;
   for j = -3:0.1:6.5
        for k = 1:200
            dist(k,1) = sqrt((i - K.Xtrain(k,1))^2 + (j - K.Xtrain(k,2))^2);
            dist(k,2) = K.ytrain(k);
        end
        %disp(dist);
        sorted_dist = sortrows(dist);
        %disp(sorted_dist);
       
        %class label prediction for k=1
        k1 = zeros(1,2);
        k5 = zeros(5,2);
        k1 = sorted_dist(1,:);
        k5 = sorted_dist(1:5,:);
       
        %class label prediction for k=5
        k5_mode = mode(k5);
        k5_classmatrix(ind_i, ind_j) = k5_mode(1,2);
        k1_classmatrix(ind_i, ind_j) = k1(1,2);
        ind_j = ind_j +1;
   end             
   ind_i = ind_i +1;
end

%plotting hte graphs
xaxis = [-3.5:0.1:6];
yaxis = [-3:0.1:6.5];

%plotting class label prediction for k=5
figure;
imagesc(xaxis,yaxis,k5_classmatrix)
title('class label prediction for k=5')
colorbar
 
%plotting class label prediction for k=1
figure;
imagesc(xaxis,yaxis,k1_classmatrix)
title('class label prediction for k=1')
colorbar

