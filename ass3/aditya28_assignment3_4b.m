clear all;
clc;
K = load('data_knnSimulation');

%declaring the distance variable
dist = zeros(200,2);
prob_class2 = zeros(96,96);
ind_i = 1;
ind_j = 1;

%calculating of points in the grid from the train points
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
        k10 = zeros(10,2);
            for m= 1:10
                k10(m,:)  = sorted_dist(m,:);
            end
%finding number of index belonging to class2
numof2_index = find(k10(:,2)==2);
numof2 = numel(numof2_index);

%finding number of index belonging to class3
numof3_index = find(k10(:,2)==3);
numof3 = numel(numof3_index);

%finding the probability in the respective classes
prob_class2(ind_i,ind_j) = numof2/10;
prob_class3(ind_i,ind_j) = numof3/10;
    ind_j = ind_j+1;                
   end             
    ind_i = ind_i+1;
end

%obtaining the plot
xaxis = [-3.5:0.1:6];
yaxis = [-3:0.1:6.5];

%probability of being class2
figure;
imagesc(xaxis,yaxis,prob_class2)
title('probability of being class2');
colorbar

%probability of being class3
figure;
imagesc(xaxis,yaxis,prob_class3)
title('probability of being class3');
colorbar


