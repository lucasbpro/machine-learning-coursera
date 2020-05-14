function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% You need to return the following variables correctly.
centroids = zeros(K, size(X,2));

% computes the mean of all points that belong to the same cluster
% (associated to the same centroid) 
for k=1:K
    [indicesCentroid,~] = find(idx == k);
    centroids(k,:) = mean(X(indicesCentroid,:));
end;

end

