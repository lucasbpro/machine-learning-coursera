function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])

% constants
K = size(centroids, 1);
m = size(X, 1);

% initilize idx -a vector which will contain the index of the centroid
% closest to X(i) 
idx = zeros(m, 1);

% find the closest centroid for each X(i)
for i=1:m
    distToCentroids = repmat(X(i,:),K,1) - centroids;
    [~, idx(i)] = min(sqrt(sum(distToCentroids.^2,2)));
end;

end

