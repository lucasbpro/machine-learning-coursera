function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim.
%
%   Mathematically, sim is the similarity between x1 and x2 computed using 
%   a Gaussian kernel with bandwidth sigma.

% Ensure that xi and xj are column vectors
xi = x1(:); xj = x2(:);

% Returns similarity
sim = exp((-norm(xi-xj)^2)/(2*sigma^2));

end
