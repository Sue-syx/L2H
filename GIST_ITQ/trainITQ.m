function ITQparam = trainITQ(ITQparam)

% Input:
%          X: n*d, n is the number of images
%          ITQparam:
%                           ITQparam.pcaW---PCA of all the database
%                           ITQparam.nbits---encoding length
% Output:
%             ITQparam:
%                              ITQparam.pcaW---PCA of all the database
%                              ITQparam.nbits---encoding length
%                              ITQparam.r---ITQ rotation projection

V = ITQparam(1).pcaW;
nbits = ITQparam(1).nbits;

% initialize with a orthogonal random rotation
R = randn(nbits, nbits);
[U11, ~, ~] = svd(R);
R = U11(:, 1: nbits);

% ITQ to find optimal rotation
for iter=0:50
    Z = V * R; 
    UX = ones(size(Z,1),size(Z,2)).*-1;  
    UX(Z>=0) = 1; 
    C = UX' * V;
    [UB, ~, UA] = svd(C);    
    R = UA * UB';
    %fprintf('iteration %d has finished\r',iter);
end

ITQparam(1).r = R;