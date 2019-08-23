%% GIST
image_path = '.\caltech101';
imgfeature_path = '.\caltech101feature';
Images= dir([image_path '\*.jpg']);  %¶ÁÈ¡Í¼Æ¬¿â
%creat_GIST(image_path, imgfeature_path);
load ([imgfeature_path '\GIST.mat'], 'GIST');

%% PCA+ITQ
[coeff,score,latent] = pca(GIST);
lat = find(latent>0.001);
new_feature = score(:, 1:max(lat));
ITQparam.pcaW = new_feature;
[~, ITQparam.nbits] = size(new_feature);
ITQparam = trainITQ(ITQparam);

%% Binary Code
V = ITQparam.pcaW;
% rotate the data
U = V * ITQparam.r;
B = compactbit(U);

%% Test
rimg = randi([1 1000]);
img = imread([[image_path '\'] Images(rimg).name]); 
figure(1);
imshow(img);
BC = B(rimg,:);
H = [sum(xor(B, BC),2) reshape(1:1000,1000,1)];
newH = sortrows(H);
figure(2);
for i=2:11
    subplot(2,5,i-1);
    img = imread([[image_path '\'] Images(newH(i,2)).name]); 
    imshow(img,'border','tight','InitialMagnification','fit');
end