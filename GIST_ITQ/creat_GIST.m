%  function creat_GIST（image_path, imgfeature_path)
%  函数功能;提取图像的GIS特征，并保存
%  image_path：读取的图片库路径
%  imgfeature_path：特征保存路径
%

function creat_GIST(image_path, imgfeature_path)
Images= dir([image_path '\*.jpg']);  %读取图片库
Nimage=length(Images);  %获取图片库中图片数量

% GIST Parameters:
clear param
param.imageSize = [256 256];
param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale (from HF to LF)
param.numberBlocks = 4;
param.fc_prefilt = 4;

Nfeatures = sum(param.orientationsPerScale) * param.numberBlocks ^ 2; 
GIST=zeros([Nimage Nfeatures]);
setwaitbar=waitbar(0,'准备开始','name','加载中...');

%加载第一个图像并计算要点：
img = imread([[image_path '\'] Images(1).name]); 
[GIST(1,:),param] = LMgist(img,'',param);   %first call 

for i1=2:Nimage     
    image=imread([[image_path '\'] Images(i1).name]);
    GIST(i1,:) = LMgist(image, '', param);
    
    jinduzhi=fix(i1/Nimage*100);
    waitbar(jinduzhi/100,setwaitbar,['已完成' num2str(jinduzhi) '%']);
    
end

save([imgfeature_path '\GIST'], 'GIST');
delete(setwaitbar);
clear setwaitbar;






