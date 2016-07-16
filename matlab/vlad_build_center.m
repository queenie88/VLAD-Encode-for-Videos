% Author: Xiukun Huang
% Date: April, 2016

clear;
run('/home/huangxiukun/CVPR_vine/vlfeat-0.9.20/toolbox/vl_setup');
en=146;% number of folders
d=256;% dimension after pca
k=256;
kmrepeats=5;
framestride=4;
fn=cell(en,1);

for i=1:en
    fn{i}=sprintf('/mnt/disk1/huangxiukun/CVPR_vlad_encode/test_folder/cnnFeatures/cnnFeatures_tagNumIs146_%04d.h5',i);
end
datapath='data/';
note= sprintf('framestride: %d, no-norm on lcd before pca; d%d after pca, k-%d-means++ init, vlad %d repeats; gmm %d repeats',framestride,d,k,kmrepeats,kmrepeats);
data=[];
start=[1,1,1];
count=[Inf,Inf,Inf];
stride=[1,1,framestride];
fc=0;

for i=1:en
    tic
    h5disp(fn{i});
    datas = h5read(fn{i},'/pool5lcd',start,count,stride);
    datas = permute(datas,[2,1,3]);
    dim_ds = size(datas);
    datas_mat = reshape(datas,[dim_ds(1), prod(dim_ds(2:3)) ] );
    datas_mat = datas_mat';
    dim_mat = size(datas_mat)
    fc = fc + dim_mat(1);
    data=[data;datas_mat];
    tinner=toc
end


save([datapath,'samplestat.mat'],'start','count','stride','note');


%pca
tic
[coeff,score,latent,~,~,mu] = pca(data);
save([datapath,'pca.mat'],'coeff','latent','mu');
t=toc


data_after_pca=score(:,1:d);
%whitening
data_after_pca=bsxfun(@rdivide,data_after_pca,sqrt(latent(1:d))');

data_after_pca=data_after_pca';

%vlad
tic
centers = vl_kmeans(data_after_pca, k,'initialization','plusplus','NumRepetitions',kmrepeats);
kdtree = vl_kdtreebuild(centers) ;
save([datapath,'vladcenters.mat'],'centers','kdtree');
t=toc
