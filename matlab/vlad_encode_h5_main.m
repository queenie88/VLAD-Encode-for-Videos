% Author: Xiukun Huang
% Date: April, 2016

clear;
run('/home/huangxiukun/CVPR_vine/vlfeat-0.9.20/toolbox/vl_setup');
d=256;
k=256;
en=146;

dicpath='data/';
froot='/mnt/disk1/huangxiukun/CVPR_vlad_encode/test_folder/cnnFeatures/';
outfrootv='/mnt/disk1/huangxiukun/CVPR_vlad_encode/test_folder/encoded_cnnFeatures/';

fn     = cell(en,1);
outfnv = cell(en,1);


for i=1:en
    f=sprintf('cnnFeatures_tagNumIs146_%04d.h5',i);
    fn{i}     = [froot,f];
    outfnv{i} = [outfrootv,f];
end


load([dicpath,'pca.mat']);
load([dicpath,'vladcenters.mat']);


%encoding data
for i=1:en
    infile=fn{i};
    
    outfile_video=outfnv{i};
    vlad_encode_h5_video(d,k,coeff,mu,latent,kdtree,centers,infile,outfile_video);
end
