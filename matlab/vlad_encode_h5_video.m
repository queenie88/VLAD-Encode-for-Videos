% Author: Xiukun Huang
% Date: April, 2016

function vlad_encode_h5_video(d,k,coeff,mu,latent,kdtree,centers,infile,outfile_videos)

%read data
data = h5read(infile,'/pool5lcd');
data = permute(data,[2,1,3]);
picname = h5read(infile,'/pic_name');
dims=size(data);

data = reshape(data,[dims(1), prod(dims(2:3)) ] );
data=data';

%pca
data=bsxfun(@minus,data,mu)*coeff(:,1:d);
%whitening
data=bsxfun(@rdivide,data,sqrt(latent(1:d))');
%note,reshape in columnwise order!
data = reshape(data',[d,dims(2),dims(3)]);
dims=size(data);% d*50*num_frame

%encoding videos
vid_p = -1;
vid_count=0;
for i=1:dims(3)
    vid_c=str2double(picname{i}(1:6));
    if vid_c ~= vid_p
        vid_p = vid_c;
        vid_count = vid_count+1;
    end
end
enc_mat_videos = zeros(d*k,vid_count,'single');
vid_p = str2double(picname{1}(1:6));
vidname={picname{1}};
frame_num = 0;
vid_count=1;
for i=1:dims(3)
    vid_c=str2double(picname{i}(1:6));
    if vid_c == vid_p
        frame_num = frame_num + 1;
    else
        data_v = data(:,:,i-frame_num:i-1);
        dim_data_v = size(data_v);
        data_v = reshape(data_v,[dim_data_v(1), prod(dim_data_v(2:3))]);
        
        nn = vl_kdtreequery(kdtree, centers, data_v) ;
        dims_v=size(data_v);
        numDataToBeEncoded=dims_v(2);
        assignments = zeros(k,numDataToBeEncoded,'single');
        assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;
        enc_mat_videos(:,vid_count) = vl_vlad(data_v,centers,assignments,'SquareRoot','NormalizeComponents');

        vidname=[vidname;picname{i}];
        vid_p = vid_c;
        vid_count = vid_count+1;
        frame_num=1;
    end
    
end
data_v = data(:,:,dims(3)-frame_num+1:dims(3));
dim_data_v = size(data_v);
data_v = reshape(data_v,[dim_data_v(1), prod(dim_data_v(2:3))]);

nn = vl_kdtreequery(kdtree, centers, data_v) ;
dims_v=size(data_v);
numDataToBeEncoded=dims_v(2);
assignments = zeros(k,numDataToBeEncoded,'single');
assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;
enc_mat_videos(:,vid_count) = vl_vlad(data_v,centers,assignments,'SquareRoot','NormalizeComponents');
%write to h5
h5create(outfile_videos,'/feature',size(enc_mat_videos),'Datatype','single','ChunkSize',size(enc_mat_videos)/2,'Deflate',9);
h5write(outfile_videos,'/feature',enc_mat_videos);
%write picname

fid = H5F.open(outfile_videos,'H5F_ACC_RDWR','H5P_DEFAULT');
type_id = H5T.copy('H5T_C_S1');
H5T.set_size(type_id,25);
pndims = size(vidname);
pndims = pndims(1);
h5_dims = fliplr(pndims);
h5_maxdims = h5_dims;
space_id = H5S.create_simple(1,h5_dims,h5_maxdims);
dset_id = H5D.create(fid,'vid_name',type_id,space_id,'H5P_DEFAULT');
H5D.write(dset_id,'H5ML_DEFAULT','H5S_ALL','H5S_ALL','H5P_DEFAULT',cell2mat(vidname)');
H5S.close(space_id);
H5T.close(type_id);
H5D.close(dset_id);
H5F.close(fid);
