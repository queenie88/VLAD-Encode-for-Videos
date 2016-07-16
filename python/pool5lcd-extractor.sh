#!/bin/sh
# Author: Xiukun Huang
# Date: April, 2016

#sh pool5lcd-extractor.sh 3 5 0 500  # folder from 3 to 5  use gpu 0  batch_size 500
echo "Event ${1} to ${2}, GPU ${3}"
for i in $(seq ${1} ${2}); do
    c=$(printf "%04d" ${i} )
    #python vgg_feature_pool5.py /mnt/disk1/huangxiukun/vine_download_video_61/20160318url_content_html/CVPR/folder_of_frame/${c} /mnt/disk1/huangxiukun/vine_download_video_61/20160318url_content_html/CVPR/cnnFeature/feature_numIs113${c} --ext jpeg --gpu ${3} --batch_size ${4} 
    python vgg_feature_pool5.py /mnt/disk1/huangxiukun/CVPR_vlad_encode/test_folder/folder_of_frame/${c} /mnt/disk1/huangxiukun/CVPR_vlad_encode/test_folder/cnnFeatures/cnnFeatures_tagNumIs146_${c} --ext jpeg --gpu ${3} --batch_size 
done
