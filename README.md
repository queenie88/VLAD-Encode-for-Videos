# VLAD-Encode-for-Videos

Use VLAD to encode videos' CNN latent concept descritor. 

The purpose of this project is to accomplish the idea in the paper, A Discriminative CNN Video Representation for Event Detection, which is published in CVPR 2015. 
The two contribution of this paper are those:

1. CNN latent concept descriptor.
2. Vector of locally aggregated descriptors (VLAD) encoding.

# Environment Required: 

For python: 

1. gpu.
2. caffe. 
3. caffe trained model parameters: VGG_ILSVRC_16_layers.caffemodel, can be found in caffe model zoo. 
4. caffe model configuration : VGG_ILSVRC_16_layers_deploy.prototxt, can be found in caffe model zoo. 
5. Linux. 

For matlab: 

1. vlfeat.
2. libsvm.
3. Linux or Windows


# Files and Functions: 

1. python/vgg_feature_pool5.py - Will extract the features of VGG_Net's pool5 layer and use the features to generate the CNN latent concept descriptors. The pool5's latent concept descriptors (pool5lcd) will be saved in the .h5 file. A demo of this python script can be found in pool5lcd-extractor.sh.
2. python/pool5lcd-extractor.sh - Will call vgg_feature_pool5.py. Input required: input_file, output_file, the beginning of file, the ending of file, GPU_number and batch_size. 
3. matlab/vlad_build_center.m - Will generate VLAD_centers for VLAD encode. Input required: input_file, the number of files. 
4. matlab/vlad_encode_h5_video.m - Will VLAD encode the CNN latent concept descritors. A demo of this matlab
5. matlab/vlad_encode_h5_main.m - Will call vlad_encode_h5_video.m. Input required: input_file, output_file, the number of files. 
