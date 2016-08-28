# Author: Xiukun Huang
# Date: April, 2016

import numpy as np
from scipy import misc
from scipy.misc import imresize
import sys
import argparse
import os
import matplotlib 
import theano
from theano import tensor as T
from theano.tensor.signal import downsample
matplotlib.use('Agg')
import time
import glob
import logging
import h5py

caffe_root = "/home/huangxiukun/caffe/root"
sys.path.insert(0, caffe_root + '/python')
import caffe 


mean = np.array([103.939, 116.779, 123.68])
mean = mean[:, np.newaxis, np.newaxis]
IMG_HEIGHT=224
IMG_WIDTH=224
vgg_dims=np.array([IMG_HEIGHT,IMG_WIDTH])
input = T.dtensor4('input')
pool_out_77 = downsample.max_pool_2d(input, (7,7), st=(7,7), ignore_border=True)
max_pool_77 = theano.function([input],pool_out_77)
pool_out_44 = downsample.max_pool_2d(input, (4,4), st=(3,3), ignore_border=True)
max_pool_44 = theano.function([input],pool_out_44)
pool_out_33 = downsample.max_pool_2d(input, (3,3), st=(2,2), ignore_border=True)
max_pool_33 = theano.function([input],pool_out_33)
pool_out_22 = downsample.max_pool_2d(input, (2,2), st=(1,1), ignore_border=True)
max_pool_22 = theano.function([input],pool_out_22)


def extract_pool5(image_list,net):
    img_ins=list();
    for image in image_list:
        image_dims=image.shape[0:2]
        min_dim=min(image_dims)

        if image_dims[0]>image_dims[1]:
            img = np.asarray(imresize(image,(int(224.0/image_dims[1]*image_dims[0]),224) ))
        if image_dims[0]<image_dims[1]:
            img = np.asarray(imresize(image,(224,int(224.0/image_dims[0]*image_dims[1])) ))
        if image_dims[0]==image_dims[1]:
            img = np.asarray(imresize(image,(224,224) ))

        # Take center crop.
        img_dims=img.shape[0:2]
        center = np.array(img_dims) / 2.0
        crop = np.tile(center, (1, 2))[0] + np.concatenate([-vgg_dims / 2.0,vgg_dims / 2.0])

        if img.ndim < 3:
            if img_dims[0]>img_dims[1]:
                img = img[crop[0]:crop[2],:]
            if img_dims[0]<img_dims[1]:
                img = img[:, crop[1]:crop[3]]
        else:
            if img_dims[0]>img_dims[1]:
                img = img[crop[0]:crop[2],: , :]
            if img_dims[0]<img_dims[1]:
                img = img[:, crop[1]:crop[3], :]

        img_in = np.zeros((3,IMG_HEIGHT,IMG_WIDTH),dtype='float32')
        if img.ndim < 3:
            img_in[0,:,:]=img
            img_in[1,:,:]=img
            img_in[2,:,:]=img
        else:
            img_in = np.asarray(np.transpose(img,(2,0,1))[(2,1,0),:,:],dtype='float32')
            
        img_ins.append(img_in)
    print len(img_ins)
    img_ins=np.asarray(img_ins)
    print img_ins.shape

    img_ins-=mean
    out=net.forward_all(blobs=['pool5',],**{net.inputs[0]:img_ins})

    outdata=out['pool5']
    outdata77=max_pool_77(outdata)
    outdata44=max_pool_44(outdata)
    outdata33=max_pool_33(outdata)
    outdata22=max_pool_22(outdata)
    img_num=len(outdata)

    feature_pt77=np.reshape(outdata77,(img_num,512,1))
    feature_pt44=np.reshape(outdata44,(img_num,512,4))
    feature_pt33=np.reshape(outdata33,(img_num,512,9))
    feature_pt22=np.reshape(outdata22,(img_num,512,36))
    
    feature=np.concatenate((feature_pt77,feature_pt44,feature_pt33,feature_pt22), axis=2)
    print 'feature shape:' 
    print feature.shape
    return feature

        
def main(argv):
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="Input image, directory."
    )
    parser.add_argument(
        "output_file",
        help="Output npy filename."
    )
    # Optional arguments.
 
    parser.add_argument(
        "--model_def",
        default=os.path.join(caffe_root,
                "models/vgg/VGG_ILSVRC_16_layers_deploy.prototxt"),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(caffe_root,
                "models/vgg/VGG_ILSVRC_16_layers.caffemodel"),
        help="Trained model weights file."
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Switch for gpu computation, value is gpu device_id ."
    )
    parser.add_argument(
        "--ext",
        default='jpeg',
        help="Image file extension to take as input when a directory " +
             "is given as the input file."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Image batch size, due to memory limit. " 
    )
    args = parser.parse_args()

    args.input_file = os.path.expanduser(args.input_file)
    args.output_file = os.path.expanduser(args.output_file)

    # initial logging
    log_path = os.path.dirname(args.output_file) 
    mkdirs(log_path)
    log_filename = args.output_file + '.log'

    log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=log_filename,
                        filemode='w',
                        level=logging.INFO,
                        format=log_format)

    logger = logging.getLogger(__name__) # get logger

    logger.info("input file: %s" % args.input_file)
    logger.info("output file: %s" % args.output_file)
    logger.info("model file: %s" % args.model_def)
    logger.info("pretrained model : %s" % args.pretrained_model)
    logger.info("gpu id : %s" % args.gpu)
    logger.info("file ext: %s" % args.ext)
    logger.info("batch_size: %s" % args.batch_size)


    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    net=caffe.Net(args.model_def,args.pretrained_model,caffe.TEST)

    print("Loading folder: %s" % args.input_file)
    image_paths = glob.glob(args.input_file + '/*.' + args.ext)
    image_paths.sort();
    image_count = 0
    image_num = len(image_paths)
    batch_size = args.batch_size
    current_batch_size = args.batch_size
    batch_count = 0
    batch_num = image_num/batch_size
    remainder = image_num % batch_size
    if remainder !=0 : batch_num += 1

    si=0;
    ei=0;

    inputs=[]
    filename=[]
    output_filename =  args.output_file + '.h5'
    f=h5py.File(output_filename,"w")
    f.create_dataset("pic_name",(0,),dtype="S25",maxshape=(None,))
    f.create_dataset('pool5lcd',(0,512,50),dtype="float32",maxshape=(None,512,50),compression="gzip", compression_opts=9)
        
    for im_f in image_paths:
        inputs.append(misc.imread(im_f))
        filename.append(os.path.basename(im_f))
        image_count += 1
        if image_count%1000 ==0 : print('image count: %d' % image_count)
        if ((image_count % batch_size)==0) or (image_count == image_num):
            if (image_count == image_num) and (remainder != 0) : current_batch_size=remainder
            print("Extracting features %d inputs." % len(inputs))
            logger.info("Extracting features %d inputs." % len(inputs))
            batch_count += 1
            # extraction.
            start = time.time()
            features = extract_pool5(inputs,net)

            print("Done in %.2f s." % (time.time() - start))
            logger.info("Done in %.2f s." % (time.time() - start))

            del inputs[:]

            print("batch num: %d,  features extracted in batch #: %d" % (batch_num,batch_count))
            logger.info("batch num: %d,  features extracted in batch #: %d" % (batch_num,batch_count))
            print ("pool5-lcd:%s" % ( features.shape,))
            logger.info ("pool5-lcd:%s" % ( features.shape,))

            ei += current_batch_size
            f['pic_name'].resize(ei,axis=0)
            f['pic_name'][si:ei]=filename
            f['pool5lcd'].resize(ei,axis=0)
            f['pool5lcd'][si:ei]=features
            
            f.flush() 
            si=ei

            print("Save results into %s done..." % output_filename)
            logger.info("Save results into %s done..." % output_filename)

            del filename[:]
    
    f.close()
    print("Extraction finished!")
    logger.info("Extraction finished!")

if __name__ == '__main__':
    main()
