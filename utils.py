import cv2
import numpy as np
import os
import tensorflow as tf
import pcl


BASE_DIR = '/home/cv/261/3DPoseEstimation/3D_Models/png'


classes=["Budha", "Chair", "Coindog", "Glass", "Gnome", "Gun", "Mouse", "Pikaqiu", "Skull"]


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def load_and_normalize_img_raw(addr):
    img = cv2.imread(addr)
    img =cv2.resize(img, (224, 224))
    img = img / 255
    img = img.astype(np.float32)
    img_raw = img.tobytes()
    return img_raw


def img2tfrecord(sample_type):
    writer = tf.python_io.TFRecordWriter(sample_type+".tfrecords")
    for i, name in enumerate(classes):
        #class_path = BASE_DIR+'/'+name+"_depth_small_size/"
        class_path = BASE_DIR + '/' + name + "/"
        addrs = os.listdir(class_path)
        #tfrecord_filename = BASE_DIR + "/" + name + "_"+sample_type+".tfrecords"
        label_path = BASE_DIR+"/"+name+".txt"
        with open(label_path, 'r') as label_file:
            lines_list = label_file.readlines()

        if(sample_type == "train"):
            sample_addrs = addrs[:int(0.9*len(addrs))]
        elif(sample_type == "valid"):
            sample_addrs = addrs[int(0.9*len(addrs)):]

        for index, img_name in enumerate(sample_addrs):          #3D_Models/png/Buhda_depth_small_size_
            img_path = class_path+img_name
            img_raw = load_and_normalize_img_raw(img_path)
            label_line = lines_list[int(img_name.split('_')[1][:-4])]  #from Buhda_123.png obtain '123' as index
            #print(name+'_'+label_line.split(',')[0]+".png", img_name)
            assert(name+'_'+label_line.split(',')[0]+".png" == img_name)
            label1 = float(label_line.split(',')[1])
            label2 = float(label_line.split(',')[2])
            feature = {sample_type+"/label1": _float_feature(label1),
                       sample_type+"/label2": _float_feature(label2),
                       sample_type+"/image": _bytes_feature(img_raw)}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
    writer.close()
    print(sample_type+"image to tfrecord done")


#TODO
def pcd2tfrecord(sample_type):
    writer = tf.python_io.TFRecordWriter(sample_type+"_pcd.tfrecords")
    for i, name in enumerate(classes):
        class_path = BASE_DIR + '/' + name + "_pcd/"
        addrs = os.listdir(class_path)
        label_path = BASE_DIR+"/"+name+".txt"
        with open(label_path, 'r') as label_file:
            lines_list = label_file.readlines()

        if(sample_type == "train"):
            sample_addrs = addrs[:int(0.9*len(addrs))]
        elif(sample_type == "valid"):
            sample_addrs = addrs[int(0.9*len(addrs)):]

        for index, img_name in enumerate(sample_addrs):          #3D_Models/png/Buhda_pcd
            img_path = class_path+img_name
            img_raw = load_and_normalize_img_raw(img_path)
            label_line = lines_list[int(img_name.split('_')[1][:-4])]  #from Buhda_123.png obtain '123' as index
            #print(name+'_'+label_line.split(',')[0]+".png", img_name)
            assert(name+'_'+label_line.split(',')[0]+".png" == img_name)
            label1 = float(label_line.split(',')[1])
            label2 = float(label_line.split(',')[2])
            feature = {sample_type+"/label1": _float_feature(label1),
                       sample_type+"/label2": _float_feature(label2),
                       sample_type+"/image": _bytes_feature(img_raw)}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
    writer.close()
    print(sample_type+"image to tfrecord done")



def read_and_decode(sample_type):
    filename_queue = tf.train.string_input_producer([sample_type+".tfrecords"])

    feature = {sample_type+"/label1": tf.FixedLenFeature([], tf.float32),
               sample_type+"/label2": tf.FixedLenFeature([], tf.float32),
               sample_type+"/image": tf.FixedLenFeature([], tf.string)}
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, feature)
    img = tf.decode_raw(features[sample_type+"/image"], tf.float32)
    img = tf.reshape(img, [224, 224, 3])
    label1 = tf.cast(features[sample_type+"/label1"], tf.float32)
    label2 = tf.cast(features[sample_type+"/label2"], tf.float32)
    label = tf.stack([label1, label2])

    return img, label




