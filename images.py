#!/usr/bin/env python
# coding: utf-8


import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image as kp_image
import matplotlib.pyplot as plt

def loadimg(path_to_img):
    """
    加载图像并将其缩放到合适的尺寸
    """
    img = Image.open(path_to_img)
    longer_dim = max(img.size) # 图片中宽和高更长的一个
    max_dim = 900 # 输出图像的最大尺寸
    scale = max_dim / longer_dim
    # img[0]宽,img[1]高，对图片的宽和高进行缩放
    img = img.resize((round(img.size[0] * scale),round(img.size[1] * scale)),Image.ANTIALIAS)
    img = kp_image.img_to_array(img)
    img = np.expand_dims(img,axis = 0) #扩展数组形状
    return img

def showimg(img):
    out = np.squeeze(img,axis = 0)
    out = out.astype('uint8')
    plt.axis('off')
    plt.imshow(out)

def pre_process_img(path_to_img):
    """
    图片预处理 标准化
    """
    img = loadimg(path_to_img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def deprocess_img(processed_img):
    img = processed_img.copy()
    if len(img.shape) == 4:
        # 删除维度
        img = np.squeeze(img,0)
    assert len(img.shape) == 3,('deprocess_img的尺寸输入必须为'
                                '[1,height,width,channel] or [height,width,channel]')
    if len(img.shape) != 3:
        raise ValueError('无效的输入')

    # 减去ImageNet的平均像素值，使其中心为0
    # 去均值，有利于三通道的训练效果
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1] # 将图像由BGR格式转换为RGB格式。
    
    img = np.clip(img,0,255).astype('uint8') # 裁剪x中元素到指定范围
    return img

def saveimg(bestimg,path):
    img = Image.fromarray(bestimg) # 实现array到image的转换
    img.save(path)

