#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import images
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
import matplotlib.pyplot as plt
                               
# 内容表示层
content_layers = ['block1_conv2']
# 风格表示层
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]
# 卷积层数
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def model_init():
    """
    加载预训练的（imagenet）vgg19模型并提供对中间层输出的访问
    然后初始化一个新模型，该模型将图片作为输入并输出vgg19层输出的列表。
    返回内容特征和样式特征的模型
    """
    
    vgg19 = VGG19(include_top = False,weights = 'imagenet')
    vgg19.trainable = False

    # 依据层名或下标获得层对象
    content_outputs = [vgg19.get_layer(layername).output for layername in content_layers]
    style_outputs = [vgg19.get_layer(layername).output for layername in style_layers]
    model_outputs = content_outputs + style_outputs
   
    # 构造新模型
    model = Model(vgg19.input,model_outputs)
    
    return model    


def content_loss(base_content,target):
    """
    内容损失
    """
    c_loss = tf.reduce_mean(tf.square(base_content - target))/2 # 计算tensor（图像）的平均值
    return c_loss

def gram_matrix(input_tensor):
    channel = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor,[-1,channel]) # 图片格式改为 [H*W,C]
    gram = tf.matmul(a,a,transpose_a = True) # 矩阵相乘
    return gram

def style_loss(base_style,target):
    """
    风格损失
    """
    a = gram_matrix(base_style)
    b = gram_matrix(target)
    h,w,c = base_style.get_shape().as_list() #每层特征图的尺寸和通道数
    s_loss = tf.reduce_mean(tf.square(a - b))/(4*(h**2)*(w**2)*(c**2))
    return s_loss


def get_feature_representation(model,path,mode):
    """
    加载图片和模型，返回风格特征图或内容特征图
    """
    img = images.pre_process_img(path)
    feature_outputs = model(img)
    if mode == 'style':
        return [feature[0] for feature in feature_outputs[num_content_layers:]]
    if mode =='content':
        return [feature[0] for feature in feature_outputs[:num_content_layers]]


def loss(model,loss_weights,init_image,content_features,style_features):
    """
    损失函数
    
    参数:
    
     model : vgg-19
     loss_weights : 风格权重和内容权重，权重越大越相似
     init_image : 需要训练更新的生成图像
     content_features : 预训练的内容特征
     style_features : 预训练的风格特征
     
    返回:

    总的损失，风格损失，内容损失
    """
    style_weight,content_weight = loss_weights
    
    # 输入图片获得内容特征和风格特征
    features = model(init_image)
    gen_style_feature = features[num_content_layers:]
    gen_content_feature = features[:num_content_layers]
    
    total_style_loss = 0
    total_content_loss = 0
    
    # 分配风格权重，计算风格损失
    weight_per_style_layer = 1.0/ float(num_style_layers) #每层风格的权重=0.2
    for style_pic_features,gen_pic_stylefeatures in zip(style_features,gen_style_feature):
        total_style_loss += weight_per_style_layer * style_loss(style_pic_features,gen_pic_stylefeatures)
    
    # 分配内容权重，计算内容损失
    weight_per_content_layer = 1.0/ float(num_content_layers)
    for content_pic_features,gen_pic_contentfeatures in zip(content_features,gen_content_feature):
        total_content_loss += weight_per_content_layer * content_loss(content_pic_features,gen_pic_contentfeatures)

    # 计算总的损失
    total_style_loss *= style_weight
    total_content_loss *= content_weight
    total_loss = total_style_loss + total_content_loss
    return total_loss,total_content_loss,total_style_loss
    

def compute_grads(cfg):
    """
    计算梯度
    """
    with tf.GradientTape() as tape:
        allloss = loss(**cfg)
    #针对所生成图像的计算梯度
    total_loss = allloss[0]
    return tape.gradient(total_loss,cfg['init_image']),allloss

def plt_loss(tot_loss):
    x = []
    for i in range(1,len(tot_loss)+1):
        x.append(i*100)
    plt.plot(x, tot_loss)
    # plt.subplot(1, 2, 2)
    plt.title('tot_loss')

    plt.show()


def run_nst(content_path,style_path,iteration = 1000,content_weight = 1e3,style_weight = 1):
    model = model_init()
    # 冻结参数，不训练
    for layer in model.layers:
        layer.trainable = False

    # 获得内容特征和风格特征
    content_features = get_feature_representation(model,content_path,mode = 'content')
    style_features = get_feature_representation(model,style_path,mode = 'style')

    # 用内容图像初始化生成的图像
    init_image = images.pre_process_img(content_path)
    init_image = tf.Variable(init_image,dtype = tf.float32)

    # 使用Adam优化器
    opt = tf.keras.optimizers.Adam(5,beta_1 = 0.99,epsilon = 1e-1) #lr = 0.001

    # 内容权重和风格权重
    loss_weights = (content_weight,style_weight)
    
    cfg = {
        'model':model,
        'loss_weights':loss_weights,
        'init_image':init_image,
        'content_features':content_features,
        'style_features':style_features
    }

    # VGG 自带的一个常量，之前VGG训练通过归一化，所以现在同样需要作此操作
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means
    
    # 存储损失和图片
    best_loss, best_img = float('inf'), None
    imgs = []
    tot_loss = []
    k = 0
    start = datetime.now()
    for i in range(iteration):
        grads, all_loss = compute_grads(cfg)
        losss, content_losss, style_losss = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
        

        if losss < best_loss:
            # Update best loss and best image from total loss.
            best_loss = losss
            best_img = images.deprocess_img(init_image.numpy())

        if i % 100 == 0:
            end = datetime.now()
            print('[INFO]Iteration: {}'.format(i))
            print('Total loss: {:.4e}, '
                  'style loss: {:.4e}, '
                  'content loss: {:.4e}'
                  .format(losss, style_losss, content_losss))
            print(f'100 iters takes {end -start}')
            if k == 0:
                k = 1
            else:
                tot_loss.append(int(losss))
            start = datetime.now()
        if i % 100 == 0:
            # Use the .numpy() method to get the concrete numpy array
            plot_img = init_image.numpy()
            plot_img = images.deprocess_img(plot_img)
            path = 'output/style9/output_' + str(i) + '.jpg'
            images.saveimg(plot_img, path)
            imgs.append(plot_img)
    plt_loss(tot_loss)
    images.saveimg(best_img, 'output/style11/output_' + str(iteration) + '.jpg')
    return best_img, best_loss
    
    

