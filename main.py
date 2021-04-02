#!/usr/bin/env python
# coding: utf-8

import model

content_path = 'input/content/content3.jpg'
style_path = 'input/style/style7.jpg'

if __name__ == "__main__":
    # print(f'GPU AVAILABLE :{tf.test.is_gpu_available()}')
    model.run_nst(content_path, style_path, iteration=2000)

