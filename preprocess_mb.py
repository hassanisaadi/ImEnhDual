#! /usr/bin/env python2

import numpy as np
import os
import subprocess
import cv2

def read_im(fname, downsample):
    if downsample:
        if not os.path.isfile(fname + '.H.png'):
            subprocess.check_call('convert {} -resize 50% {}.H.png'.format(fname, fname).split())
        fname += '.H.png'
    x = cv2.imread(fname).astype(np.float32)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose(2, 0, 1)
    x = (x - x.mean()) / x.std()
    return x[None]

def tofile(fname, x):
    if x is None:
        open(fname + '.dim', 'w').write('0\n')
        open(fname, 'w')
    else:
        x.tofile(fname)
        open(fname + '.type', 'w').write(str(x.dtype))
        open(fname + '.dim', 'w').write('\n'.join(map(str, x.shape)))


X = []
Y = []
meta = []
output_dir = 'data.mb.2014'
test_samples = 3

### 2014 dataset ###
base1 = 'data.mb/unzip/vision.middlebury.edu/stereo/data/scenes2014/datasets'
for dir in sorted(os.listdir(base1)):
    if dir.endswith('imperfect'):
        print(dir.split('0')[0])

        base2_imperfect = os.path.join(base1, dir)
        
        y = read_im(os.path.join(base2_imperfect, 'im0.png'), True)
        XX = []

        base3 = os.path.join(base2_imperfect, 'ambient')
        num_light = len(os.listdir(base3))

        for l in range(num_light):
            print('l = '+str(l))
            imgs = []
            for fname in sorted(os.listdir(base3 + '/L{}'.format(l+1))):
                base4 = os.path.join(base3, 'L{}'.format(l+1))
                cam = int(fname[2])
                if cam == 0 and not fname.endswith('H.png'):
                    cam = 1
                    exp = fname[4]
                    fname2 = fname[0:2] + str(cam) + 'e' + exp + fname[5:]
                    if not os.path.isfile(base4 + '/' + fname2):
                        continue
                    # print('  fname = '+fname)
                    # print('  fname2= '+fname2)
                    im0 = read_im(os.path.join(base4,fname ), True)
                    im1 = read_im(os.path.join(base4,fname2), True)
                    imgs.append(im0)
                    imgs.append(im1)
            _, _, height, width = imgs[0].shape
            print(imgs[0].shape)
            print(len(imgs))
            XX.append(np.concatenate(imgs).reshape(len(imgs)//2, 2, 3, height, width))

        X.append(XX)
        Y.append(y)
        meta.append((y.shape[2], y.shape[3]))
print('2014 dataset processed!')

print('There are {} images in this dataset.'.format(len(meta)))
print('X len = ' + str(len(X)))
print('Y len = ' + str(len(Y)))

meta = np.array(meta, dtype=np.int32)
tofile('{}/meta.bin'.format(output_dir), meta)

for i in range(len(X)):
    if i <= len(X) - test_samples:
        tmp = 'train'
    else:
        tmp = 'test'
    for j in range(len(X[i])):
        tofile('{}/x_{}_{}_{}.bin'.format(output_dir, tmp, i+1, j+1), X[i][j])
    tofile('{}/y_{}_{}.bin'.format(output_dir, tmp, i+1), Y[i])

