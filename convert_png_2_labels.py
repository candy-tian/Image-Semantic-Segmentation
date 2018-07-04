from skimage.io import imread, imsave
import numpy as np

object_dic = {'background': 0, 'aeroplane': 1,  'bicycle': 2,  'bird': 3,  'boat': 4,
              'bottle': 5,  'bus': 6,  'car': 7,  'cat': 8,
              'chair': 9,  'cow': 10, 'diningtable': 11, 'dog': 12,
              'horse': 13, 'motorbike': 14, 'person': 15, 'potted-plant': 16,
              'sheep': 17, 'sofa': 18, 'train': 19, 'tv/monitor': 20}

pixel_dic = {(0, 0, 0): 0, (128, 0, 0): 1,
             (0, 128, 0): 2, (128, 128, 0): 3,
             (0, 0, 128): 4, (128, 0, 128): 5,
             (0, 128, 128): 6, (128, 128, 128): 7,
             (64, 0, 0): 8, (192, 0, 0): 9,
             (64, 128, 0): 10, (192, 128, 0): 11,
             (64, 0, 128): 12, (192, 0, 128): 13,
             (64, 128, 128): 14, (192, 128, 128): 15,
             (0, 64, 0): 16, (128, 64, 0): 17,
             (0, 192, 0): 18, (128, 192, 0): 19,
             (0, 64, 128): 20}


def convert_from_color_segmentation(raw_label):
    label = np.zeros((raw_label.shape[0], raw_label.shape[1]), dtype=np.uint8)
    for c, i in pixel_dic.items():
        m = np.all(raw_label == np.array(c).reshape(1, 1, 3), axis=2)
        label[m] = i
    return label

dir = './data/SegmentationClass/'
image_list = []
file = open('./data/trainval.txt')
for line in file:
    if line != '':
        image_list.append(line.strip()[:] + '.png')

for img_file in image_list:
    img = imread(dir + img_file)
    label = convert_from_color_segmentation(img)
    imsave('./data/labels/' + img_file, label)
print('All segmentations have been converted to labels.')
