数据：
1.将Pascal VOC2012数据集中记录图像分割任务的训练集图像名称和验证集图像名称的train.txt和val.txt放在data文件夹下。
2.将Pascal VOC2012数据集中的JPEGImages和SegmentationClass两个文件夹放在data文件夹下。
3.data文件夹中的labels存放经处理得到的每一张图像分割的正确答案（即label）。
4.tfrecords文件夹中存放生成的tfrecord（tf的数据存储格式）。
5.checkpoint中存放训练得到的模型参数。
训练：
1.运行convert_png_to_labels.py生成labels。
2.运行create_tfrecords.py将训练数据和验证数据写成tfrecord。
3.运行unetvgg16.py进行模型训练，存储训练参数。
4.运行metric.py导入训练得到的模型，在测试集上评估模型性能。