���ݣ�
1.��Pascal VOC2012���ݼ��м�¼ͼ��ָ������ѵ����ͼ�����ƺ���֤��ͼ�����Ƶ�train.txt��val.txt����data�ļ����¡�
2.��Pascal VOC2012���ݼ��е�JPEGImages��SegmentationClass�����ļ��з���data�ļ����¡�
3.data�ļ����е�labels��ž�����õ���ÿһ��ͼ��ָ����ȷ�𰸣���label����
4.tfrecords�ļ����д�����ɵ�tfrecord��tf�����ݴ洢��ʽ����
5.checkpoint�д��ѵ���õ���ģ�Ͳ�����
ѵ����
1.����convert_png_to_labels.py����labels��
2.����create_tfrecords.py��ѵ�����ݺ���֤����д��tfrecord��
3.����unetvgg16.py����ģ��ѵ�����洢ѵ��������
4.����metric.py����ѵ���õ���ģ�ͣ��ڲ��Լ�������ģ�����ܡ�