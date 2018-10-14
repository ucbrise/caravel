mkdir ckpts
cd ckpts

wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
tar xzf resnet_v1_50_2016_08_28.tar.gz
rm resnet_v1_50_2016_08_28.tar.gz

wget http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz
tar xzf resnet_v1_152_2016_08_28.tar.gz
rm resnet_v1_152_2016_08_28.tar.gz

wget https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_96.tgz
tar -xvf mobilenet_v2_1.0_96.tgz
rm  mobilenet_v2_1.0_96.tgz

cd ..