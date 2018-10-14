all:
	echo "Please don't just run make"
	echo "Run me with make setup or make run"

setup: clone-repos ckpts s3-dir

clone-repos:
	git clone http://github.com/tensorflow/models.git tf-models

ckpts:
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

s3-dir:
	mkdir learningsys-2018-gpu-mux
	s3 sync s3://learningsys-2018-gpu-mux learningsys-2018-gpu-mux

format:
	isort *.py
	black *.py
	