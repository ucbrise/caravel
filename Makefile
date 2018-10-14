all:
	echo "Please don't just run make"
	echo "Run me with make setup or make run"

setup: clone-repos ckpts s3-dir

clone-repos:
	git clone http://github.com/tensorflow/models.git tf-models

ckpts:
	bash bin/download-ckpts.sh

s3-dir:
	mkdir learningsys-2018-gpu-mux
	aws s3 sync s3://learningsys-2018-gpu-mux learningsys-2018-gpu-mux

format:
	bash bin/format-code.sh

push: format
	git commit -a
	git push origin master
