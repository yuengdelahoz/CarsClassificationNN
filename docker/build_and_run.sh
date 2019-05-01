#! /bin/sh
#
# docker_setup.sh
# Copyright (C) 2019 ysdelahoz <ysdelahoz@ENB302-PC1>
#
# Distributed under terms of the MIT license.
#


docker build -t cars_classification /home/ysdelahoz/Projects/CarsClassificationNN/docker &&
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="1" --rm -it -v /home/ysdelahoz/Projects/CarsClassificationNN/src:/usr/src/app cars_classification 
