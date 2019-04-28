#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 yueng.delahoz <yueng.delahoz@ip-172-16-20-42.ec2.internal>
#
# Distributed under terms of the MIT license.

"""

"""
from mat4py import loadmat
import pickle
import os
import cv2
import numpy as np
import shutil
import traceback

PATH = os.path.dirname(os.path.relpath(__file__))
devkit = os.path.join(PATH,'devkit')
ann = loadmat(devkit+'/cars_train_annos.mat').get('annotations')
labels = loadmat(devkit+'/cars_meta.mat').get('class_names')

def get_training_set():
	training_set = dict()
	file_path = os.path.join(PATH,'traininig_set.pickle')
	if not os.path.exists(file_path):
		bbox_x1 = ann.get('bbox_x1')
		bbox_y1 = ann.get('bbox_y1')
		bbox_x2 = ann.get('bbox_x2')
		bbox_y2 = ann.get('bbox_y2')
		label = ann.get('class')
		fnames = ann.get('fname')
		for i,fname in enumerate(fnames):
			data = dict()
			data['bounding_box']= [bbox_x1[i],bbox_y1[i],bbox_x2[i],bbox_y2[i]]
			data['class'] = label[i]-1
			data['class_name'] = labels[data['class']]
			training_set.update({fname:data})
		pickle.dump(training_set,open(file_path,'wb'))
	else:
		training_set = pickle.load(open( file_path , "rb"))
	return training_set

def get_training_labels():
	return labels

def clear_folder(name):
	if os.path.isdir(name):
		try:
			shutil.rmtree(name)
		except:
			traceback.print_exc()

def create_folder(name,clear_if_exists = True):
	if clear_if_exists:
		clear_folder(name)
	try:
		os.makedirs(name)
		return name
	except:
		pass

def remove_folder(folder):
	try:
		shutil.rmtree(folder)
	except:
		pass
	os.makedirs(folder)

def build_images():
	remove_folder('images/cropped')
	remove_folder('images/resized')
	remove_folder('images/output')
	train_set = get_training_set()
	min_size =99999
	max_size =0
	sml_img = ""
	big_img = ""

	for img_name,values in train_set.items():
		img = cv2.imread('images/original/{}'.format(img_name))
		x1,y1,x2,y2 = values['bounding_box']
		width = x2-x1
		height = y2-y1
		cropped_img = img[y1:y1+height,x1:x1+width]
		cv2.imwrite('images/cropped/{}'.format(img_name),cropped_img )
		if cropped_img.size > max_size:
			big_img = img_name
			max_size = img.size

	sml_img = cv2.imread('images/cropped/{}'.format(big_img))
	h_ref,w_ref,_ = sml_img.shape

	max_h =0
	for image in os.scandir('images/cropped'):
		img = cv2.imread(image.path)
		h,w,_ = img.shape
		aspect_ratio = w/h
		new_h = int(w_ref/aspect_ratio)
		res_img = cv2.resize(img,(w_ref,new_h))
		cv2.imwrite('images/resized/{}'.format(image.name),res_img )
		if new_h > max_h:
			max_h = new_h

	for image in os.scandir('images/resized'):
		img = cv2.imread(image.path)
		h,w,_ = img.shape
		b_image = np.zeros((int(max_h),w,3),np.uint8)
		offset = int((max_h-h)/2)
		print(b_image.shape,'offset',offset)
		b_image[offset:offset+h,0:w]=img
		aspect_ratio = w/int(max_h)
		new_w = 500
		new_h = int(new_w/aspect_ratio)
		res_img = cv2.resize(b_image ,(new_w,new_h))
		cv2.imwrite('images/output/{}'.format(image.name),res_img )

