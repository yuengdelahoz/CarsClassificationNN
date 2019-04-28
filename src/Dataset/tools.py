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
import sys

PATH = os.path.dirname(os.path.relpath(__file__))
devkit = os.path.join(PATH,'devkit')
ORIGINAL_IMAGES_PATH= os.path.join(PATH,'images/original')
CROPPED_IMAGES_PATH= os.path.join(PATH,'images/cropped')
RESIZED_IMAGES_PATH= os.path.join(PATH,'images/resized')
INPUT_IMAGES_PATH= os.path.join(PATH,'images/input')

train_original_images_path = os.path.join(ORIGINAL_IMAGES_PATH,'cars_train')
train_cropped_images_path = os.path.join(CROPPED_IMAGES_PATH,'cars_train')
train_resized_images_path = os.path.join(RESIZED_IMAGES_PATH,'cars_train')


def print_no_newline(string):
	sys.stdout.write(string)
	sys.stdout.flush()
	print('\r', end='')

def get_train_or_test_annotations(is_for_training=True):
	annotations_pickle = dict()
	if is_for_training:
		pickle_path = os.path.join(PATH,'train_annotations.pickle') 
		annotations_path =  os.path.join(devkit,'cars_train_annos.mat')
	else:
		pickle_path = os.path.join(PATH,'test_annotations.pickle')
		annotations_path =  os.path.join(devkit,'cars_test_annos_withlabels.mat')
	if not os.path.exists(pickle_path):
		annotations = loadmat(annotations_path).get('annotations')
		labels = loadmat(os.path.join(devkit,'cars_meta.mat')).get('class_names')
		bbox_x1 = annotations.get('bbox_x1')
		bbox_y1 = annotations.get('bbox_y1')
		bbox_x2 = annotations.get('bbox_x2')
		bbox_y2 = annotations.get('bbox_y2')
		label = annotations.get('class')
		fnames = annotations.get('fname')
		for i,fname in enumerate(fnames):
			data = dict()
			data['bounding_box']= [bbox_x1[i],bbox_y1[i],bbox_x2[i],bbox_y2[i]]
			data['class'] = label[i]-1
			data['class_name'] = labels[data['class']]
			annotations_pickle.update({fname:data})
		pickle.dump(annotations_pickle,open(pickle_path,'wb'))
	else:
		annotations_pickle = pickle.load(open( pickle_path , "rb"))
	return annotations_pickle


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

def remove_and_create_folder(folder):
	try:
		shutil.rmtree(folder)
	except:
		pass
	os.makedirs(folder)

def generate_cropped_images(is_for_training):
	cropped_images_path = os.path.join(PATH,'images/cropped')
	if is_for_training:
		original_images_path = os.path.join(ORIGINAL_IMAGES_PATH,'cars_train')
		cropped_images_path = os.path.join(cropped_images_path ,'cars_train')
	else:
		original_images_path = os.path.join(ORIGINAL_IMAGES_PATH,'cars_test')
		cropped_images_path = os.path.join(cropped_images_path ,'cars_test')

	remove_and_create = False
	try:
		if os.path.exists(cropped_images_path):
			if len(os.listdir(cropped_images_path)) != len(os.listdir(original_images_path)):
				remove_and_create = True
		else:
			remove_and_create = True
	except:
		remove_and_create = True
	if remove_and_create:
		remove_and_create_folder(cropped_images_path)
		annotations = get_train_or_test_annotations(is_for_training)
		total = len(annotations.values())
		print('Cropping images for {}'.format('training' if is_for_training else 'testing'))
		cnt = 0
		for img_name,values in annotations.items():
			print_no_newline('{} - {}%'.format(img_name,int(100*cnt/total)))
			img = cv2.imread('{}/{}'.format(original_images_path,img_name))
			x1,y1,x2,y2 = values['bounding_box']
			width = x2-x1
			height = y2-y1
			cropped_img = img[y1:y1+height,x1:x1+width]
			cv2.imwrite('{}/{}'.format(cropped_images_path,img_name),cropped_img)
			cnt +=1
		print('Done cropping images')

def generate_resized_images(is_for_training):
	if is_for_training:
		cropped_images_path = os.path.join(CROPPED_IMAGES_PATH,'cars_train')
		resized_images_path = os.path.join(RESIZED_IMAGES_PATH,'cars_train')
	else:
		cropped_images_path = os.path.join(CROPPED_IMAGES_PATH,'cars_test')
		resized_images_path = os.path.join(RESIZED_IMAGES_PATH,'cars_test')

	remove_and_create = False
	if not os.path.exists(cropped_images_path):
		generate_cropped_images(is_for_training)
	if os.path.exists(resized_images_path):
		if len(os.listdir(resized_images_path)) != len(os.listdir(cropped_images_path)):
			remove_and_create = True
	else:
		remove_and_create = True

	if remove_and_create:
		remove_and_create_folder(resized_images_path)
		print('Resizing images for {}'.format('training' if is_for_training else 'testing'))
		cnt = 0
		total = len(os.listdir(cropped_images_path))
		for image in os.scandir(cropped_images_path):
			print_no_newline('{} - {}%'.format(image.name,int(100*cnt/total)))
			img = cv2.imread(image.path)
			h,w,_ = img.shape
			aspect_ratio = w/h
			new_w = 240
			new_h = int(new_w/aspect_ratio)
			res_img = cv2.resize(img,(new_w,new_h))
			cv2.imwrite('{}/{}'.format(resized_images_path,image.name),res_img )
			cnt +=1
		print('Done resizing images')

def __get_resized_image_max_height():
	max_h = 0
	if not os.path.exists(train_resized_images_path):
		generate_resized_images(is_for_training=True)
	for image in os.scandir(train_resized_images_path):
		img = cv2.imread(image.path)
		h,w,_ = img.shape
		if h > max_h:
			max_h = h
	return int(max_h)

def generate_input_images(is_for_training):
	if is_for_training:
		resized_images_path = os.path.join(RESIZED_IMAGES_PATH,'cars_train')
		input_images_path = os.path.join(INPUT_IMAGES_PATH,'cars_train')
	else:
		resized_images_path = os.path.join(RESIZED_IMAGES_PATH,'cars_test')
		input_images_path = os.path.join(INPUT_IMAGES_PATH,'cars_test')
	remove_and_create = False

	if not os.path.exists(resized_images_path):
		result = generate_resized_images(is_for_training)
	if os.path.exists(input_images_path):
		if len(os.listdir(resized_images_path)) != len(os.listdir(input_images_path)):
			remove_and_create = True
	else:
		remove_and_create = True

	if remove_and_create:
		remove_and_create_folder(input_images_path )
		max_h = __get_resized_image_max_height()
		print('max_h',max_h)
		print('Resizing input images for {}'.format('training' if is_for_training else 'testing'))
		cnt = 0
		total = len(os.listdir(resized_images_path))
		for image in os.scandir(resized_images_path):
			print_no_newline('{} - {}%'.format(image.name,int(100*cnt/total)))
			img = cv2.imread(image.path)
			h,w,_ = img.shape
			input_image = np.zeros((max_h,w,3),np.uint8)
			offset = int((max_h-h)/2)
			input_image[offset:offset+h,0:w]=img
			cv2.imwrite('{}/{}'.format(input_images_path,image.name),input_image)
			cnt+=1

def generate_train_test_input_images():
	generate_input_images(is_for_training=True)
	generate_input_images(is_for_training=False)
