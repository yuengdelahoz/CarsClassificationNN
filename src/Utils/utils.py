#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 yuengdelahoz <yuengdelahoz@TAOs-Macbook-Pro.local>
#
# Distributed under terms of the MIT license.

"""
"""

import sys, os, shutil, cv2, shutil, pickle, traceback
import numpy as np
from collections import namedtuple
import threading
import cv2
import shutil

def clear_folder(name):
	if os.path.isdir(name):
		try:
			shutil.rmtree(name)
		except Exception as e:
			traceback.print_exc()

def create_folder(name,clear_if_exists = True):
	if clear_if_exists:
		clear_folder(name)
	try:
		os.makedirs(name)
		return name
	except:
		pass

def is_model_stored(topology_path):
	try:
		model_files = os.listdir(topology_path)
		model_stored = False
		for mf in model_files:
			if 'model' in mf:
				model_stored = True
				break
		return model_stored
	except:
		return False

def print_no_newline(string):
	sys.stdout.write(string)
	sys.stdout.flush()
	print('\r', end='')

if __name__ == '__main__':
	# generate_new_labels()
	createSuperLabels()
	# paint_all_images_with_text()


