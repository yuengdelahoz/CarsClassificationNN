import os,sys
import numpy as np
import cv2
import collections
from tools import clear_folder
import pickle
import tarfile
import wget
import traceback

Datasets = collections.namedtuple('Datasets', ['training', 'testing','validation'])

class Dataset():
	def __init__(self,images, gt = 'superlabel'):
		self.instances = images
		self.num_of_images = len(images)
		self.images_path = os.path.dirname(__file__)+'/images' 
		self.ground_truth = gt
		self.index = 0

	def next_batch(self, batch_size):
		if batch_size > self.num_of_images:
			raise ValueError("Dataset error...batch size is greater than the number of samples")

		start = self.index
		self.index += batch_size

		if self.index > self.num_of_images:
			np.random.shuffle(self.instances)
			# Shuffle the data
			self.index = batch_size
			start = 0

		end = self.index
		imgs = self.instances[start:end]
		imagesBatch = []
		labelsBatch = []
		for img in imgs:
			if img.endswith('.png'):
				try:
					image = cv2.imread(self.images_path+'/input/'+img)
					label = np.load(self.images_path+'/'+self.ground_truth+'/'+img.replace('png','npy'))
				except:
					continue
				imagesBatch.append(image)
				labelsBatch.append(label)
		return np.array((imagesBatch,labelsBatch))

class DataHandler:
	__TOTAL_NUMBER_OF_TRAIN_IMAGES = 8144
	__TOTAL_NUMBER_OF_TEST_IMAGES = 8041
	__TRAIN_IMAGES_URL = "http://imagenet.stanford.edu/internal/car196/cars_train.tgz"
	__TEST_IMAGES_URL = "http://imagenet.stanford.edu/internal/car196/cars_test.tgz"
	def __init__(self):
		self.path = os.path.dirname(os.path.relpath(__file__))
		self.train_tarfile_path= os.path.join(self.path,'cars_train.tgz')
		self.test_tarfile_path= os.path.join(self.path,'cars_test.tgz')

	def build_datasets(self):
		images_path = os.path.join(self.path,'images')
		train_images_path = os.path.join(images_path,'cars_train')
		test_images_path = os.path.join(images_path,'cars_test')
		attempt_download_and_or_extraction = False
		data_ready = False
		if os.path.exists(images_path):
			try:
				if len(os.listdir(train_images_path)) != DataHandler.__TOTAL_NUMBER_OF_TRAIN_IMAGES :
					clear_folder(train_images_path)
					attempt_download_and_or_extraction = True
			except:
				clear_folder(train_images_path)
				attempt_download_and_or_extraction = True
			try:
				if len(os.listdir(test_images_path)) != DataHandler.__TOTAL_NUMBER_OF_TEST_IMAGES :
					clear_folder(test_images_path)
					attempt_download_and_or_extraction = True
			except:
				clear_folder(test_images_path)
				attempt_download_and_or_extraction = True
			else:
				data_ready = True
		else:
			attempt_download_and_or_extraction = True

		if attempt_download_and_or_extraction:
			is_download_complete=self.__maybe_download_files()
			if is_download_complete:
				print('Extracting images')
				is_train_data_ready = self.__extract_tarfile(self.train_tarfile_path,images_path )  
				is_test_data_ready = self.__extract_tarfile(self.test_tarfile_path,images_path )  
				if is_train_data_ready and is_test_data_ready :
					print('Extraction completed')
					data_ready = True
				else:
					print('Extraction incompleted')
		# if data_ready:
			# tar_file = self.path+'/images.tar'
			# if os.path.exists(tar_file):
				# try:
					# os.remove(tar_file)
					# print('images.tar was removed')
				# except:
					# pass

			# dataset_pickle_path = self.path+"/dataset.pickle"
			# if not os.path.exists(dataset_pickle_path):
				# keys = os.listdir(images_path+'/input')
				# np.random.shuffle(keys)
				# sz = len(keys)
				# train_idx = int(sz*0.7)
				# test_idx = int(sz*0.95)
				# dset = {'training':keys[:train_idx]}
				# dset.update({'testing':keys[train_idx:test_idx]})
				# dset.update( {'validation':keys[test_idx:]})
				# pickle.dump(dset,open(dataset_pickle_path,"wb"))
			# else:
				# dset = pickle.load(open(dataset_pickle_path,'rb'))

			# return Datasets(training=Dataset(dset['training']),
					# testing=Dataset(dset['testing']),
					# validation=Dataset(dset['validation']))
 
	def __extract_tarfile(self,tar_file,path):
		try:
			tf = tarfile.open(tar_file)
			tf.extractall(path)
			return True
		except:
			traceback.print_exc()
			return False
		
	def __maybe_download_files(self):
		if os.path.exists(self.train_tarfile_path) and os.path.exists(self.test_tarfile_path):
			return True
		else:
			if not os.path.exists(self.train_tarfile_path):
				print('Downloading train images')
				wget.download(DataHandler.__TRAIN_IMAGES_URL,out=self.path)
			if not os.path.exists(self.test_tarfile_path):
				print('Downloading test images')
				wget.download(DataHandler.__TEST_IMAGES_URL,out=self.path)
		return os.path.exists(self.train_tarfile_path) and os.path.exists(self.test_tarfile_path)

if __name__ == '__main__':
	DataHandler().build_datasets()
