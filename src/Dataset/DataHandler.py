import os,sys
import numpy as np
import cv2
import collections
import pickle
import tarfile
import wget
import traceback

PATH = os.path.dirname(os.path.relpath(__file__))
sys.path.append(PATH)
from tools import clear_folder, generate_train_test_input_images, generate_one_hot_vectors


from enum import Enum
class DTYPES(Enum):
	TRAINING = 1
	TESTING = 2
	VALIDATION = 3

Datasets = collections.namedtuple('Datasets', ['training', 'testing','validation'])
class Dataset():
	def __init__(self,images, dType):
		self.instances = list(images.keys())
		self.num_of_images = len(self.instances)
		print('num_of_images',self.num_of_images ,dType)
		if dType == DTYPES.TRAINING:
			self.images_path = os.path.join(PATH,'images/input/cars_train')
		elif dType in [DTYPES.TESTING,DTYPES.VALIDATION]:
			self.images_path = os.path.join(PATH,'images/input/cars_test')
		self.type = dType
		self.ground_truth = images
		self.index = 0

	def next_batch(self, batch_size=50):
		if batch_size > self.num_of_images:
			raise ValueError("Dataset error...batch size is greater than the number of samples")

		start = self.index
		self.index += batch_size

		if self.index > self.num_of_images:
			# print('shuffling dataset -',self.type)
			# print('next_batch start', start,'index',self.index)
			np.random.shuffle(self.instances)
			# Shuffle the data
			self.index = batch_size
			start = 0

		end = self.index
		batch_instances = self.instances[start:end]
		imagesBatch = []
		labelsBatch = []
		namesBatch = []
		for img in batch_instances:
			try:
				img_path = os.path.join(self.images_path,img)
				image = cv2.imread(img_path)
				image = (image-128)/128 #normalizing image
				label = self.ground_truth.get(img)
			except:
				traceback.print_exc()
				continue
			imagesBatch.append(image)
			labelsBatch.append(label)
			namesBatch.append(img)
		return np.array(imagesBatch),np.array(labelsBatch),namesBatch

class DataHandler:
	__TOTAL_NUMBER_OF_TRAIN_IMAGES = 8144
	__TOTAL_NUMBER_OF_TEST_IMAGES = 8041
	__TRAIN_IMAGES_URL = "http://imagenet.stanford.edu/internal/car196/cars_train.tgz"
	__TEST_IMAGES_URL = "http://imagenet.stanford.edu/internal/car196/cars_test.tgz"
	def __init__(self):
		self.train_tarfile_path= os.path.join(PATH,'cars_train.tgz')
		self.test_tarfile_path= os.path.join(PATH,'cars_test.tgz')

	def build_datasets(self):
		images_path = os.path.join(PATH,'images')
		original_images_path = os.path.join(images_path,'original')
		train_images_path = os.path.join(original_images_path ,'cars_train')
		test_images_path = os.path.join(original_images_path ,'cars_test')
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
				is_train_data_ready = self.__extract_tarfile(self.train_tarfile_path,original_images_path )  
				is_test_data_ready = self.__extract_tarfile(self.test_tarfile_path,original_images_path )  
				if is_train_data_ready and is_test_data_ready :
					print('Extraction completed')
					data_ready = True
				else:
					print('Extraction incompleted')
		if data_ready:
			generate_train_test_input_images()
			one_hot_vectors = generate_one_hot_vectors()
			train_set = Dataset(one_hot_vectors['train'],DTYPES.TRAINING)
			test_set = Dataset(one_hot_vectors['test'],DTYPES.TESTING)
			validation_set = Dataset(one_hot_vectors['validation'],DTYPES.VALIDATION)
			return Datasets(training=train_set,testing=test_set,validation=validation_set)
 
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
				wget.download(DataHandler.__TRAIN_IMAGES_URL,out=PATH)
			if not os.path.exists(self.test_tarfile_path):
				print('Downloading test images')
				wget.download(DataHandler.__TEST_IMAGES_URL,out=PATH)
		return os.path.exists(self.train_tarfile_path) and os.path.exists(self.test_tarfile_path)

if __name__ == '__main__':
	dsets = DataHandler().build_datasets()

