import os,sys
import numpy as np
import cv2
import collections
from Utils.utils import clear_folder
import pickle
import tarfile
import wget

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
	__TOTAL_NUMBER_OF_IMAGES = 16185 
	__IMAGES_URL = "http://imagenet.stanford.edu/internal/car196/car_ims.tgz"
	__IMAGES_METADATA_URL = "http://imagenet.stanford.edu/internal/car196/cars_annos.mat"
	def __init__(self):
		self.path = os.path.dirname(os.path.relpath(__file__))
		self.tarfile_path= self.path+'/car_ims.tgz'

	def build_datasets(self):
		images_path = self.path + '/images'
		attempt_download_and_or_extraction = False
		data_ready = False
		if os.path.exists(images_path):
			try:
				if len(os.listdir(images_path+'/input')) != __TOTAL_NUMBER_OF_IMAGES :
					clear_folder(images_path)
					attempt_download_and_or_extraction = True
				else:
					data_ready = True
			except:
				clear_folder(images_path)
				attempt_download_and_or_extraction = True
		else:
			attempt_download_and_or_extraction = True

		if attempt_download_and_or_extraction:
			tar_ready =self.__maybe_download_tarfile()
			if tar_ready:
				print('Extracting Images Into Images Folder')
				data_ready = self.__extract_tarfile()  
				if data_ready:
					print('\nImage Extraction Completed')
				else:
					print('Image Extraction Incompleted')
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
 
	def __extract_tarfile(self):
		tf = tarfile.open(self.tarfile_path)
		tf.extractall
		# """Extract the first file enclosed in a tar file as a list of words."""
		# cnt = 0
		# with tarFile(self.path+'/images.tar') as z:
			# for member in z.filelist:
				# try:
					# print('extracting',member.filename,end='\r')
					# z.extract(member,path=self.path+'/images')
				# except tarfile.error as e:
					# return False
			# return True
		
	def __maybe_download_tarfile(self):
		if os.path.exists(destination):
			return True
		else:
			wget.download(__IMAGES_URL,out=self.tarfile_path)
		return os.path.exists(self.tarfile_path)

if __name__ == '__main__':
	DataHandler()
