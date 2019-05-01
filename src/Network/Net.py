import tensorflow as tf
import numpy as np
import cv2
import time
from .Layers import Layer
from Dataset.DataHandler import DataHandler
from Utils import utils
import operator
import functools
import cv2
import os
import shutil
from pprint import pprint
import sys
PATH = os.path.dirname(os.path.relpath(__file__))
MODELS_PATH = os.path.join(PATH,'models')

class Network:
	def __init__(self):
		# Read Dataset
		self.dataset = None
		self.name = None

	def initialize(self,topology):
		self.x = tf.placeholder(tf.float32, shape =[None,240,240,3],name='input_images')
		self.y = tf.placeholder(tf.float32, shape = [None,196],name='labels')
		self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')
		if topology == 'topology_01':
			self.topology1()
		elif topology == 'topology_02':
			self.topology2()
		elif topology == 'topology_03':
			self.topology3()
		print('\n'+self.name)

	def topology1(self): # 5 layers, 4 conv and one fully connected
		# number of parameters = 531157
		self.name = 'topology_01'
		L1 = Layer().Convolutional([4,4,3,3],self.x)# L1.output.shape = [?,240,240,3]
		L2 = Layer().Convolutional([5,5,3,2],L1.output)# L2.output.shape = [?,120,120,2]
		L3 = Layer().Convolutional([6,6,2,4],L2.output,k_pool=None)# L3.output.shape = [?,60,60,4]
		L4 = Layer().Convolutional([7,7,4,3],L3.output) # L4.output.shape = [?,60,60,3]
		L5 = Layer().Convolutional([8,8,3,3],L4.output,k_pool=None) # L5.output.shape = [?,30,30,3]
		L_drop = Layer().Dropout(L5.output,self.keep_prob)
		L_out = Layer(act_func='sigmoid').Dense([30*30*3,196],tf.reshape(L_drop.output,[-1,30*30*3]),output=True)
		self.output = L_out.output

		# This is just for the README File
		self.layers = dict()
		self.layers.update({'L1':L1.__dict__})
		self.layers.update({'L2':L2.__dict__})
		self.layers.update({'L3':L3.__dict__})
		self.layers.update({'L4':L4.__dict__})
		self.layers.update({'L5':L5.__dict__})
		self.layers.update({'L6':L_drop.__dict__})
		self.layers.update({'L_out':L_out.__dict__})

	def topology2(self): # 5 layers, 4 conv and one fully connected
		# number of parameters = 1802261
		self.name = 'topology_02'
		L1 = Layer().Convolutional([10,10,3,3],self.x,k_pool=None) # output.shape = [?,240,240,3]
		L2 = Layer().Convolutional([5,5,3,10],L1.output)# output.shape = [?,120,120,10]
		L3 = Layer().Convolutional([6,6,10,4],L2.output,k_pool=None)# output.shape = [?,120,120,4]
		L4 = Layer().Convolutional([7,7,4,3],L3.output,k_pool=None) # output.shape = [?,120,120,3]
		L5 = Layer().Convolutional([8,8,3,3],L4.output) # output.shape = [?,60,60,3]
		L6 = Layer().Convolutional([9,9,3,2],L5.output) # output.shape = [?,30,30,2]
		L_drop = Layer().Dropout(L6.output,self.keep_prob)
		LFC = Layer().Dense([30*30*2,900],tf.reshape(L_drop.output,[-1,30*30*2]))
		L_out = Layer(act_func='sigmoid').Dense([900,196],LFC.output,output=True)
		self.output = L_out.output

		# This is just for the README File
		self.layers = dict()
		self.layers.update({'L1':L1.__dict__})
		self.layers.update({'L2':L2.__dict__})
		self.layers.update({'L3':L3.__dict__})
		self.layers.update({'L4':L4.__dict__})
		self.layers.update({'L5':L5.__dict__})
		self.layers.update({'L6':L6.__dict__})
		self.layers.update({'L7':L_drop.__dict__})
		self.layers.update({'L8':LFC.__dict__})
		self.layers.update({'L_out':L_out.__dict__})

	def topology3(self): # 8 layers, 7 conv and one fully connected, no dropout
		# number of parameters = 5645667
		self.name = 'topology_03'
		L1 = Layer().Convolutional([3,3,3,3],self.x,k_pool=None) # output.shape = [?,240,240,3]
		L2 = Layer().Convolutional([5,5,3,2],L1.output,k_pool=None)# output.shape = [?,240,240,2]
		L3 = Layer().Convolutional([3,3,2,4],L2.output,k_pool=None)# output.shape = [?,240,240,4]
		L4 = Layer().Convolutional([3,3,4,3],L3.output,k_pool=None) # output.shape = [?,240,240,3]
		L5 = Layer().Convolutional([3,3,3,3],L4.output,k_pool=None) # output.shape = [?,240,240,3]
		L6 = Layer().Convolutional([4,4,3,2],L5.output,k_pool=None) # output.shape = [?,240,240,2]
		L7 = Layer().Convolutional([4,4,2,2],L6.output,k_pool=2) # output.shape = [?,120,120,2]
		L_out = Layer(act_func='sigmoid').Dense([120*120*2,196],tf.reshape(L7.output,[-1,120*120*2]),output=True)
		self.output = L_out.output

		# This is just for the README File
		self.layers = dict()
		self.layers.update({'L1':L1.__dict__})
		self.layers.update({'L2':L2.__dict__})
		self.layers.update({'L3':L3.__dict__})
		self.layers.update({'L4':L4.__dict__})
		self.layers.update({'L5':L5.__dict__})
		self.layers.update({'L6':L6.__dict__})
		self.layers.update({'L7':L7.__dict__})
		self.layers.update({'L_out':L_out.__dict__})


	def train(self,iterations=10000,learning_rate = 1e-04):
		# reading dataset
		if self.dataset is None:
			self.dataset = DataHandler().build_datasets()

		# loss function
		loss = tf.losses.softmax_cross_entropy(self.y,self.output)
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
		completed_iterations = tf.Variable(0, trainable=False, name='completed_iterations')

		# accuracy
		prediction = tf.nn.softmax(self.output)
		correct_prediction = tf.math.equal(tf.math.argmax(prediction, axis=0), tf.math.argmax(self.y, axis=0))
		accuracy = tf.math.reduce_mean(tf.dtypes.cast(correct_prediction, tf.float32),name='accuracy')

		# Creating session and initilizing variables
		init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		topology_path = os.path.join(MODELS_PATH,self.name)
		model_path = os.path.join(topology_path,'model')
		with tf.Session() as sess:
			model_stored = utils.is_model_stored(topology_path)
			if model_stored:
				print('Restoring Graph')
				saver.restore(sess,model_path)
			else:
				sess.run(init)

			lstt = tf.trainable_variables()
			acum = 0
			for lt in lstt:
				ta = lt.get_shape()
				lstd = ta.as_list()
				mult = functools.reduce(operator.mul, lstd, 1)
				acum = acum + mult
			print('Number of parameters',acum) # number of trainable parameters

			comp_iters = sess.run(completed_iterations)
			utils.create_folder(topology_path,clear_if_exists = not (comp_iters >0)) # clear Model folder if training has never taken place
			remaining_iterations = iterations - comp_iters
			print('Remaining Iterations:', remaining_iterations, '- Completed Iterations: ',comp_iters)
			init_time = time.time()
			last_saved_time = time.time()
			readme_path = os.path.join(topology_path,'README.txt')
			with open(readme_path,'w') as f:
				f.write('Network Topology:\n')
				for k,v in self.layers.items():
					f.write(k + " : " + str(v) + '\n')

				msg = "\nNumber of parameters = {}\nNumber of iterations = {}\nLearning rate = {}\n".format(acum,(comp_iters + remaining_iterations),learning_rate)
				f.write(msg)

			for i in range(remaining_iterations):
				start = time.time()
				batch = self.dataset.training.next_batch()
				normBatch = batch[0] 
				labelBatch = batch[1]
				# labelBatch = batch[1]
				# pred = prediction.eval(feed_dict={self.x:normBatch, self.y:labelBatch, self.keep_prob:1.0})
				train_step.run(feed_dict={self.x:normBatch,self.y:labelBatch, self.keep_prob:0.5})
				if i%100==0 or i==remaining_iterations-1:
					loss_value = loss.eval(feed_dict={self.x:normBatch, self.y:labelBatch, self.keep_prob:1.0})
					print("iter {}, mean square error {}, step duration -> {:.2f} secs, time since last saved -> {:.2f} secs".format(i, loss_value,(time.time()-start),time.time()-last_saved_time))
					update= comp_iters + i+1
					print('updating completed iterations:',sess.run(completed_iterations.assign(update)))

					save_path = saver.save(sess,model_path)
					print("Model saved in file: %s" % save_path)

					batch = self.dataset.validation.next_batch()
					# normBatch = np.array([(img-128)/128 for img in batch[0]])
					normBatch = batch[0] 
					# labelBatch = [lbl for lbl in batch[1]]
					labelBatch = batch[1]
					# pre = prediction.eval(feed_dict={self.x:normBatch,self.keep_prob:1.0})
					# print('prediction shape', pre.shape)
					# r = np.argmax(pre[0])
					# l = np.argmax(labelBatch[0])
					# print('prediction',r,'label',l)
					print('Validation accuracy',sess.run(accuracy ,feed_dict={self.x:normBatch, self.y:labelBatch, self.keep_prob:1.0}))
					last_saved_time = time.time()
			frozen_model_path = os.path.join(topology_path,'frozen/model.pb')
			if remaining_iterations > 0 or not os.path.exists(frozen_model_path):
				print('freezing graph')
				self.freeze_graph_model(sess)
			else:
				print('Nothing to be done')
			print('total time -> {:.2f} secs'.format(time.time()-init_time))
		try:
			tf.reset_default_graph()
		except:
			pass

	def evaluate(self,topology=None):
		if topology is None:
			topology = self.name

		if self.dataset is None:
			self.dataset = DataHandler().build_datasets()

		topology_path = os.path.join(MODELS_PATH,topology)
		if not utils.is_model_stored(topology_path ):
			print("No model stored to be restored.")
			return
		print('Evaluating',topology)
		try:
			tf.reset_default_graph()
		except:
			pass
		model_meta_path = os.path.join(topology_path ,'model.meta') 
		saver = tf.train.import_meta_graph(model_meta_path)
		g = tf.get_default_graph()
		x = g.get_tensor_by_name("input_images:0")
		y = g.get_tensor_by_name("labels:0")
		keep_prob = g.get_tensor_by_name("keep_prob:0")
		output= g.get_tensor_by_name("cars:0")
		accuracy_op =g.get_tensor_by_name("accuracy:0") 
		with tf.Session() as sess:
			model_path = os.path.join(topology_path ,'model')
			saver.restore(sess,model_path)
			print("Model restored.")
			# Evaluating testing set
			accuracies= []
			epoch = 0
			completed_batches = 0
			while epoch < 1: # run testing in one epoch
				batch_size = 50
				completed_batches += batch_size
				batch = self.dataset.testing.next_batch(batch_size)
				epoch = completed_batches/self.dataset.testing.num_of_images
				testImages = batch[0] 
				testLabels = batch[1]
				accuracy = sess.run(accuracy_op ,feed_dict={x:testImages,y: testLabels,keep_prob:1.0})
				print('{}/{} completed batches - Testing set accuracy: {:.2f}%'.format(completed_batches,self.dataset.testing.num_of_images,accuracy*100))
				accuracies.append(accuracy)
			accuracy = np.mean(accuracies)
			eval_metrics = '\nAccuracy (Testing set): {:.2f}%'.format(accuracy)
			print(eval_metrics)
			readme_path = os.path.join(topology_path ,'README.txt')
			with open(readme_path,'a') as f:
				f.write(eval_metrics)
		try:
			tf.reset_default_graph()
		except :
			pass

		shutil.copyfile(os.path.join(PATH,'../Dataset/dataset.pickle'),os.path.join(topology_path,'dataset.pickle'))

	def freeze_graph_model(self, session = None, g = None , topology = None):
		if topology is None:
			if self.name is not None:
				topology = self.name
			else:
				print('no topology was chosen')
				return

		topology_path = os.path.join(MODELS_PATH,topology)
		if not utils.is_model_stored(topology_path ):
			print("No model stored to be restored.")
			return
		try:
			tf.reset_default_graph()
		except:
			pass
		if g is None:
			g = tf.get_default_graph()

		if session is None:
			session = tf.Session()
			saver = tf.train.import_meta_graph(os.path.join(topology_path ,'model.meta'))
			saver.restore(session,os.path.join(topology_path ,'model'))

		graph_def_original = g.as_graph_def();
		# freezing model = converting variables to constants
		graph_def_simplified = tf.graph_util.convert_variables_to_constants(
				sess = session,
				input_graph_def = graph_def_original,
				output_node_names =['input_images','keep_prob','cars'])
		#saving frozen graph to disk
		output_folder = utils.create_folder(os.path.join(topology_path ,'frozen'))
		if output_folder is not None:
			model_path = tf.train.write_graph(
					graph_or_graph_def = graph_def_simplified,
					logdir = output_folder,
					name = 'model.pb',
					as_text=False)
			print("Model saved in file: %s" % model_path)
		else:
			print('Output folder could not be created')
