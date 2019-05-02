from Network.Net import Network
import sys,os, shutil
PATH = os.path.dirname(os.path.relpath(__file__))

try:
	arg = sys.argv[1]
except:
	arg = None
topologies = ['topology_01','topology_02','topology_03']
run_eval = False
if arg in topologies:
	dataset_path_1 = os.path.join(PATH,'Dataset/dataset.pickle')
	topology_path = os.path.join(PATH,'Network/models/{}'.format(arg))
	if not os.path.exists(dataset_path_1): 
		dataset_path_2 = os.path.join(topology_path,'dataset.pickle')
		if os.path.exists(dataset_path_2):
			shutil.copyfile(dataset_path_2,dataset_path_1)
			run_eval = True
	else:
		run_eval = True
	if run_eval:
		n = Network()
		n.evaluate(arg)
	else:
		print('Missing dataset.pickle file')

else:
	print('Gotta use a valid argument, e.g, python3 evaluate.py topology\noptions =',topologies )

