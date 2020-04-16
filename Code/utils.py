import numpy as np
import torch
from tqdm import tqdm, trange
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef
from concurrent.futures import as_completed, ProcessPoolExecutor


def add_padding_idx(vec):
	if len(vec.shape) == 1:
		return np.asarray([np.sort(np.asarray(v) + 1).astype('int')
						 for v in tqdm(vec)])
	else:
		vec = np.asarray(vec) + 1
		vec = np.sort(vec, axis=-1)
		return vec.astype('int')


def np2tensor_hyper(vec, dtype):
	vec = np.asarray(vec)
	if len(vec.shape) == 1:
		return [torch.as_tensor(v, dtype=dtype) for v in vec]
	else:
		return torch.as_tensor(vec, dtype = dtype)


def walkpath2str(walk):
	return [list(map(str, w)) for w in tqdm(walk)]


def roc_auc_cuda(y_true, y_pred):
	try:
		y_true = y_true.cpu().detach().numpy().reshape((-1, 1))
		y_pred = y_pred.cpu().detach().numpy().reshape((-1, 1))
		return roc_auc_score(
			y_true, y_pred), average_precision_score(
			y_true, y_pred)
	except BaseException:
		return 0.0, 0.0


def accuracy(output, target):
	pred = output >= 0.5
	truth = target >= 0.5
	acc = torch.sum(pred.eq(truth))
	acc = float(acc) * 1.0 / (truth.shape[0] * 1.0)
	return acc


def build_hash(data):
	dict1 = set()

	for datum in data:
		# We need sort here to make sure the order is right
		datum.sort()
		dict1.add(tuple(datum))
	del data
	return dict1


def build_hash2(data):
	dict2 = set()
	for datum in tqdm(data):
		for x in datum:
			for y in datum:
				if x != y:
					dict2.add((x, y))
	return dict2


def build_hash3(data):
	dict2 = set()
	for datum in tqdm(data):
		for i in range(3):
			temp = np.copy(datum).astype('int')
			temp[i] = 0
			dict2.add(tuple(temp))

	return dict2


def parallel_build_hash(data, func, args, num, initial = None):
	import multiprocessing
	cpu_num = multiprocessing.cpu_count()
	data = np.array_split(data, cpu_num * 3)
	dict1 = initial.copy()
	pool = ProcessPoolExecutor(max_workers=cpu_num)
	process_list = []

	if func == 'build_hash':
		func = build_hash
	if func == 'build_hash2':
		func = build_hash2
	if func == 'build_hash3':
		func = build_hash3

	for datum in data:
		process_list.append(pool.submit(func, datum))

	for p in as_completed(process_list):
		a = p.result()
		dict1.update(a)

	pool.shutdown(wait=True)
	
	# if args.data in ['schic','ramani']:
	# 	print (num[0])
	# 	new_list_of_set = [set() for i in range(int(num[0]+1))]
	# 	for s in dict1:
	# 		try:
	# 			new_list_of_set[s[0]].add(s)
	# 		except:
	# 			print (s)
	# 			raise EOFError
	# 	dict1 = new_list_of_set
	return dict1

def generate_negative_edge(x, length):
	pos = np.random.choice(len(pos_edges), length)
	pos = pos_edges[pos]
	negative = []

	temp_num_list = np.array([0] + list(num_list))

	id_choices = np.array([[0, 1], [1, 2], [0, 2]])
	id = np.random.choice([0, 1, 2], length * neg_num, replace=True)
	id = id_choices[id]

	start_1 = temp_num_list[id[:, 0]]
	end_1 = temp_num_list[id[:, 0] + 1]

	start_2 = temp_num_list[id[:, 1]]
	end_2 = temp_num_list[id[:, 1] + 1]

	if len(num_list) == 3:
		for i in range(neg_num * length):
			temp = [
				np.random.randint(
					start_1[i],
					end_1[i]) + 1,
				np.random.randint(
					start_2[i],
					end_2[i]) + 1]
			while tuple(temp) in dict2:
				temp = [
					np.random.randint(
						start_1[i],
						end_1[i]) + 1,
					np.random.randint(
						start_2[i],
						end_2[i]) + 1]
			negative.append(temp)

	return list(pos), negative


def generate_outlier(k=20):
	inputs = []
	negs = []
	split_num = 4
	pool = ProcessPoolExecutor(max_workers=split_num)
	data = np.array_split(potential_outliers, split_num)
	dict_pair = build_hash2(np.concatenate([train_data, test]))

	process_list = []

	for datum in data:
		process_list.append(
			pool.submit(
				generate_outlier_part,
				datum,
				dict_pair,
				k))

	for p in as_completed(process_list):
		in_, ne = p.result()
		inputs.append(in_)
		negs.append(ne)
	inputs = np.concatenate(inputs, axis=0)
	negs = np.concatenate(negs, axis=0)

	index = np.arange(len(inputs))
	np.random.shuffle(index)
	inputs, negs = inputs[index], negs[index]

	pool.shutdown(wait=True)

	x = np2tensor_hyper(inputs, dtype=torch.long)
	x = pad_sequence(x, batch_first=True, padding_value=0).to(device)

	return (torch.tensor(x).to(device), torch.tensor(negs).to(device))

def pass_(x):
    return x


def generate_outlier_part(data, dict_pair, k=20):
	inputs = []
	negs = []
	
	for e in tqdm(data):
		point = int(np.where(e == 0)[0])
		start = 0 if point == 0 else int(num_list[point - 1])
		end = int(num_list[point])
		
		count = 0
		trial = 0
		while count < k:
			trial += 1
			if trial >= 100:
				break
			j = np.random.randint(start, end) + 1
			condition = [(j, n) in dict_pair for n in e]
			if np.sum(condition) > 0:
				continue
			else:
				temp = np.copy(e)
				temp[point] = j
				inputs.append(temp)
				negs.append(point)
				count += 1
	inputs, index = np.unique(inputs, axis=0, return_index=True)
	negs = np.array(negs)[index]
	return np.array(inputs), np.array(negs)


def check_outlier(model, data_):
	data, negs = data_
	bs = 1024
	num_of_batches = int(np.floor(data.shape[0] / bs)) + 1
	k = 3
	outlier_prec = torch.zeros(k).to(device)
	
	model.eval()
	with torch.no_grad():
		for i in tqdm(range(num_of_batches)):
			inputs = data[i * bs:(i + 1) * bs]
			neg = negs[i * bs:(i + 1) * bs]
			outlier = model(inputs, get_outlier=k)
			outlier_prec += (outlier.transpose(1, 0) == neg).sum(dim=1).float()
		# for kk in range(k):
		# 	outlier_prec[kk] += (outlier[:,kk].view(-1)==neg).sum().float()
		outlier_prec = outlier_prec.cumsum(dim=0)
		outlier_prec /= data.shape[0]
		for kk in range(k):
			print("outlier top %d hitting: %.5f" % (kk + 1, outlier_prec[kk]))


class Word2Vec_Skipgram_Data_Empty(object):
	"""Word2Vec model (Skipgram)."""
	
	def __init__(self):
		return
	
	def next_batch(self):
		"""Train the model."""
		
		return 0, 0, 0, 0, 0
	