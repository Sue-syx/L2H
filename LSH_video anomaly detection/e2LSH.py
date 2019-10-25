import random
import numpy as np
from utils import euclideanDistance


class TableNode(object):
    def __init__(self, index):
        self.val = index
        self.buckets = {}

class BStates(object):
	def __init__(self, cm, fp, rm, count=0, state=1):
		self.val = fp
		self.cm = cm
		self.rm = rm
		self.count = count
		self.state = state


class cmlist(object):
	def __init__(self, index):
		self.val = index
		self.list = {}


# 随机生成一组 a,b
def genPara(n, r):
    """
    :param n: length of data vector
    :param r:
    :return: a, b
    """
    a = []
    for i in range(n):
        a.append(random.gauss(0, 1))
    b = random.uniform(0, r)
    return a, b


# 随机生成 k 个 hash func
def gen_e2LSH_family(n, k, r):
    """
    :param n: length of data vector
    :param k:
    :param r:
    :return: a list of parameters (a, b)
    """
    hashfunc = []
    for i in range(k):
        hashfunc.append(genPara(n, r))
    return hashfunc


# 生成 k 个 hash结果
def gen_HashVals(e2LSH_family, v, r):
    """
    :param e2LSH_family: include k hash funcs(parameters)
    :param v: data vector
    :param r:
    :return hash values: a list
    """
    # hashVals include k values
    hashVals = []
    for hab in e2LSH_family:
        hashVal = (np.inner(hab[0], v) + hab[1]) // r
        hashVals.append(hashVal)
    return hashVals


def H2(hashVals, fpRand, k, C):
    """
    :param hashVals: k hash vals
    :param fpRand: ri', the random vals that used to generate fingerprint
    :param k, C: parameter
    :return: the fingerprint of (x1, x2, ..., xk), a int value
    """
    return int(sum([(hashVals[i] * fpRand[i]) for i in range(k)]) % C)


def e2LSH(dataSet, k, L, r, tableSize):
    """
    generate hash table

    * hash table: a list, [node1, node2, ... node_{tableSize - 1}]
    ** node: node.val = index; node.buckets = {}
    *** node.buckets: a dictionary, {fp:[v1, ..], ...}

    :param dataSet: a set of vector(list)
    :param k:
    :param L:
    :param r:
    :param tableSize:
    :return: 3 elements, hash table, hash functions, fpRand
    """

    hashTable = [TableNode(i) for i in range(tableSize)]

    n = len(dataSet[0])
    m = len(dataSet)

    C = pow(2, 32) - 5
    hashFuncs = []
    fpRand = [random.randint(-10, 10) for i in range(k)]

    for i in range(L):

        e2LSH_family = gen_e2LSH_family(n, k, r)

        # hashFuncs: [[h1, ...hk], [h1, ..hk], ..., [h1, ...hk]]
        # hashFuncs include L hash functions group, and each group contain k hash functions
        hashFuncs.append(e2LSH_family)

        for dataIndex in range(m):

            # generate k hash values
            hashVals = gen_HashVals(e2LSH_family, dataSet[dataIndex], r)

            # generate fingerprint
            fp = H2(hashVals, fpRand, k, C)

            # generate index
            index = fp % tableSize

            # find the node of hash table
            node = hashTable[index]

            # node.buckets is a dictionary: {fp: vector_list}
            if fp in node.buckets:

                # bucket is vector list
                bucket = node.buckets[fp]

                # add the data index into bucket
                bucket.append(dataIndex)

            else:
                node.buckets[fp] = [dataIndex]

    return hashTable, hashFuncs, fpRand


def e2lsh_test(dataSet, Atotal, k, L, r, tableSize):
	"""

	:param dataSet:
	:param Atotal:
	:param k:
	:param L:
	:param r:
	:param tableSize:
	:return:
	"""
	hashTable = [TableNode(i) for i in range(tableSize)]
	dim0 = len(dataSet[0])
	dim1 = k
	m = len(dataSet)
	C = pow(2, 32) - 5
	hashFuncs = []
	Alength = dim0 * dim1
	fpRand = [random.randint(-10, 10) for i in range(k)]

	for i in range(L):
		# Atotal L个A
		A = [Atotal[jndex] for jndex in range(Alength*i, Alength*(i+1))]
		A = np.array(A)
		A = A.reshape(dim0, dim1)

		e2LSH_family = []
		for i in range(k):
			b = random.uniform(0, r)
			e2LSH_family.append((A[:, i], b))

		# hashFuncs: [[h1, ...hk], [h1, ..hk], ..., [h1, ...hk]]
		# hashFuncs include L hash functions group, and each group contain k hash functions
		hashFuncs.append(e2LSH_family)

		for dataIndex in range(m):

			# generate k hash values
			hashVals = gen_HashVals(e2LSH_family, dataSet[dataIndex], r)

			# generate fingerprint
			fp = H2(hashVals, fpRand, k, C)

			# generate index
			index = fp % tableSize

			# find the node of hash table
			node = hashTable[index]

			# node.buckets is a dictionary: {fp: vector_list}
			if fp in node.buckets:

				# bucket is vector list
				bucket = node.buckets[fp]

				# add the data index into bucket
				bucket.append(dataIndex)

			else:
				node.buckets[fp] = [dataIndex]

	return hashTable, hashFuncs, fpRand


def nn_search(dataSet, query, k, L, r, tableSize):
    """
    :param dataSet:
    :param query:
    :param k:
    :param L:
    :param r:
    :param tableSize:
    :return: the data index that similar with query
    """

    result = set()

    hashTable, hashFuncGroups, fpRand = e2LSH(dataSet, k, L, r, tableSize)
    C = pow(2, 32) - 5
    for hashFuncGroup in hashFuncGroups:

        # get the fingerprint of query
        queryFp = H2(gen_HashVals(hashFuncGroup, query, r), fpRand, k, C)

        # get the index of query in hash table
        queryIndex = queryFp % tableSize

        # get the bucket in the dictionary
        if queryFp in hashTable[queryIndex].buckets:
            result.update(hashTable[queryIndex].buckets[queryFp])

    return result


def generate_cm(hashTable, dataSet, k=20):
	"""

	:param hashTable:
	:param dataSet:
	:param k:
	:return:
	"""
	dim0 = len(dataSet[0])
	cm_list = [cmlist(i) for i in range(k)]
	for i in range(k):
		node = hashTable[i]
		for fp in node.buckets:
			bucket = node.buckets[fp]
			cm = [0 for i in range(dim0)]
			for index in bucket:
				temp = dataSet[index]
				cm = [cm[i] + temp[i] for i in range(dim0)]

			cm = [cm[i] / len(bucket) for i in range(dim0)]
			max = 0
			for index in bucket:
				temp = dataSet[index]
				delta = [temp[i] - cm[i] for i in range(dim0)]
				dist = np.square(np.linalg.norm(delta))
				max = dist if dist >= max else max
			bucket_state_temp = BStates(cm,fp,max,len(bucket),1)
			cm_list[i].list[fp] = bucket_state_temp

	return cm_list

def update_cm(dataSet, hashTable,queryi,query,queryFp,queryIndex,cm_list):
	"""

	:param dataSet:
	:param hashTable:
	:param queryi:
	:param query:
	:param queryFp:
	:param queryIndex:
	:param cm_list:
	:return:
	"""
	dim0 = len(query)
	node = hashTable[queryIndex]
	if queryFp in node.buckets:
		bucket = node.buckets[queryFp]
		bucket.append(queryi)
	else:
		node.buckets[queryFp] = queryi

	temp = cm_list[queryIndex].list[queryFp]
	sum = [temp.cm[i] * temp.count + query[i] for i in range(dim0)]
	cm_list[queryIndex].list[queryFp].count = cm_list[queryIndex].list[queryFp].count + 1
	cm_list[queryIndex].list[queryFp].cm = [sum[i] / cm_list[queryIndex].list[queryFp].count for i in range(dim0)]
	cm = cm_list[queryIndex].list[queryFp].cm
	max = 0
	for index in bucket:
		temp = dataSet[index]
		delta = [temp[i] - cm[i] for i in range(dim0)]
		dist = np.square(np.linalg.norm(delta))
		max = dist if dist >= max else max
	cm_list[queryIndex].list[queryFp].rm = max

	return  cm_list

def update_buckets(cm_list, threshold, tablesize):
	"""

	:param cm_list:
	:param threshold:
	:param tablesize:
	:return:
	"""
	sum = 0
	for index in range(tablesize):
		node = cm_list[index]
		for fp in node.list:
			sum = sum + node.list[fp].count

	for index in range(tablesize):
		node = cm_list[index]
		for fp in node.list:
			prob = node.list[fp].count / sum
			if prob < threshold:
				cm_list[index].list[fp].state = 0
			else:
				cm_list[index].list[fp].state = 1

	return cm_list



