import e2LSH
import pyswarms as ps
from utils import *


def optimal_lsh(params):
    """
    每个粒子都是L*d*k维的，找能达到全局最优的粒子
    :param params:
    :return:
    """
    dataSet = readData("test_data.csv")
    dim0 = len(dataSet[0])
    dim1 = k = 20
    tableSize = 20
    L = 5
    Atotal = params[0:dim1*dim0*L]
    hashTable, hashFuncs, fpRand = e2LSH.e2lsh_test(dataSet, Atotal, k, L, r=1, tableSize=20)
    c = [0 for i in range(dim0)]
    for i in range(len(dataSet)):
        temp = dataSet[i]
        c = [c[j] + temp[j] for j in range(dim0)]
    c = [c[i] / len(dataSet) for i in range(dim0)]

    cm_list = []
    # SWB(gab) SBB_gab
    SWB_gab=0
    SBB_gab=0
    for i in range(tableSize):
        node = hashTable[i]
        for fp in node.buckets:
            bucket = node.buckets[fp]
            cm = [0 for i in range(len(dataSet[0]))]
            for index in bucket:
                temp = dataSet[index]
                cm = [cm[i] + temp[i] for i in range(dim0)]
            cm = [cm[i] / len(bucket) for i in range(dim0)]
            cm_list.append(cm)
            SWB_bm=0
            for index in bucket:
                temp = dataSet[index]
                sub = [temp[i] - cm[i] for i in range(dim0)]
                SWB_bm=SWB_bm + np.inner(sub,sub)
            temp = [cm[i] - c[i] for i in range(dim0)]
            SBB_gab = SBB_gab + len(bucket) * np.inner(temp, temp)
            SWB_gab = SWB_gab + SWB_bm
    J = SBB_gab / SWB_gab

    return -J



def pso(x):
    """
    Higher-level method to do forward_prop in the whole swarm.
    :param x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search
    :return: numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [optimal_lsh(x[i]) for i in range(n_particles)]
    return np.array(j)


if __name__ == "__main__":

    C = pow(2, 32) - 5
    dataSet = readData("test_data.csv")
    k = 20
    L = 5
    tableSize = 20
    lamda = 0.5 # the scale parameter / abnormal degree
    threshold = 0.5 #

    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    dim0 = len(dataSet[0])
    dim1 = k
    trainlen = len(dataSet)
    dimensions = dim0 * dim1 * L
    optimizer = ps.single.GlobalBestPSO(n_particles=1, dimensions=dimensions, options=options)
    cost, Atotal = optimizer.optimize(pso, iters=1)
    hashTable, hashFuncGroups, fpRand = e2LSH.e2lsh_test(dataSet, Atotal, k, L, r=1, tableSize=20)
    cm_list = e2LSH.generate_cm(hashTable, dataSet, k=20)

    #
    testdataSet = readData("test_data.csv")
    testlen = len(testdataSet)
    dataSet.append(testdataSet)
    start = trainlen
    result = []
    for testindex in range(testlen):
        for hashFuncGroup in hashFuncGroups:
            query = dataSet[start + testindex]
            queryFp = e2LSH.H2(e2LSH.gen_HashVals(hashFuncGroup, query, r=1), fpRand, k, C)
            queryIndex = queryFp % tableSize
            min = float('inf')
            index = 0
            for fp in cm_list[queryIndex].list:
                if cm_list[queryIndex].list[fp].state == 1:
                    temp = abs(fp - queryFp)
                    if temp <= min:
                        min = temp
                        index = fp
            temp = cm_list[queryIndex].list[index]
            veclen = [query[i] - temp.cm[i] for i in range(len(query))]
            Az = 1 - 1 / (1 + np.exp(-lamda * ((temp.rm / np.square(np.linalg.norm(veclen)))) - 1))
            if Az > threshold:
                result.append("abnormal")
            else:
                result.append("normal")

            cm_list = e2LSH.update_cm(dataSet, hashTable, start + testindex, query, queryFp, queryIndex, cm_list)
            cm_list = e2LSH.update_buckets(cm_list, threshold=0, tablesize=20)

    query = [-2.7769, -5.6967, 5.9179, 0.37671, 1]
    indexes = e2LSH.nn_search(dataSet, query, k=20, L=5, r=1, tableSize=20)
    for index in indexes:
        print(euclideanDistance(dataSet[index], query))



