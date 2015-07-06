import codecs
import itertools
from pyspark import SparkContext, SparkConf
import time
import numpy as np
import sys
from operator import add
import operator
from compiler.ast import flatten
import random

conf = SparkConf()
sc = SparkContext(appName = "tag")

data_extend = sc.textFile('/Users/nali/Beifei/ximalaya2015/data/tag/albumtag5000.txt').map(lambda x: x.split(','))\
       .filter(lambda x: len(x) > 1)\
       .map(lambda x: map(lambda a: (x[0], a), x[1:]))\
       .flatMap(lambda x: x)\
       .filter(lambda x: len(x) == 2)\
       .map(lambda x: (x[0], x[1]))

als_sc = data_extend.map(lambda x: x[0]).distinct()
als = als_sc.collect()

al_tag_dict =  dict(data_extend.groupByKey().map(lambda x: (x[0], flatten(x[1].data))).collect())
tag_al_dict =  dict(data_extend.map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: (x[0], flatten(x[1].data))).collect())

class Node:
    def __init__(self, id, children, depth):
        self.id = id
        self.children = children
        self.depth = depth

def move(inputNode, inputDict, childrenAll):
    ancestorsAll = inputDict[inputNode]
    ancestors = list(set(ancestorsAll) - set(childrenAll))
    if len(ancestors) == 0:
        return "End"
    else:
        ancestor = random.sample(ancestors, 1)[0]
        return ancestor

def update(x):
    subNodes = []
    lastNodes = []
    thisNodes = []
    for i in range(len(x)):
        subNodes.append(x[i][0])
        lastNodes = lastNodes + x[i][3]
        thisNodes = thisNodes + x[i][2]
    return [subNodes, lastNodes, thisNodes]

def constructTree():
    iterations = []
    step = als_sc.map(lambda x: (move(x, al_tag_dict, []),x))\
            .filter(lambda x: x != "End")\
            .groupByKey()\
            .map(lambda x: (x[0], flatten(x[1].data), flatten(x[1].data), [x[0]]))
    tmp = step.collect()
    nodes = []
    for i in range(len(tmp)):
        node = Node(tmp[i][0], tmp[i][1], 0)
        nodes.append(node)
    iterations.append(nodes)

    for i in range(1,10):
        if i%2 != 0:
            inputDict = tag_al_dict
        else:
            inputDict = al_tag_dict
        step = step.map(lambda x: (move(x[0], inputDict,x[2]), x))\
                .filter(lambda x: x[0] != "End")\
                .groupByKey()\
                .map(lambda x: (x[0], x[1].data))\
                .map(lambda x: (x[0], update(x[1])))\
                .map(lambda x: (x[0], x[1][0], x[1][1], x[1][2] + [x[0]]))

        tmp = step.collect()
        nodes = []
        for j in range(len(tmp)):
            node = Node(tmp[j][0], tmp[j][1], i)
            nodes.append(node)
        iterations.append(nodes)
    return iterations

fingerPrints = []
for i in range(100):
    fingerPrints.append(constructTree())

def query(x, fingerprint, topK):
    def getNodeClass(node, tree, depth):
        allNodes = tree[depth]
        for eachNode in allNodes:
            if eachNode.id == node:
                break
        return eachNode
    def searchForVertices(node, tree, depth):
        nodeClass = getNodeClass(node, tree, depth)
        children = nodeClass.children
        seq = range(depth)
        seq.reverse()
        for i in seq:
            newChildren = []
            for child in children:
                newChildren = newChildren + getNodeClass(child, tree, i).children
            children = newChildren
        return children

    def searchForAncestor(node, tree, depth):
        nextNodes = tree[depth]
        ancestor = "End"
        for eachNextNode in nextNodes:
            if node in eachNextNode.children:
                ancestor = eachNextNode
                break
        return ancestor
    
    sim = {}
    for i in range(100):
        iteration = fingerprint[i]
        subVertices = []
        neighbors = []
        simNeighbor = 0.8
        ancestor = searchForAncestor(x, iteration, 0)
        neighbors = list(set(ancestor.children)-set(x))
        if len(neighbors) != 0:
            for neighbor in neighbors:
                if neighbor in sim.keys():
                    sim[neighbor] = sim[neighbor] + simNeighbor
                else:
                    sim[neighbor] = simNeighbor
        node = ancestor
        for i in range(1, 10):
            simNeighbor = 0.8**(i+1)
            neighbors = []
            subVertices = []
            ancestor = searchForAncestor(node.id, iteration, i)
            if ancestor != "End":
                subVertices = list(set(ancestor.children)-set(node.id))
                for vertex in subVertices:
                    neighbors = neighbors + searchForVertices(vertex, iteration, i)
                if len(neighbors) != 0:
                    for neighbor in neighbors:
                        if neighbor in sim.keys():
                            sim[neighbor] = sim[neighbor] + simNeighbor
                        else:
                            sim[neighbor] = simNeighbor
                node = ancestor
            else:
                break
    if sim != {}:
        sim = list(np.array(sorted(sim.items(), key=operator.itemgetter(1)))[:,0][:topK])
    else:
        sim = []
    return (x, sim)

topSearch = als_sc.map(lambda x: query(x, fingerPrints, 10))
topSearch.map(lambda x: x[0]+","+",".join(x[1])).saveAsTextFile("/Users/nali/Beifei/ximalaya2015/code_ximalaya/code_simrank/topSearch5000")
