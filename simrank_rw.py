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

data_extend = sc.textFile('/Users/nali/Beifei/ximalaya2015/data/tag/albumtag100.txt').map(lambda x: x.split(','))\
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
    def __init__(self, id, children_ids, depth):
        self.id = id
        self.children_ids = children_ids
        self.depth = depth
        self.parent_id = None

class Tree:
    def __init__(self, albums): 
        #Initialize depth 0
        self.nodes = map(lambda x: Node(x, [], 0), albums)
        self.nodeMap = {}
        for node in self.nodes:
            self.nodeMap[node.id] = node
        
    def connect(self, parent_id, children_ids):
        #connect children and parent
        for child_id in children_ids:
            child_node = self.nodeMap[child_id]
            child_node.parent_id = parent_id
        child_depth = self.nodeMap[children_ids[0]].depth
        parent_node = self.nodeMap[parent_id]
        parent_node.depth = child_depth + 1
        parent_node.children_ids +=  children_ids

    def find_all_chilren_ids(self, parent_id):
        node = self.nodeMap[parent_id]
        children = node.children_ids
        seq = range(node.depth)
        seq.reverse()
        for i in seq:
            newChildren = []
            for child in children:
                newChildren = newChildren + self.nodeMap[child].children_ids
            children = newChildren
        return children

def move(inputNode, childrenAll):
    tags = al_tag_dict[inputNode]
    tag = random.sample(tags, 1)[0]
    ancestorsAll = tag_al_dict[tag]
    ancestors = list(set(ancestorsAll) - set(childrenAll))
    if len(ancestors) == 0:
        return "End"
    else:
        ancestor = random.sample(ancestors, 1)[0]
        return ancestor

def update(x):
    subNodes = []
    for each in x:
        subNodes.append(each[0])
    return subNodes

def printx(x):
    print x

trees = []
for i in range(10):
    tree = Tree(als)
    step = als_sc.map(lambda x: (move(x, []), x))\
            .filter(lambda x: x[0] != 'End')\
            .groupByKey()\
            .map(lambda x: (x[0], x[1].data))

    #        .map(lambda x: (x[0], flatten(x[1].data)))\
    #        .filter(lambda x: len(x) == 2 & len(x[1]) != 0)\

    step.foreach(lambda x: tree.connect(x[0], x[1]))

    for i in range(1,10):
        step = step.map(lambda x: (move(x[0], x[1]+[x[0]]), x))\
                .filter(lambda x: x[0] != 'End')\
                .groupByKey()\
                .map(lambda x: (x[0], x[1].data))
        step.map(lambda x: (x[0], update(x[1]))).foreach(lambda x: tree.connect(x[0], x[1]))
        step = step.map(lambda x: (x[0], flatten(x[1])))
    trees.append(tree)

def query(album, trees, topK):
    sim = {}
    for tree in trees:
        neighbors = []
        simNeighbor = 0.8
        nodeMap = tree.nodeMap
        node = nodeMap[album]
        parent_id = node.parent_id
        if parent_id != None:
            ancestor = nodeMap[parent_id]
            neighbors = list(set(ancestor.children)-set(album))
            if len(neighbors) != 0:
                for neighbor in neighbors:
                    if neighbor in sim.keys():
                        sim[neighbor] = sim[neighbor] + simNeighbor
                    else:
                        sim[neighbor] = simNeighbor
            node = ancestor
            for i in range(1, 10):
                simNeighbor = 0.8*(i+1)
                neighbors = []
                subVertices = []
                parent_id = node.parent_id
                if parent_id != None:
                    ancestor = nodeMap[parent_id]
                    subVertices = list(set(ancestor.children_ids)-set(node.id))
                    for vertex in subVertices:
                       neighbors = neighbors + tree.find_all_children_ids(vertex)
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
    return (album, sim)

topSearch = als_sc.map(lambda x: query(x, trees, 10))
topSearch.map(lambda x: x[0]+","+",".join(x[1])).saveAsTextFile("/Users/nali/Beifei/ximalaya2015/code_ximalaya/code_simrank/rw/topSearch100")

sys.exit()

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
