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

data_extend = sc.textFile('/Users/nali/Beifei/ximalaya2015/data/tag/albumtag.txt').map(lambda x: x.split(','))\
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
            self.nodeMap[(node.id, 0)] = node
        
    def connect(self, parent_id, children_ids, depth):
        #connect children and parent
        for child_id in children_ids:
            child_node = self.nodeMap[(child_id, depth-1)]
            child_node.parent_id = parent_id
        self.nodeMap[(parent_id, depth)] = Node(parent_id, children_ids, depth)

    def find_all_children_ids(self, parent_id, depth):
        node = self.nodeMap[(parent_id, depth)]
        children = node.children_ids
        if depth > 1:
            seq = range(1, depth)
            seq.reverse()
            for i in seq:
                newChildren = []
                for child in children:
                    newChildren = newChildren + self.nodeMap[(child, i)].children_ids
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
    step_coll = step.collect()
    for each in step_coll:
        tree.connect(each[0], each[1], 1)
#    step.foreach(lambda x: tree.connect(x[0], x[1]))
    for j in range(2,11):
        step = step.map(lambda x: (move(x[0], x[1]+[x[0]]), x))\
                .filter(lambda x: x[0] != 'End')\
                .groupByKey()\
                .map(lambda x: (x[0], x[1].data))
        step_coll = step.map(lambda x: (x[0], update(x[1]))).collect()
        for each in step_coll:
            tree.connect(each[0], each[1], j)
#        step.map(lambda x: (x[0], update(x[1]))).foreach(lambda x: tree.connect(x[0], x[1]))
        step = step.map(lambda x: (x[0], flatten(x[1])))
    trees.append(tree)

for tree in trees:
    print len(tree.nodeMap.keys())

def query(album, trees, topK):
    sim = {}
    for tree in trees:
        #calculate the similarities in the 1st level
        neighbors = []
        simNeighbor = 0.8
        node = tree.nodeMap[(album, 0)] #get the node object
        parent_id = node.parent_id
        if parent_id != None:
            #get the ancestor object
            ancestor = tree.nodeMap[(parent_id, 1)]              
            #get the 1st level neighbors
            neighbors = list(set(ancestor.children_ids)-set(album))
            if len(neighbors) != 0:
                for neighbor in neighbors:
                    if neighbor in sim.keys():
                        sim[neighbor] += simNeighbor
                    else:
                        sim[neighbor] = simNeighbor
            node = ancestor
            for i in range(2,11):
                simNeighbor = 0.8**i
                neighbors = []
                subVertices = []
                parent_id = node.parent_id
                if parent_id != None:
                    #get the ancestor object 
                    ancestor = tree.nodeMap[(parent_id, i)]
                    #omit the vertex which has been detected
                    subVertices = list(set(ancestor.children_ids)-set(node.id))
                    if len(subVertices) != 0:
                        for vertex in subVertices:
                           neighbors = neighbors + tree.find_all_children_ids(vertex, i-1)
                    if len(neighbors) != 0:
                        for neighbor in neighbors:
                            if neighbor in sim.keys():
                                sim[neighbor] += simNeighbor
                            else:
                                sim[neighbor] = simNeighbor
                    node = ancestor
                else:
                    break
    if sim != {}:
        sim = list(np.array(sorted(sim.items(), key=operator.itemgetter(1), reverse=True))[:,0][:topK])
    else:
        sim = []
    return (album, sim)

for album in als:
    print query(album, trees, 10)

topSearch = als_sc.map(lambda x: query(x, trees, 10))
topSearch.map(lambda x: x[0]+","+",".join(x[1])).saveAsTextFile("/Users/nali/Beifei/ximalaya2015/code_ximalaya/code_simrank/rw/topSearch")
