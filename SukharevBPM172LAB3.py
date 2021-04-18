import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Node:
    def __init__(self, id_):
        self.id_ = id_
        self.k = 3
        self.hash_table = []
        self.files = set()
        
    def findBucket(self, key):
        if key in self.files or self.id_ == key:
            return 1, []
        
        i = self.findBucketIndex(key)
        bucket = []
        d = self.id_ ^ key
        while i >= 0 and len(bucket) < self.k:
            for node in self.hash_table[i]:
                if key^node[-1] < d:
                    bucket.append((key^node[-1], node[0]))
            i -= 1
        bucket.sort()
        while len(bucket) > self.k:
            bucket.pop()
        return 0, bucket
        
    def findBucketIndex(self, key):
        x = self.id_ ^ key
        index = 0
        cur = 1
        while cur * 2 <= x:
            index += 1
            cur *= 2
        return index

class Network:
    def __init__(self, n):
        self.n = n
        self.k = 3
        self.max_id = self.n*self.n
        self.hash_size = 1
        cur_val = 1
        while cur_val < self.max_id:
            cur_val *= 2
            self.hash_size += 1
        self.addNodes()
        self.createGraph()
        self.addFiles()
        
    def addNodes(self):
        self.nodes = []
        self.node_ids = set()
        self.node_id_map = {}
        for i_node in range(self.n):
            id_ = random.randint(1, self.max_id)
            while id_ in self.node_ids:
                id_ = random.randint(1, self.max_id)
            self.nodes.append(Node(id_))
            self.node_ids.add(id_)
            self.node_id_map[id_] = i_node
          
    def addFiles(self):
        self.file_ids = set()
        for i_file in range(3*self.n):
            file_id = random.randint(1, self.max_id-1)
            while file_id in self.node_ids or file_id in self.file_ids:
                file_id = random.randint(1, self.max_id-1)
            self.file_ids.add(file_id)
            node_id = random.randint(1, self.n-1)
            self.nodes[node_id].files.add(file_id)
            
            dist_node = [(file_id^self.nodes[i_node].id_, i_node) for i_node in range(len(self.nodes))]
            dist_node.sort()
            for i_node in range(2):
                self.nodes[dist_node[i_node][-1]].files.add(file_id)
            
    def createGraph(self):
        self.graph = [[] for i in range(self.n)]
        for i_node in range(self.n):
            self.nodes[i_node].hash_table = [[] for i in range(self.hash_size + 1)]
            id_ = self.nodes[i_node].id_
            node_ids = [(id_^self.nodes[i].id_, i) for i in range(self.n)]
            node_ids.sort()
            ind = 0
            d = 2
            i_bucket = 0
            while d <= (1 << self.hash_size) and ind < self.n:
                cur_nodes = []
                while ind + 1 < self.n and node_ids[ind + 1][0] < d:
                    ind += 1
                    cur_nodes.append(node_ids[ind][-1])
                if len(cur_nodes) > 0:
                    to = cur_nodes[random.randint(0, len(cur_nodes)-1)]
                    self.nodes[i_node].hash_table[i_bucket].append((to, self.nodes[to].id_))
                    self.graph[i_node].append(to)
                d *= 2
                i_bucket += 1
                
    def getDegrees(self):
        min_d = self.n
        max_d = 0
        avg_d = 0
        for node in self.nodes:
            d = 0
            for bucket in node.hash_table:
                d += len(bucket)
            min_d = min(min_d, d)
            max_d = max(max_d, d)
            avg_d += d
        avg_d /= self.n
        return min_d, max_d, avg_d

    def getFiles(self):
        min_num = self.n*self.n
        max_num = 0
        avg_num = 0
        for node in self.nodes:
            num = len(node.files)
            min_num = min(min_num, num)
            max_num = max(max_num, num)
            avg_num += num
        avg_num /= len(self.nodes)
        return min_num, max_num, avg_num
                
    def findValue(self, node, key):
        min_buck_sz = self.n
        max_buck_sz = 0
        avg_buck_sz = 0
        num = 0
        path = [node]
        self.used = set()
        self.used.add(node)
        res, bucket = self.nodes[node].findBucket(key)
        if len(bucket) == 0:
            return res, path, 0, 0, 0
        best_dist = bucket[0][0]+1
        while len(bucket) > 0:
            if bucket[0][0] > best_dist:
                return 0, path, min_buck_sz, max_buck_sz, avg_buck_sz / num
            else:
                best_dist = bucket[0][0]
            cur_node = bucket[0][-1]
            num += 1
            path.append(cur_node)
            self.used.add(cur_node)
            res, new_bucket = self.nodes[cur_node].findBucket(key)
            if res == 1:
                return res, path, min_buck_sz, max_buck_sz, avg_buck_sz / num
            sz = len(new_bucket)
            min_buck_sz = min(min_buck_sz, sz)
            max_buck_sz = max(max_buck_sz, sz)
            avg_buck_sz += sz
            bucket_set = set()
            for item in bucket:
                if not item[-1] in self.used:
                    bucket_set.add(item)
            for item in new_bucket:
                if not item[-1] in self.used:
                    bucket_set.add(item)
            bucket =  sorted(bucket_set)
            while len(bucket) > self.k:
                bucket.pop()
                
        return 0, path, min_buck_sz, max_buck_sz, avg_buck_sz / num
          
    def showPath(self, path, file_key):
        plt.figure(figsize=(30, 20))
        graph = nx.DiGraph()
        for fr in range(self.n):
            for to in self.graph[fr]:
                graph.add_edge(self.nodes[fr].id_, self.nodes[to].id_)
                
        node_map = [0 for i in range(self.n)]
        ind = 0
        for node in graph.nodes:
            node_map[self.node_id_map[node]] = ind
            ind += 1
        node_colors=['green' for i in range(graph.number_of_nodes())]
        for i in range(self.n):
            if file_key in self.nodes[i].files:
                node_colors[node_map[i]]='yellow'
        pos = nx.kamada_kawai_layout(graph)
        f = plt.figure()
        
        prev_node = -1
        for i in range(len(path)):
            node = path[i]
            node_colors[node_map[node]]='blue'
            nx.draw(graph, with_labels=True, pos=pos, node_color=node_colors)
            img_name = 'img'+str(i)+'.png'
            plt.savefig(img_name, format='PNG')
            f.clear()
            prev_node = node

def analysis():
    n_networks = 10
    n = 1000
    min_degree = n
    max_degree = 0
    avg_degree = 0
    min_files = n*n
    max_files = 0
    avg_files = 0
    min_len = n
    max_len = 0
    avg_len = 0
    min_buck = n
    max_buck = 0
    avg_buck = 0
    for i_network in range(n_networks):
        network = Network(n)
        min_d, max_d, avg_d = network.getDegrees()
        min_degree = min(min_degree, min_d)
        max_degree = max(max_degree, max_d)
        avg_degree += avg_d
        min_f, max_f, avg_f = network.getFiles()
        min_files = min(min_files, min_f)
        max_files = max(max_files, max_f)
        avg_files += avg_f
        for i in range(n):           
            node = i
            files = list(network.file_ids)
            file = files[random.randint(0, len(files)-1)]
            res, path, min_buck_sz, max_buck_sz, avg_buck_sz = network.findValue(node, file)
            path_len = len(path)
            min_len = min(min_len, path_len)
            max_len = max(max_len, path_len)
            avg_len += path_len
            min_buck = min(min_buck, min_buck_sz)
            max_buck = max(max_buck, max_buck_sz)
            avg_buck += avg_buck_sz
    avg_degree /= n_networks
    avg_files /= n_networks
    avg_len /= n * n_networks
    avg_buck /= n * n_networks
    print('min edge degree: ', min_degree)
    print('avg edge degree: ', avg_degree)
    print('max edge degree: ', max_degree)
    print('min file number: ', min_files)
    print('avg file number: ', avg_files)
    print('max file number: ', max_files)
    print('min length: ', min_len)
    print('avg length: ', avg_len)
    print('max length: ', max_len)
    print('min msg len: ', min_buck)
    print('avg msg len: ', avg_buck)
    print('max msg len: ', max_buck)
    
def showGraph(n):
    network = Network(n)
    node = random.randint(0, n-1)
    files = list(network.file_ids)
    file = files[random.randint(0, len(files)-1)]
    res, path, min_buck_sz, max_buck_sz, avg_buck_sz = network.findValue(node, file)
    network.showPath(path, file)
    print(file)
	
	
showGraph(100)

analysis()
