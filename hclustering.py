import sys
import pandas as pd
from distance_helper import eucledian_distance
import numpy as np
from itertools import groupby

from collections import defaultdict
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class Leaf:
    def __init__(self,data):
        self.type = 'Leaf'
        self.height = 0
        self.data = ','.join([str(a) for a in data])

class Node:
    def __init__(self,height):
        self.type = 'Node'
        self.height = height
        self.nodes = []

def agglomerative(categorical_numerical_map, data):
    csdata = data.copy(deep = True) # copying original dataframe, this copy will be used for centroid selection while the original will be used for clustering
    clusters = [tuple(x) for x in csdata.to_numpy()] # converting points to tuples for distance calculations
    clusters = [tuple(j) for i, j in groupby(clusters)] #combining points that are the same the be within the same cluster to begin with
    dendrogram = defaultdict(list)
    dendrogram[0] = tuple(clusters)

    cluster_distance = {}

    cluster_node = {}
    node_height = defaultdict(list)
    for cluster in clusters:
        cluster_distance[tuple(cluster)] = 0
    import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    while len(clusters) > 1:
        distance_map = {}
        for index, cluster in enumerate(clusters):
            #edge case of last one we dont need to compute the distance of the last tuple
            if index == len(clusters) - 1:
                break
            # import pdb; pdb.set_trace()
            other_clusters = clusters[index+1:]
            flatten_points_c1 = []
            flatten(cluster, flatten_points_c1)
            flatten_points_c1 = tuple(flatten_points_c1)
            distance_from_other_cluster = []
            for c2 in other_clusters:
                flattened_points_c2 = []
                flatten(c2, flattened_points_c2)
                flattened_points_c2 = tuple(flattened_points_c2)
                distance_from_other_cluster.append(single_link_distance(categorical_numerical_map, flatten_points_c1, flattened_points_c2 ))

            smallest_distance = min(distance_from_other_cluster) #obtain the smallest distance
            smallest_distance_index = distance_from_other_cluster.index(smallest_distance) #get the index of that smallest distance
            distance_map[index] = (smallest_distance, smallest_distance_index + index + 1) #must cast cluster to tuple in order to hash

        #this contains the smallest of the smallest distance computed from all clusters
        # import pdb; pdb.set_trace()
        final_smallest_distance_index = min(distance_map, key=distance_map.get)
        final_smallest_distance_value = distance_map[final_smallest_distance_index] #this value contains the tuple of (distance, index to this cluster)
        distance = final_smallest_distance_value[0]
        # import pdb; pdb.set_trace()
        index_target_cluster = final_smallest_distance_value[1]
        target_cluster = clusters[index_target_cluster]
        current_cluster = clusters[final_smallest_distance_index]
        merged_cluster = (target_cluster, current_cluster)
        # merged_cluster = target_cluster + current_cluster
        # import pdb; pdb.set_trace()
        remove_cluster(clusters,target_cluster) #removing the one that we're gonna merge
        remove_cluster(clusters, current_cluster) #must cast it to list since it's currently a tuple
        clusters.append(merged_cluster) #add the merged cluster in
        cluster_distance[merged_cluster] = distance
        final_merged_distance = distance + cluster_distance.get(target_cluster,0)
        dendrogram[final_merged_distance].append(merged_cluster)

        # if final_merged_distance not in node_height:
        #     node = Node(final_merged_distance)
        #     for leaf in merged_cluster:
        #         LeafNode = Leaf(list(leaf))
        #         node.nodes.append(LeafNode.__dict__)
        #     node_height[final_merged_distance] = node
        # else:
        #     for leaf in merged_cluster:
        #         LeafNode = Leaf(leaf)
        #         node_height[final_merged_distance].nodes.append(LeafNode.__dict__)
        node = Node(final_merged_distance)
        for leaf in merged_cluster:
            if leaf in cluster_node:
                node.nodes.append(cluster_node[leaf].__dict__)
            else:
                LeafNode = Leaf(list(leaf))
                node.nodes.append(LeafNode.__dict__)
        node_height[final_merged_distance].append(node)
        cluster_node[merged_cluster] = node
    return node

def single_link_distance(categorical_numerical_map, cluster1,cluster2):
    smallest_distance = float('inf')
    for point in cluster1:
        for point2 in cluster2:
            distance_p1_p2 = eucledian_distance(categorical_numerical_map, point, point2)
            if distance_p1_p2 < smallest_distance:
                smallest_distance = distance_p1_p2
    return smallest_distance

def remove_cluster(clusters,target):
    ind = 0
    size = len(clusters)
    while ind != size and not np.array_equal(clusters[ind],target):
        ind += 1
    if ind != size:
        clusters.pop(ind)
    else:
        raise ValueError('target not found in cluster.')

def flatten(x, res):
    for value in x:
        if type(value[0]) == tuple:
            flatten(value,res)
        else:
            res.append(value)


def main():
    n = len(sys.argv)
    filepath = None
    k = 0  # number of clusters desired
    alpha = 0

    if n != 3:
        print("args error")
    else:
        filepath = sys.argv[1]
        k = int(sys.argv[2])
        try:
            alpha = sys.argv[3]
        except:
            pass

    fileReader = open(filepath, 'r')
    line1 = fileReader.readline().strip('\n') #get first line but strip down the \n
    fileReader.close() #close since it was open
    categorial_numerical_map = {}
    for index, value in enumerate(line1.split(',')):
        if value == '0':
            categorial_numerical_map[index] = 'categorical'
        else:
            categorial_numerical_map[index] = 'numerical'

    data = pd.read_csv(filepath)  # the points

    data = data.rename(columns={x: y for x, y in zip(data.columns, range(0, len(
        data.columns)))})  # rename columns with dimension value
    print(data)
    node = agglomerative(categorial_numerical_map,data) #dendrogram has all level

    with open('output.txt', 'w') as file:
        file.write(json.dumps(node.__dict__,indent=4, cls=NpEncoder))

    import pdb; pdb.set_trace()
    # kmeanspp(data,k)


if __name__ == '__main__':
#     res = []
#     flatten((((5, 14, 1),), ((5, 15, 1))), res
# )

    main()
