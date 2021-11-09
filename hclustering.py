import sys
import pandas as pd
from distance_helper import eucledian_distance
import numpy as np
from itertools import groupby

from collections import defaultdict
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D





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
        self.data = ([str(a) for a in data])

class Node:
    def __init__(self,height):
        self.type = 'Node'
        self.height = height
        self.nodes = []

def agglomerative(categorical_numerical_map, data, distance_method):
    csdata = data.copy(deep = True) # copying original dataframe, this copy will be used for centroid selection while the original will be used for clustering
    clusters = [tuple(x) for x in csdata.to_numpy()] # converting points to tuples for distance calculations
    clusters = sorted(clusters)
    clusters = [tuple(j) for i, j in groupby(clusters)] #combining points that are the same the be within the same cluster to begin with
    dendrogram = defaultdict(list)
    dendrogram[0] = tuple(clusters)
    # import pdb; pdb.set_trace()

    cluster_distance = {}

    cluster_node = {}
    node_height = defaultdict(list)
    for cluster in clusters:
        cluster_distance[tuple(cluster)] = 0
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
                if distance_method == 1:
                    distance_from_other_cluster.append(single_link_distance(categorical_numerical_map, flatten_points_c1, flattened_points_c2 ))
                elif distance_method == 2:
                    distance_from_other_cluster.append(complete_link_distance(categorical_numerical_map, flatten_points_c1, flattened_points_c2 ))
                elif distance_method == 3:
                    distance_from_other_cluster.append(average_link_distance(categorical_numerical_map, flatten_points_c1, flattened_points_c2 ))

            smallest_distance = min(distance_from_other_cluster) #obtain the smallest distance
            smallest_distance_index = distance_from_other_cluster.index(smallest_distance) #get the index of that smallest distance
            distance_map[index] = (smallest_distance, smallest_distance_index + index + 1) #must cast cluster to tuple in order to hash

        #this contains the smallest of the computed distance from all clusters
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

        node = Node(final_merged_distance)
        for leaf in merged_cluster:
            if leaf in cluster_node:
                node.nodes.append(cluster_node[leaf].__dict__)
            else:
                LeafNode = Leaf(list(leaf))
                node.nodes.append(LeafNode.__dict__)
        node_height[final_merged_distance].append(node)
        cluster_node[merged_cluster] = node
    # import pdb; pdb.set_trace()
    node.type = 'Root'
    return node

def single_link_distance(categorical_numerical_map, cluster1,cluster2):
    smallest_distance = float('inf')
    for point in cluster1:
        for point2 in cluster2:
            distance_p1_p2 = eucledian_distance(point, point2)
            if distance_p1_p2 < smallest_distance:
                smallest_distance = distance_p1_p2
    return smallest_distance

def complete_link_distance(categorical_numerical_map, cluster1,cluster2):
    biggest_distance = 0
    for point in cluster1:
        for point2 in cluster2:
            distance_p1_p2 = eucledian_distance(point, point2)
            if distance_p1_p2 > biggest_distance:
                biggest_distance = distance_p1_p2
    return biggest_distance

def average_link_distance(categorical_numerical_map, cluster1, cluster2):
    average_distance = 0
    for point in cluster1:
        for point2 in cluster2:
            average_distance += eucledian_distance(point, point2)
    return average_distance / (len(cluster1) * len(cluster2))

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

def get_alpha_cluster(alpha, root,result):
    if root['type'] == 'Leaf':
        result.append(root)
        return
    current_height = root['height']
    if current_height < alpha:
        result.append(root)
        return

    for node in root['nodes']:
        if type(node) == dict:
            get_alpha_cluster(alpha, node, result)
        else:
            get_alpha_cluster(alpha, node.__dict__,result)




def get_all_points(cluster, result):
    if cluster['type'] == 'Leaf':
        datas = cluster['data']
        for data in datas:
            result.append(eval(data))
        return

    for node in cluster['nodes']:
        get_all_points(node, result)


# print("Cluster " + str(centroids.index(c))+":")
#         print("Center: " + str(c))
#         print("Max Dist. to Center: " + str(dfp2c.groupby(['centroid']).max()[centroids.index(c)][centroids.index(c)]) )
#         print("Min Dist. to Center: " + str(dfp2c.groupby(['centroid']).min()[centroids.index(c)][centroids.index(c)]) )
#         print("Avg Dist. to Center: " + str(dfp2c.groupby(['ce
def compute_center(points):
    center = []
    for i in range(0, len(points[0])):
        all_i = [a[i] for a in points]
        center.append(sum(all_i)/len(all_i))
        # import pdb; pdb.set_trace()
    return tuple(center)

def evaluation(clusters):
    for index, cluster in enumerate(clusters):
        all_points = []
        get_all_points(cluster, all_points)
        center = compute_center(all_points)
        distance_to_center = [eucledian_distance(point,center) for point in all_points]
        max_distance = max(distance_to_center)
        min_distance = min(distance_to_center)
        average_distance = sum(distance_to_center) / len(distance_to_center)
        print('********************Cluster {0}*********************'.format(index + 1))
        print('There is {0} points in total'.format(len(all_points)))
        print('Cluster Height: ', cluster['height'])
        print('Points: ', all_points)
        print('Center: ', tuple(center))
        print('Max distance to center: ', max_distance)
        print('Min distance to center: ', min_distance)
        print('Average distance to center: ', average_distance)


def graph(clusters, dimension):
    if dimension == 3:
        fig = plt.figure()
        ax = Axes3D(fig)

    for index, cluster in enumerate(clusters):
        all_points = []
        get_all_points(cluster,all_points)
        all_points = np.array(all_points)
        if dimension == 2:
            x, y = all_points.T
            plt.scatter(x,y)
        elif dimension == 3:

            x,y,z = all_points.T
            # import pdb; pdb.set_trace()

            ax.scatter3D(z, y, x)
    # plt.show()
    plt.show()

def compacted_ratio(clusters):
    radius_clusters = []
    centroid_of_clusters = []
    centroid_distances = []

    for index, cluster in enumerate(clusters):
        all_points = []
        get_all_points(cluster, all_points)
        center = compute_center(all_points)
        distance_to_center = [eucledian_distance(point, center) for point in all_points]
        max_distance = max(distance_to_center)
        centroid_of_clusters.append(center)
        radius_clusters.append(max_distance)
    for index, centroid in enumerate(centroid_of_clusters):
        if index == len(centroid_of_clusters) - 1:
            break
        other_centroids = centroid_of_clusters[index+1:]
        distance_from_centroid_to_centroid = [eucledian_distance(centroid, c2) for c2 in other_centroids]
        centroid_distances = centroid_distances + distance_from_centroid_to_centroid
    # import pdb; pdb.set_trace()

    average_centroid_distance = sum(centroid_distances) / len(centroid_distances)
    average_radius = sum(radius_clusters) / len(radius_clusters)
    return  average_centroid_distance / average_radius

def main():
    n = len(sys.argv)
    k = 0  # number of clusters desired
    alpha = 0
    filepath = sys.argv[1]
    distance_method = int(sys.argv[2])

    try:
        alpha = float(sys.argv[3])
    except:
        alpha = 0


    fileReader = open(filepath, 'r')
    line1 = fileReader.readline().strip('\n') #get first line but strip down the \n
    fileReader.close() #close since it was open
    categorial_numerical_map = {}


    data = pd.read_csv(filepath, index_col=False)  # the points

    for index, value in enumerate(line1.split(',')):
        if value == '0':
            categorial_numerical_map[index] = 'categorical'
            data = data.drop(data.columns[index], axis=1)
        else:
            categorial_numerical_map[index] = 'numerical'
    data = data.rename(columns={x: y for x, y in zip(data.columns, range(0, len(
        data.columns)))})  # rename columns with dimension value
    # print(data)

    number_of_columns = len(data.columns)


    entire_hiearchy = agglomerative(categorial_numerical_map,data, distance_method) #dendrogram has all level
    if alpha != 0:
        alpha_cut_off_clusters = []
        get_alpha_cluster(alpha, entire_hiearchy.__dict__, alpha_cut_off_clusters)
        print('\n\nWith an threshold value of {0}, we have {1} total clusters:\n\n'.format(alpha, len(alpha_cut_off_clusters)))
        evaluation(alpha_cut_off_clusters)
        if number_of_columns == 2 or number_of_columns == 3:
            graph(alpha_cut_off_clusters, number_of_columns)
        else:
            ratio = compacted_ratio(alpha_cut_off_clusters)
            print('\n\n RATIO =', ratio)
        with open(filepath.replace('.csv', '') + '_output_alpha.txt', 'w') as file:
            file.write('//We end up with {0} clusters with alpha value of {1} \n'.format(len(alpha_cut_off_clusters),alpha))
            for index, cluster in enumerate(alpha_cut_off_clusters):
                file.write('//********************Cluster {0}*********************\n'.format(index + 1))
                file.write(json.dumps(cluster, indent=4, cls=NpEncoder) + '\n')
    else:
        print('\n\nWith an threshold value of 0, we have 1 singular cluster!\n\n')
        evaluation([entire_hiearchy.__dict__])


    # import pdb; pdb.set_trace()

    with open(filepath.replace('.csv', '') + '_output.json', 'w') as file:
        file.write(json.dumps(entire_hiearchy.__dict__,indent=4, cls=NpEncoder))

    # import pdb; pdb.set_trace()
    # kmeanspp(data,k)


if __name__ == '__main__':
#     res = []
#     flatten((((5, 14, 1),), ((5, 15, 1))), res
# )

    main()
