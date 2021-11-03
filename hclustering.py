import sys
import pandas as pd
from distance_helper import eucledian_distance
import copy
def agglomerative(categorical_numerical_map, data):
    csdata = data.copy(deep = True) # copying original dataframe, this copy will be used for centroid selection while the original will be used for clustering

    clusters = [[tuple(x)] for x in csdata.to_numpy()] # converting points to tuples for distance calculations
    dendrogram = {}
    # import pdb; pdb.set_trace()
    while len(clusters) > 1:
        distance_map = {}
        for index, point in enumerate(clusters):
            #edge case of last one we dont need to compute the distance of the last tuple
            if index == len(clusters) - 1:
                break
            # import pdb; pdb.set_trace()
            other_clusters = clusters[index+1:]
            distance_from_other_cluster = [single_link_distance(categorical_numerical_map,point,c2) for c2 in other_clusters] #compute the distance from current cluster to all other cluster
            smallest_distance = min(distance_from_other_cluster) #obtain the smallest distance
            smallest_distance_index = distance_from_other_cluster.index(smallest_distance) #get the index of that smallest distance
            distance_map[index] = (smallest_distance, smallest_distance_index + index + 1) #must cast cluster to tuple in order to hash

        #this contains the smallest of the smallest distance computed from all clusters
        # import pdb; pdb.set_trace()
        final_smallest_distance_index = min(distance_map, key=distance_map.get)
        final_smallest_distance_value = distance_map[final_smallest_distance_index] #this value contains the tuple of (distance, index to this cluster)
        distance = final_smallest_distance_value[0]
        index_target_cluster = final_smallest_distance_value[1]
        merged_cluster = clusters[final_smallest_distance_index] + clusters[index_target_cluster]
        # import pdb; pdb.set_trace()
        clusters.remove(clusters[index_target_cluster]) #removing the one that we're gonna merge
        clusters.remove(clusters[final_smallest_distance_index]) #must cast it to list since it's currently a tuple
        clusters.append(list(merged_cluster)) #add the merged cluster in
        clusters_copy = copy.deepcopy(clusters) #need to do this for memory issue
        dendrogram[distance] = clusters_copy

        # import pdb; pdb.set_trace()
            # distance_map[value] =
    return dendrogram

def single_link_distance(categorical_numerical_map, cluster1,cluster2):
    smallest_distance = float('inf')
    for point in cluster1:
        for point2 in cluster2:
            distance_p1_p2 = eucledian_distance(categorical_numerical_map, point, point2)
            if distance_p1_p2 < smallest_distance:
                smallest_distance = distance_p1_p2
    return smallest_distance



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
    dendrogram = agglomerative(categorial_numerical_map,data)
    import pdb; pdb.set_trace()
    # kmeanspp(data,k)

if __name__ == '__main__':
    main()
