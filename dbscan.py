import sys
import numpy as np
import pandas as pd
from pandas.io.formats.format import return_docstring
from distance_helper import eucledian_distance

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


def coreCluster(x,data,core,distArray,clusterCount,epsilon,numpoints):
    if pd.isnull(core.loc[x.name]['cluster']):
        if core['cluster'].isnull().all():
            clusterCount = 0
        else:
            clusterCount = core['cluster'].max() + 1
        x.at['cluster'] = clusterCount
        data.at[x.name, 'cluster'] = clusterCount
        core.at[x.name, 'cluster'] = clusterCount

        x.at['visited'] = 1
        data.at[x.name,'visited'] = 1
        core.at[x.name,'visited'] = 1

        densityConnected(x,data,core,distArray,clusterCount,epsilon,numpoints)
    # print(x['cluster'])
    return

def densityConnected(x,data,core,distArray,clusterCount,epsilon,numpoints): # maybe there's a way to vectorize this? no idea tbh
    for idx in x['neighbors']:
        data.at[idx,'cluster'] = clusterCount
        # x['cluster']
        if idx in core.index:
            p = core.loc[idx]
            core.at[idx,'cluster'] = clusterCount
            p.at['cluster'] = clusterCount

            # print(p)
            if p['visited'] == 0:
                data.at[idx,'visited'] = 1
                core.at[idx,'visited'] = 1
                p.at['visited'] = 1
                densityConnected(p,data,core,distArray,clusterCount,epsilon,numpoints)

            

    return
def main():
    
    if len(sys.argv) == 4:
        filepath = sys.argv[1]
        epsilon = float(sys.argv[2])
        numpoints = int(sys.argv[3])
    else:
        print("Invalid arguments, please see README")
        return

    data = pd.read_csv(filepath, index_col=False) # the points
    r = open(filepath)
    restAttrs = r.readline().split(',')
    for ix, a in enumerate(restAttrs):
        if int(a) != 1:
            data = data.drop(data.columns[ix], axis=1)
    data = data.rename(columns={x:y for x,y in zip(data.columns,range(0,len(data.columns)))}) # rename columns with dimension value 


    distArray = data.apply(lambda r: euclideanDF(r, data), axis = 1)
    # distArray.values[[np.arange(distArray.shape[0])]*2] = -1
    # s = pd.Series(data=[1,2,3],index=['a','b','c'])
    np.fill_diagonal(distArray.values, -1)
    
    # print(distArray)

    classification = distArray[(distArray[:] <= epsilon) & (distArray[:] != -1)].count()

    # print(classification)

    classification = classification.apply(lambda r: 'c' if r >= numpoints else 'n')
    classification = pd.DataFrame({'type':classification.values})
    # print(classification)
    visited = pd.DataFrame({'visited':([0] * len(data))})
    # print(visited)


    data = pd.concat([data,classification,visited], axis = 1)
    # print(data)

    core = (data.loc[data['type'] == 'c']).copy(deep = True)
    # .drop(columns = ['type'])
    core['neighbors'] = np.empty((len(core), 0)).tolist()

    # print(distArray)

    neighbors = distArray.apply(lambda r: findNeighbors(r,epsilon))
    # print(neighbors)
    core['neighbors'] = neighbors
    core['cluster'] = np.nan
    data['cluster'] = np.nan

    clusterCount = -1
    core.apply(lambda x: coreCluster(x,data,core,distArray,clusterCount,epsilon,numpoints), axis = 1)
    
    data['cluster'] = data['cluster'].fillna(-1)

    data['type'] = data.apply(lambda x: 'b' if ((x['type'] == 'n')and(x['cluster']) != -1) else x['type'], axis = 1) 
    # classification = distArray[(distArray[:] <= epsilon) & (distArray[:] != -1)].count()
    # classification = classification.apply(lambda r: 'c' if r >= numpoints else 'n')
    # classification = pd.DataFrame({'type':classification.values})


    # print(str(data.to_string()))
    # print(str(core.to_string()))







    # points = data.drop(['visited','type'], axis = 1) 
    clusters = sorted(data['cluster'].unique())
    # print(clusters)

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
            other_centroids = centroid_of_clusters[index + 1:]
            distance_from_centroid_to_centroid = [eucledian_distance(centroid, c2) for c2 in other_centroids]
            centroid_distances = centroid_distances + distance_from_centroid_to_centroid
        # import pdb; pdb.set_trace()

        average_centroid_distance = sum(centroid_distances) / len(centroid_distances)
        average_radius = sum(radius_clusters) / len(radius_clusters)
        return average_centroid_distance / average_radius

    radius_clusters = []
    centroid_of_clusters = []
    centroid_distances = []

    for c in clusters:

        if c != -1:
            print("Cluster: " + str(int(c)))
            clusteri = data.loc[data['cluster'] == c]
            clusteri = clusteri.drop(['visited', 'type', 'cluster'], axis = 1)
            centroid = tuple(clusteri.mean())

            #added for compactness calculation
            centroid_of_clusters.append(centroid)

            print("Center: " + str(centroid))
            centroid = pd.DataFrame([centroid])
            centDists = euclideanDF2(clusteri,centroid)
            print("Max Dist. to Center: " + str(centDists.max()) )
            print("Min Dist. to Center: " + str(centDists.min()) )
            print("Avg Dist. to Center: " + str(centDists.mean()) )
            clusteri = data.loc[data['cluster'] == c]
            clusteri = clusteri.drop(['visited', 'cluster'], axis = 1)
            print(str(len(clusteri)) + " Points:")
            print(clusteri.to_string())

            #added for compact ness
            radius_clusters.append(centDists.max())

    #calculating ratio! comment this out if not needed, only used for analysis purposes
    for index, centroid in enumerate(centroid_of_clusters):
        if index == len(centroid_of_clusters) - 1:
            break
        other_centroids = centroid_of_clusters[index + 1:]
        distance_from_centroid_to_centroid = [eucledian_distance(centroid, c2) for c2 in other_centroids]
        centroid_distances = centroid_distances + distance_from_centroid_to_centroid

    import pdb;
    pdb.set_trace()

    average_centroid_distance = sum(centroid_distances) / len(centroid_distances)
    average_radius = sum(radius_clusters) / len(radius_clusters)
    ratio = average_centroid_distance / average_radius

    if clusters[0] == -1:
        clusteri = data.loc[data['cluster'] == -1]
        clusteri = clusteri.drop(['visited', 'cluster'], axis = 1)
        print( str(len(clusteri)) + " Noise Points/Outliers (" + str((round((len(clusteri)/len(data)*100),2)))+" percent of dataset):")
        print(str(clusteri.to_string()))
    else:
        print("0 Noise Points/Outliers")

    
    # print(distArray)
    # data = data.drop(['visited','type'], axis = 1)
    # # print(data)

    # data.columns = ['x','y','cluster'] 

    # # 2D
    # groups = data.groupby('cluster')
    # fig, ax = plt.subplots()
    # for name, group in groups:
    #     ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
    # ax.legend()

    # plt.show()

    # # 3D
    # # fig = plt.figure(figsize=(10, 10))
    # # ax = plt.axes(projection='3d')
    # # ax.scatter3D(data['x'], data['y'], data['z'], c=data['cluster'])
    # # plt.show()

    # print("End")


def euclideanDF(df1, df2):

    r = df1 - df2
    r = r.pow(2).sum(axis = 1).apply(np.sqrt)

    return r

def euclideanDF2(v1, v2):

    r = pd.DataFrame(v1.values - v2.values, columns=v1.columns)
    r = r.pow(2).sum(axis = 1).apply(np.sqrt)

    return r




def findNeighbors(df1, e):
    r = df1.index[(df1 <= e) & (df1 != -1)].tolist()

    return r




if __name__ == "__main__":
    main()