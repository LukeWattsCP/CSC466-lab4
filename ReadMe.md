```bash
virtualenv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```
#kmeans.py
```bash
python3 kmeans.py <csvfile> <k> <mode>
<csvfile> = dataset path that we want to use 
<k> = the number of clusters desired
<mode>          -> 0 = kmeans (randomly selected centroids)
                -> 1 = kmeans hybrid (first centroid is random, 2nd to k are selected using k++ styled distance measures)
                -> 2 = kmeans++ (centroids slected using distances)

OUTPUT: 
The number of centroid recalculations performed.
For each cluster:
    The number of the cluster.
    The centroid location of the cluster in n-dimensions.
    The points that have the minimum, maximum, and average distance from the cluster centroid.
    The number of points in the cluster.
    For each point in the cluster:
        The location of the point in n-dimensions.
```


#hclustering.py
```bash
python3 hcluster.py <csvfile> <distance_method> <alpha>
<csvfile> = dataset path that we want to use 
<distance_method> -> 1 = singly link
                  -> 2 = complete link
                  -> 3 = average link
<alpha> the threshold value (this could be optional)

OUTPUT: 
<csvfile>_output.json --> gives the whole entire dendrogram (hiearchy)

if alpha != 0:
    <csvfile>_output_alpha.txt --> this gives information on how many clusters we will have after we cut off at that threshold off the dendrogram. Also list what each cluster contain and its height and etc.
    
    Console Output: The evaluation of all the clusters that were cut off from the alpha value
        -> Points, Center, Max Distance To Center, Min Distance To Center, Average Distance To Center

Note: If alpha is not provided, it would just take the threshold as 0 and the evaluation would just be a singular cluster (entire hiearchical) consisted of all points.
```


#dbscan.py
```bash
python3 dbscan.py <csvfile> <epsilon> <minpoints>
<csvfile> = dataset path that we want to use 
<epsilon> = radius value from point to form neighborhood
<minpoints> = the minimum number of points accessible to a point within distance epsilon to form a neighborhood

OUTPUT: 
TODO
```