```bash
virtualenv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```
#kmeans.py
```bash
python3 kmeans.py <csvfile> <k> <mode>
<csvfile> = dataset path that we want to use 
<k> the number of clusters desired
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
if alpha is provided:
    <csvfile>_output_alpha.txt --> this gives information on how many clusters we will have after we cut off at that threshold off the dendrogram. Also list what each cluster contain and its height and etc.
    <csvfile>_output.json --> gives the whole entire dendrogram (hiearchy)
```