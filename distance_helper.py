def eucledian_distance(current_data: tuple, other_data: tuple):
    i = 0
    distance_total = 0
    while i < len(current_data):
        categorical = False
        if not categorical: #means numeric

            try:
                current_value = float(current_data[i])
                other_value = float(other_data[i])
                distance_total = distance_total + (current_value - other_value) ** 2
            except: #this is in case there's an empty data point, so we simply just ignore and continue computing distance
                i += 1
                continue
        else: #for categorical
            #just compute it using ascii value of each word and add them up
            current_value = list(current_data[i])
            other_value = list(other_data[i])
            # import pdb; pdb.set_trace()
            similarity = jaccard_similarity(current_value, other_value)
            distance = 1 - similarity
            distance_total = distance_total + distance
        i += 1
    return distance_total ** (0.5)

def jaccard_similarity(current_value, other_value):
    intersection = len(list(set(current_value).intersection(list(set(other_value)))))
    union = (len(current_value) + len(other_value) - intersection)
    return float(intersection) / union