def find_closed_observation(item, candidates, distance_metric, threshold=0):
    distances = []
    found = False
    for c in candidates:
        dist = distance_metric(item[:-1], c[:-1])
        distances.append(dist)
        if dist <= threshold:
            found = True
    return distances, found
