import numpy as np
from PIL import Image
import shapely


def get_map(file_name):
    image_path = "./environments/maps/" + file_name + ".png"
    color_array = {}
    count = 2 
    image = Image.open(image_path)
    image = image.convert("RGB")
    old_map = np.asarray(image)
    map = np.zeros((old_map.shape[0], old_map.shape[1]))
    for x in range(old_map.shape[0]):
        for y in range(old_map.shape[1]):
            if np.array_equal(old_map[x, y], np.array((0,0,0))):
                map[x,y] = 1
            #adds any other color than black or white as pitfall, can add additional colors here for different enviroments
            elif np.array_equal(old_map[x,y], np.array((255,0,0))):
                map[x,y] = -1
            elif not np.array_equal(old_map[x, y], np.array((255,255,255))):
                map_bytes = old_map[x,y].tobytes()
                if map_bytes in color_array.keys():
                    map[x,y] = color_array[map_bytes]
                else:
                    map[x,y] = count
                    color_array[map_bytes] = count
                    count +=1
    return map


def read_obstacles(configuration):
    obstacles = []
    obstacle_points = []
    with open(configuration) as fp:
        for line in fp.readlines():
            tokens = line.strip().split()
            if not len(tokens):
                continue
            elif tokens[0] == 'polygon':
                points = list(zip(*[iter(map(float, tokens[1:]))] * 2))
                obstacle = shapely.geometry.Polygon(points)
                obstacles.append(obstacle)
                obstacle_points.append(points)
    return obstacles, obstacle_points

