import numpy as np
import os
from tqdm import tqdm
import json


# SPLIT_NUM = 20
# OVERLAP = 1

root = 'generated_meshes'
save_root = 'splited_meshes'
points_group = 'group.json'

lst = [os.path.join(root, x) for x in os.listdir(root) if x.endswith('.obj')]

for name in tqdm(lst):
    points = []
    surface = []

    with open(name) as fp:
        while 1:
            line = fp.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append([float(strs[1]), float(strs[2]), float(strs[3])])
            if strs[0] == "f":
                # print(strs)
                # assert False
                surface.append([int(strs[1]), int(strs[2]), int(strs[3])])
                # surface.append([int(strs[1].split('//')[0]), int(strs[2].split('//')[0]), int(strs[3].split('//')[0])])

    points = np.array(points)
    surface = np.array(surface)
    
    # print('finish reading')

    with open(points_group) as fp:
        jsonfile = json.load(fp)

    metadata = jsonfile['1']

    SPLIT_NUM = len(metadata)

    # print(metadata)
    # assert False

    split_points = [{} for _ in range(SPLIT_NUM)]

    # counts = [0 for _ in range(SPLIT_NUM)]

    for i in range(31):
        temp_metadata = metadata[i]
        for j in range(len(temp_metadata)):
            split_points[i][temp_metadata[j]] = list(points[temp_metadata[j] - 1]) + [j + 1]
            # print(split_points[i][j + 1])
            # assert False
        # print(split_points[i])
        # assert False

    # print('finish spliting')
    # # print(points.shape)
    # # print(counts)
    # print('start distributing')


    split_surfaces = [[] for _ in range(SPLIT_NUM)]


    for i in range(SPLIT_NUM):
        temp_obj = split_points[i]
        for j in range(surface.shape[0]):
            temp_surface = list(surface[j])
            # print(temp_surface)
            a, b, c = temp_surface[0], temp_surface[1], temp_surface[2]
            if a in temp_obj.keys() and b in temp_obj.keys() and c in temp_obj.keys():
                split_surfaces[i].append([
                    temp_obj[a][-1], temp_obj[b][-1], temp_obj[c][-1], 
                    a, b, c
                ])
                # print([temp_obj[a][-1], temp_obj[b][-1], temp_obj[c][-1]])
                # print(a, b, c)
                # assert False
            # print(temp_surface)
            # assert False

    # print('finish distributing')

    # print('start writing')

    for i in range(SPLIT_NUM):
        # for k, v in split_points[i].items():
            # print(k, v)
            # assert False
        # for f in split_surfaces[i]:
            # print(f)
            # assert False
        save_path = f'{name.replace(root, save_root).split(".")[0]}/{i}.obj'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding="utf-8") as fp:
            fp.write("".join([f'v {v[0]:f} {v[1]:f} {v[2]:f}\n' 
                            for k, v in split_points[i].items()]))
            fp.write("".join([f'f {f[0]} {f[1]} {f[2]}\n' for f in split_surfaces[i]]))
        
        # assert False
