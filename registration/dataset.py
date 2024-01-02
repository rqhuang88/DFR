import os
from pathlib import Path
import numpy as np
import potpourri3d as pp3d
import torch
from torch.utils.data import Dataset
from utils import farthest_point_sample,pc_normalize
#
from tqdm import tqdm
from itertools import permutations



class ScapeDataset(Dataset):

    def __init__(self, root_dir, name="scape-remeshed",use_cache=True,train=True,single=False):


        self.root_dir = root_dir
        self.cache_dir = root_dir


        # check the cache
        split = "train" if train else "test"
        if use_cache:
            load_cache = os.path.join(self.cache_dir, f"cache_{name}_{split}.pt")
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                (
                    # main
                    self.verts_list,
                    self.faces_list,
                    # misc
                    self.used_shapes,
                    self.vts_list,
                    self.fps_list
                ) = torch.load(load_cache)

                self.combinations = list(permutations(range(len(self.verts_list)), 2))[0:len(self.verts_list)-1]
                self.combinations.append((0,0))
                
                
                
                return
            print("  --> dataset not in cache, repopulating")

        # Load the meshes
        # define files and order
        shapes_split = "shapes_" + split
        self.used_shapes = sorted([x.stem for x in (Path(root_dir) / shapes_split).iterdir() if 'DS_' not in x.stem])
        
        self.combinations = list(permutations(range(len(self.used_shapes)), 2))[0:len(self.used_shapes)-1]
        self.combinations.append((0,0))
        
        self.root_dir = root_dir
        #
        mesh_dirpath = Path(root_dir) / shapes_split
        extfps = '.npy'
        # Get all the files
        ext = '.off'
        self.verts_list = []
        self.faces_list = []
        self.vts_list = []
        self.fps_list = []
        # Load the actual files
        for shape_name in tqdm(self.used_shapes):
            
            verts, faces = pp3d.read_mesh(str(mesh_dirpath / f"{shape_name}{ext}"))  # off ob
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))
            fps = farthest_point_sample(verts,verts.shape[0]).squeeze()
            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.fps_list.append(fps)
            # vts
            vts = torch.tensor(np.ascontiguousarray(verts)).float()
            self.vts_list.append(vts)


        print('done')

        # save to cache
        if use_cache:
            ensure_dir_exists(self.cache_dir)
            torch.save(
                (
                    self.verts_list,
                    self.faces_list,
                    self.used_shapes,
                    self.vts_list,
                    self.fps_list
                ),
                load_cache,
            )

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):

        # get indexes
        idx1, idx2 = self.combinations[idx]
        fps1 = self.fps_list[idx1]
        fps2 = self.fps_list[idx2]
        fps1 = fps1[:3000]
        fps2 = fps2[:3000]
        self.vts_list[idx1] = pc_normalize(self.vts_list[idx1])
        self.vts_list[idx2] = pc_normalize(self.vts_list[idx2])
        shape1 = {
            "xyz": self.verts_list[idx1],
            "faces": self.faces_list[idx1],
            "vts": self.vts_list[idx1],
            "vts_sample": self.vts_list[idx1][fps1],
            "name": self.used_shapes[idx1],
            "fps": fps1
        }

        shape2 = {
            "xyz": self.verts_list[idx2],
            "faces": self.faces_list[idx2],
            "vts":self.vts_list[idx2] ,
            "vts_sample": self.vts_list[idx2][fps2],
            "name": self.used_shapes[idx2],
            "fps": fps2
        }
        # print(shape1.keys())
        return {"shape1": shape1, "shape2": shape2}


def shape_to_device(dict_shape, device):
    names_to_device = ["xyz", "faces"]
    for k, v in dict_shape.items():
        if "shape" in k:
            for name in names_to_device:
                if v[name] is not None:
                    v[name] = v[name].to(device)  # .float()
            dict_shape[k] = v
        else:
            dict_shape[k] = v.to(device)

    return dict_shape
def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)