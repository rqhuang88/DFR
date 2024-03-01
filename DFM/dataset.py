import os
from pathlib import Path
import numpy as np
import potpourri3d as pp3d
import torch
from torch.utils.data import Dataset
from utils import farthest_point_sample
#
from tqdm import tqdm
from itertools import permutations
import scipy.io as scio
from diffusion_net.geometry import get_all_operators

class M2PDataset(Dataset):
    def __init__(self, root_dir, name="scape-remeshed",
                 with_wks=None, use_cache=True, 
                 op_cache_dir=None, train=True):

        self.root_dir = root_dir
        self.cache_dir = root_dir
        self.op_cache_dir = op_cache_dir
        # check the cache
        split = "train" if train else "test"
        wks_suf = "" if with_wks is None else "wks_"
        if use_cache:
            load_cache = os.path.join(self.cache_dir, f"cache_{name}_{wks_suf}{split}.pt")
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                (
                    self.verts_list,
                    self.face_list,
                    self.phi_list,
                    self.phi_inv_list,
                    self.fps_list,
                    self.eval_list,
                    self.vts_list,
                ) = torch.load(load_cache)

                self.combinations = list(permutations(range(len(self.verts_list)), 2))
                return
            print("  --> dataset not in cache, repopulating")

        # Load the meshes
        # define files and order
        shapes_split = "shapes_" + split
        self.used_shapes = sorted([x.stem for x in (Path(root_dir) / shapes_split).iterdir() if 'DS_' not in x.stem])

        # set combinations
        self.combinations = list(permutations(range(len(self.used_shapes)), 2))

        mesh_dirpath = Path(root_dir) / shapes_split
        vts_dirpath = Path(root_dir) / "corres"
        # Get all the files
        ext = '.off'
        self.verts_list = []
        self.faces_list = []
        self.phi_list = []
        self.fps_list = []
        self.eval_list = []
        self.vts_list = []
        # Load the actual files
        for shape_name in tqdm(self.used_shapes):
            data_dir = './mesh_results/'
            try:
                verts, faces = pp3d.read_mesh(str(mesh_dirpath / f"{shape_name}{ext}")) 
                vts = np.loadtxt(os.path.join(vts_dirpath, f'{shape_name}.vts'), dtype=int) - 1
            except:
                print(shape_name)
                exit()

            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))
            fps = farthest_point_sample(verts, verts.shape[0]).squeeze()
            # FPS to make large batch size
            
            self.fps_list.append(fps)
            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.vts_list.append(vts)
            
            # Precompute operators
            (
                self.frames_list,
                self.massvec_list,
                self.L_list,
                self.eval_list,
                self.phi_list,
                self.gradX_list,
                self.gradY_list,
            ) = get_all_operators(
                self.verts_list,
                self.faces_list,
                k_eig=128,
                op_cache_dir=self.op_cache_dir,
            )

        # save to cache
        if use_cache:
            torch.save(
                (
                    self.verts_list,
                    self.faces_list,
                    self.phi_list,
                    self.fps_list,
                    self.eval_list,
                    self.vts_list
                    
                ),
                load_cache,
            )

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        # get indexes
        idx1, idx2 = self.combinations[idx]
        fps1 = self.fps_list[idx1][:4995]
        fps2 = self.fps_list[idx2][:4995]
        
        verts1 = self.verts_list[idx1][fps1]
        verts2 = self.verts_list[idx2][fps2]
        verts1_sample = verts1[:3000]
        verts2_sample = verts2[:3000]
        fps1_sample = torch.arange(0,3000).long()
        fps2_sample = torch.arange(0,3000).long()
        
        shape1 = {
            "verts":verts1,
            "verts_smaple":verts1_sample,
            "fps": fps1_sample,
            "phi": self.phi_list[idx1][fps1],
            "eval": self.eval_list[idx1],
        }

        shape2 = {
            "verts":verts2,
            "verts_smaple":verts2_sample,
            "fps": fps2_sample,
            "phi": self.phi_list[idx2][fps2],
            "eval": self.eval_list[idx2],
        }
        
        # Compute fmap
        evec_1, evec_2 = self.phi_list[idx1][:, :50], self.phi_list[idx2][:, :50]
        vts1, vts2 = self.vts_list[idx1], self.vts_list[idx2]

        # try:            
        C12_gt = torch.pinverse(evec_2[vts2]) @ evec_1[vts1]
        C21_gt = torch.pinverse(evec_1[vts1]) @ evec_2[vts2]
        # except:
        #     C12_gt = torch.zeros_like(torch.pinverse(evec_2[:1000]) @ evec_1[:1000])
        #     C21_gt = torch.zeros_like(torch.pinverse(evec_1[:1000]) @ evec_2[:1000])

        return {"shape1": shape1, "shape2": shape2, "C12_gt": C12_gt, "C21_gt": C21_gt}
    
    

def shape_to_device(dict_shape, device):
    names_to_device = ["verts", "phi", "verts_smaple", "fps", "eval"]
    for k, v in dict_shape.items():
        if "shape" in k:
            for name in names_to_device:
                if v[name] is not None:
                    v[name] = v[name].to(device)  # .float()
            dict_shape[k] = v
        else:
            dict_shape[k] = v.to(device)

    return dict_shape


