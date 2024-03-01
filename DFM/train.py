import argparse
import yaml
import os
import torch
from dataset import M2PDataset, shape_to_device
from utils import M2PLoss
from sklearn.neighbors import NearestNeighbors
from models.dgcnn_sample import DecoderSimpleDGCNN, DecoderSimpleDGCNN_sample
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

def euclidean_dist(x, y):
    bs, m, n = x.size(0), x.size(1), y.size(1)
    xx = torch.pow(x, 2).sum(2, keepdim=True).expand(bs, m, n)
    yy = torch.pow(y, 2).sum(2, keepdim=True).expand(bs, n, m).transpose(1, 2)
    dist = xx + yy - 2 * torch.bmm(x, y.transpose(1, 2))
    return dist

def knnsearch(x, y, alpha):
    # distance = euclidean_dist(x, y)
    distance = torch.cdist(x.float(), y.float())
    output = F.softmax(-alpha*distance, dim=-1)
    return output

def convert_C(Phi1, Phi2, A1, A2, alpha):
    T12 = knnsearch(A1, A2, alpha)
    T21 = knnsearch(A2, A1, alpha)
    C12_new = torch.bmm(torch.pinverse(Phi2), torch.bmm(T21, Phi1))
    C21_new = torch.bmm(torch.pinverse(Phi1), torch.bmm(T12, Phi2))
    return C12_new, C21_new

def z_aug(pc, pc_sample):
    # scale = np.diag(np.random.RandomState.uniform(1, 1, 3).astype(np.float32))
    rng = np.random.RandomState()
    angle = rng.uniform(-20, 20) / 180.0 * np.pi   ## multiway 
    rot_matrix = np.array([
        [np.cos(angle), 0., np.sin(angle)],
        [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]
    ], dtype=np.float32)
    # matrix = scale.dot(rot_matrix.T)
    device = pc.device
    bs = pc.shape[0]
    matrix = torch.from_numpy(rot_matrix).unsqueeze(0).repeat(bs, 1, 1).to(device)
    pc_rot = torch.bmm(pc, matrix)
    pc_sample_rot = torch.bmm(pc_sample, matrix)
    return pc_rot, pc_sample_rot

def train_net(cfg):
    if torch.cuda.is_available() and cfg["misc"]["cuda"]:
        device = torch.device(f'cuda:{cfg["misc"]["device"]}')
    else:
        device = torch.device("cpu")

    # important paths
    base_path = os.path.dirname(__file__)
    op_cache_dir = os.path.join(base_path, cfg["dataset"]["cache_dir"])
    dataset_path_train = os.path.join(cfg["dataset"]["root_dataset"], cfg["dataset"]["root_train"])
    dataset_path_test = os.path.join(cfg["dataset"]["root_dataset"], cfg["dataset"]["root_test"])

    save_dir_name = f'trained_{cfg["dataset"]["name"]}'
    model_save_path = os.path.join(base_path, f"ckpt/{save_dir_name}/ep" + "_{}.pth")
    if not os.path.exists(os.path.join(base_path, f"ckpt/{save_dir_name}/")):
        os.makedirs(os.path.join(base_path, f"ckpt/{save_dir_name}/"))

    # decide on the use of WKS descriptors
    with_wks = None if cfg["fmap"]["C_in"] <= 3 else cfg["fmap"]["C_in"]

    # create dataset
    train_dataset = M2PDataset(dataset_path_train, name=cfg["dataset"]["name"],
                               with_wks=with_wks, use_cache=True, 
                               op_cache_dir=op_cache_dir, train=True)

    test_dataset = M2PDataset(dataset_path_test, name=cfg["dataset"]["name"],
                              with_wks=with_wks, use_cache=True, 
                              op_cache_dir=op_cache_dir, train=False)


    # data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg["training"]['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg["training"]['batch_size'], shuffle=False)

    if cfg["training"]['model'] == 'dgcnn':
        point_backbone = DecoderSimpleDGCNN(device).to(device)
    elif cfg["training"]['model'] == 'dgcnnsample':
        point_backbone = DecoderSimpleDGCNN_sample(device=device).to(device)
        
    lr = float(cfg["optimizer"]["lr"])
    optimizer = torch.optim.Adam(point_backbone.parameters(), lr=lr, betas=(cfg["optimizer"]["b1"], cfg["optimizer"]["b2"]))
    criterion = M2PLoss(w_ortho=cfg["loss"]["w_ortho"],
                        w_bij=cfg["loss"]["w_bij"]).to(device)

    # Training loop
    print("start training")
    alpha_list = np.linspace(cfg["loss"]["min_alpha"], cfg["loss"]["max_alpha"]+1, cfg["training"]["epochs"])
    val_best_loss = 1e10
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        if epoch % cfg["optimizer"]["decay_iter"] == 0:
            lr *= cfg["optimizer"]["decay_factor"]
            print(f"Decaying learning rate, new one: {lr}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        alpha_i = alpha_list[epoch-1]
        
        point_backbone.train()
        for i, data in tqdm(enumerate(train_loader)):
            data = shape_to_device(data, device)
            # prepare iteration data
            V1, V2 = data["shape1"]['verts'], data["shape2"]['verts']
            V1_sample, V2_sample = data["shape1"]['verts_smaple'], data["shape2"]['verts_smaple']
            FPS1, FPS2 = data["shape1"]['fps'], data["shape2"]['fps']
            V1, V1_sample = z_aug(V1, V1_sample)
            V2, V2_sample = z_aug(V2, V2_sample) ## aug 
            evecs1, evecs2 = data["shape1"]['phi'], data["shape2"]['phi']
            
            feat1_p, feat2_p = point_backbone(V1.permute(0,2,1), V1_sample.permute(0,2,1), FPS1), point_backbone(V2.permute(0,2,1), V2_sample.permute(0,2,1), FPS2)
            C12_p, C21_p = convert_C(evecs1, evecs2, feat1_p, feat2_p, alpha_i)
            
            loss, bij_loss, ortho_loss = criterion(C12_p, C21_p)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            point_backbone.eval()
            val_loss_sum, val_ortho_loss_sum, val_bij_loss_sum = 0, 0, 0    
            val_iters = 0
            for i, data in tqdm(enumerate(test_loader)):
                data = shape_to_device(data, device)

                V1, V2 = data["shape1"]['verts'], data["shape2"]['verts']
                V1_sample, V2_sample = data["shape1"]['verts_smaple'], data["shape2"]['verts_smaple']
                FPS1, FPS2 = data["shape1"]['fps'], data["shape2"]['fps']
                evecs1, evecs2 = data["shape1"]['phi'], data["shape2"]['phi']

                feat1_p, feat2_p = point_backbone(V1.permute(0,2,1), V1_sample.permute(0,2,1), FPS1), point_backbone(V2.permute(0,2,1), V2_sample.permute(0,2,1), FPS2)
                C12_p, C21_p = convert_C(evecs1, evecs2, feat1_p, feat2_p, alpha_i)

                val_loss, val_bij_loss, val_ortho_loss = criterion(C12_p, C21_p)
                
                val_iters += 1
                val_loss_sum += val_loss
                val_ortho_loss_sum +=  val_ortho_loss
                val_bij_loss_sum += val_bij_loss
            
            print(f"epoch:{epoch}, val_loss:{val_loss_sum/val_iters}, val_ortho_loss:{val_ortho_loss_sum/val_iters}, val_bij_loss:{val_bij_loss_sum/val_iters}")
            

        # save model
        if (epoch + 1) % cfg["misc"]["checkpoint_interval"] == 0:
            torch.save(point_backbone.state_dict(), model_save_path.format(epoch))
        if val_loss_sum <= val_best_loss:
            val_best_loss = val_loss_sum
            torch.save(point_backbone.state_dict(), model_save_path.format('val_best'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the training of DQFM model.")
    parser.add_argument('--savedir', required=False, default="./ckpt", help='root directory of the dataset')
    parser.add_argument("--config", type=str, default="scape_r", help="Config file name")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"./config/{args.config}.yaml", "r"))
    print(cfg)
    train_net(cfg)
