from PIL import Image
import torch
from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

import kaolin
import numpy as np
from point_e.examples.dmtet_network import Decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_point_cloud(path=None, image=None):

    print('creating base model...')
    base_name = 'base40M' # use base300M or base1B for better results
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    print('creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    print('downloading base checkpoint...')
    base_model.load_state_dict(load_checkpoint(base_name, device))

    print('downloading upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 3.0],
    )

    # Load an image to condition on.
    if path is not None:
        img = Image.open(path)
    else:
        img = image

    # Produce a sample from the model.
    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
        samples = x

    pc = sampler.output_to_point_clouds(samples)[0]
    # fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))

    return pc

def generate_mesh(points, path=None):
    # path to save
    if path is not None:
        logs_path = path
    else:
        logs_path = './examples/logs'
        
    # We initialize the timelapse that will store USD for the visualization apps
    timelapse = kaolin.visualize.Timelapse(logs_path)

    # arguments and hyperparameters
    lr = 1e-3
    laplacian_weight = 0.1
    iterations = 5000
    save_every = 100
    multires = 2
    grid_res = 128


    points = torch.FloatTensor(points).to(device)
    if points.shape[0] > 100000:
        idx = list(range(points.shape[0]))
        np.random.shuffle(idx)
        idx = torch.tensor(idx[:100000], device=points.device, dtype=torch.long)    
        points = points[idx]
    center = (points.max(0)[0] + points.min(0)[0]) / 2
    max_l = (points.max(0)[0] - points.min(0)[0]).max()
    points = ((points - center) / max_l)* 0.9
    timelapse.add_pointcloud_batch(category='input',
                                pointcloud_list=[points.cpu()], points_type = "usd_geom_points")
    

    tet_verts = torch.tensor(np.load('./point_e/examples/samples/{}_verts.npz'.format(grid_res))['data'], dtype=torch.float, device=device)
    tets = torch.tensor(([np.load('./point_e/examples/samples/{}_tets_{}.npz'.format(grid_res, i))['data'] for i in range(4)]), dtype=torch.long, device=device).permute(1,0)
    print(tet_verts.shape, tets.shape)

    # Initialize model and create optimizer
    model = Decoder(multires=multires).to(device)
    model.pre_train_sphere(1000)

    # Laplacian regularization using umbrella operator (Fujiwara / Desbrun).
    # https://mgarland.org/class/geom04/material/smoothing.pdf
    def laplace_regularizer_const(mesh_verts, mesh_faces):
        term = torch.zeros_like(mesh_verts)
        norm = torch.zeros_like(mesh_verts[..., 0:1])

        v0 = mesh_verts[mesh_faces[:, 0], :]
        v1 = mesh_verts[mesh_faces[:, 1], :]
        v2 = mesh_verts[mesh_faces[:, 2], :]

        term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1,3), (v1 - v0) + (v2 - v0))
        term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1,3), (v0 - v1) + (v2 - v1))
        term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1,3), (v0 - v2) + (v1 - v2))

        two = torch.ones_like(v0) * 2.0
        norm.scatter_add_(0, mesh_faces[:, 0:1], two)
        norm.scatter_add_(0, mesh_faces[:, 1:2], two)
        norm.scatter_add_(0, mesh_faces[:, 2:3], two)

        term = term / torch.clamp(norm, min=1.0)

        return torch.mean(term**2)

    def loss_f(mesh_verts, mesh_faces, points, it):
        pred_points = kaolin.ops.mesh.sample_points(mesh_verts.unsqueeze(0), mesh_faces, 50000)[0][0]
        chamfer = kaolin.metrics.pointcloud.chamfer_distance(pred_points.unsqueeze(0), points.unsqueeze(0)).mean()
        if it > iterations//2:
            lap = laplace_regularizer_const(mesh_verts, mesh_faces)
            return chamfer + lap * laplacian_weight
        return chamfer
    
    vars = [p for _, p in model.named_parameters()]
    optimizer = torch.optim.Adam(vars, lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.0, 10**(-x*0.0002))) # LR decay over time

    for it in range(iterations):
        pred = model(tet_verts) # predict SDF and per-vertex deformation
        sdf, deform = pred[:,0], pred[:,1:]
        verts_deformed = tet_verts + torch.tanh(deform) / grid_res # constraint deformation to avoid flipping tets
        mesh_verts, mesh_faces = kaolin.ops.conversions.marching_tetrahedra(verts_deformed.unsqueeze(0), tets, sdf.unsqueeze(0)) # running MT (batched) to extract surface mesh
        mesh_verts, mesh_faces = mesh_verts[0], mesh_faces[0]

        loss = loss_f(mesh_verts, mesh_faces, points, it)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (it) % save_every == 0 or it == (iterations - 1): 
            print ('Iteration {} - loss: {}, # of mesh vertices: {}, # of mesh faces: {}'.format(it, loss, mesh_verts.shape[0], mesh_faces.shape[0]))
            # save reconstructed mesh
            timelapse.add_mesh_batch(
                iteration=it+1,
                category='extracted_mesh',
                vertices_list=[mesh_verts.cpu()],
                faces_list=[mesh_faces.cpu()]
            )

def main(path=None):
    pcd = generate_point_cloud()
    generate_mesh(pcd, path)

if __name__ == "__main__":
    main()