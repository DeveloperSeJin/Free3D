from PIL import Image
import torch
from tqdm.auto import tqdm

# For generating point cloud
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud
from point_e.util.pc_to_mesh import marching_cubes_mesh

# For generating mesh
# import kaolin
import numpy as np
# from dmtet_network import Decoder

# Set the device to 'cpu' or 'cuda'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_point_cloud(image=None):
    """ Generate image to point cloud using Point-e and return the generated point cloud

    Args:
        image (Image or str, required): An Input Image for generating poinc cloud. Defaults to None.

    Raises:
        RuntimeError: When the Image is None, raise error

    Returns:
        PointCloud: Generated Point Cloud
    """
    
    print('creating base model...')
    base_name = 'base300M' # use base300M or base1B for better results
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
    if image is not None:
        img = Image.open(image) if isinstance(image, str) else image
    else:
        raise RuntimeError('Image is None')

    # Produce a sample from the model.
    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
        samples = x

    pc = sampler.output_to_point_clouds(samples)[0]
    # fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))

    return pc

def generate_mesh(points, path=None):
    """ Generate point cloud to mesh using DMTet network

    Args:
        points (PointCloud or ndarray): point cloud for generate 3d mesh
        path (_type_, optional): Path to save results. Defaults to None.

    """
    print('creating SDF model...')
    name = 'sdf'
    model = model_from_config(MODEL_CONFIGS[name], device)
    model.eval()

    print('loading SDF model...')
    model.load_state_dict(load_checkpoint(name, device))
    
    mesh = marching_cubes_mesh(
    pc=points,
    model=model,
    batch_size=4096,
    grid_size=32, # increase to 128 for resolution used in evals
    progress=True,
    )

    with open('mesh_1b.ply', 'wb') as f:
        mesh.write_ply(f)
        print("Mesh saved at ./mesh_1b.ply")

def main(image=None):
    point_cloud = generate_point_cloud(image=image)
    generate_mesh(points=point_cloud)

if __name__ == "__main__":
    main('../data/corgi.jpg')