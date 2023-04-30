import torch
import argparse
import sys

from nerf.provider import NeRFDataset
from nerf.utils import *

class Option():
    def __init__(
        self,
        image=None,
        workspace='workspace',
        dmtet=True,
        test=None,
        save_mesh=None,
        eval_interval=1,
        guidance='stable-diffusion',
        seed=None,
        known_view_interval=2,
        guidance_scale=100.0,
        mcubes_resolution=256,
        decimate_target=5e4,
        tet_grid_size=128,
        init_ckpt='',
        iters=10000,
        lr=1e-3,
        ckpt='latest',
        max_steps=1024,
        num_steps=64,
        upsample_steps=32,
        update_extra_interval=16,
        max_ray_batch=4096,
        warmup_iters=2000,
        uniform_sphere_rate=0.0,
        grad_clip=-1.0,
        grad_clip_rgb=-1.0,
        bg_radius=1.4,
        density_activation='softplus',
        density_thresh=0.1,
        blob_density=10.0,
        blob_radius=0.5,
        backbone='grid',
        optim='adan',
        sd_version='2.1',
        hf_key=None,
        w=64,
        h=64,
        known_view_scale=1.5,
        known_view_noise_scale=2e-3,
        dmtet_reso_scale=8.0,
        bound=1.0,
        dt_gamma=0.0,
        min_near=0.01,
        radius_range=[1.0, 1.5],
        theta_range=[45, 105],
        phi_range=[-180, 180],
        fovy_range=[40,80],
        default_radius=1.2,
        default_theta=90,
        default_phi=0,
        default_fovy=60,
        angle_overhead=30.0,
        angle_front=60.0,
        t_range=[0.02, 0.98],
        lambda_entropy=1e-3,
        lambda_opacity=0.0,
        lambda_orient=1e-2,
        lambda_tv=0.0,
        lambda_wd=0.0,
        lambda_mesh_normal=0.5,
        lambda_mesh_laplacian=0.5,
        lambda_guidance=1.0,
        lambda_rgb=10.0,
        lambda_mask=5.0,
        lambda_normal=0.0,
        lambda_depth=0.1,
        lambda_2d_normal_smooth=0.0,
        O=None,
        O2=None
    ) -> None:
        
        self.image = image
        self.workspace= workspace
        self.dmtet=dmtet
        self.test=test
        self.save_mesh=save_mesh
        self.eval_interval=eval_interval
        self.guidance=guidance
        self.seed=seed
        self.known_view_interval=known_view_interval
        self.guidance_scale=guidance_scale
        self.mcubes_resolution=mcubes_resolution
        self.decimate_target=decimate_target
        self.tet_grid_size=tet_grid_size
        self.init_ckpt=init_ckpt
        self.iters=iters
        self.lr=lr
        self.ckpt=ckpt
        self.max_steps=max_steps
        self.num_steps=num_steps
        self.upsample_steps=upsample_steps
        self.update_extra_interval=update_extra_interval
        self.max_ray_batch=max_ray_batch
        self.warmup_iters=warmup_iters
        self.uniform_sphere_rate=uniform_sphere_rate
        self.grad_clip=grad_clip
        self.grad_clip_rgb=grad_clip_rgb
        self.bg_radius=bg_radius
        self.density_activation=density_activation
        self.density_thresh=density_thresh
        self.blob_density=blob_density
        self.blob_radius=blob_radius
        self.backbone=backbone
        self.optim=optim
        self.sd_version=sd_version
        self.hf_key=hf_key
        self.w=w
        self.h=h
        self.known_view_scale=known_view_scale
        self.known_view_noise_scale=known_view_noise_scale
        self.dmtet_reso_scale=dmtet_reso_scale
        self.bound=bound
        self.dt_gamma=dt_gamma
        self.min_near=min_near
        self.radius_range=radius_range
        self.theta_range=theta_range
        self.phi_range=phi_range
        self.fovy_range=fovy_range
        self.default_radius=default_radius
        self.default_theta=default_theta
        self.default_phi=default_phi
        self.default_fovy=default_fovy
        self.angle_overhead=angle_overhead
        self.angle_front=angle_front
        self.t_range=t_range
        self.lambda_entropy=lambda_entropy
        self.lambda_opacity=lambda_opacity
        self.lambda_orient=lambda_orient
        self.lambda_tv=lambda_tv
        self.lambda_wd=lambda_wd
        self.lambda_mesh_normal=lambda_mesh_normal
        self.lambda_mesh_laplacian=lambda_mesh_laplacian
        self.lambda_guidance=lambda_guidance
        self.lambda_rgb=lambda_rgb
        self.lambda_mask=lambda_mask
        self.lambda_normal=lambda_normal
        self.lambda_depth=lambda_depth
        self.lambda_2d_normal_smooth=lambda_2d_normal_smooth
        self.O=O
        self.O2=O2
        
                
        
# torch.autograd.set_detect_anomaly(True)
def create_3d(
    image,
    workspace='workspace',
    dmtet=True,
    test=None,
    save_mesh=None,
):
    opt = Option(image=image, workspace=workspace, dmtet=dmtet, test=test, save_mesh=save_mesh)

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True

    elif opt.O2:
        opt.fp16 = True
        opt.backbone = 'vanilla'

    # parameters for image-conditioned generation
    if opt.image is not None:

        if opt.text is None:
            # use zero123 guidance model when only providing image
            opt.guidance = 'zero123' 
            opt.fovy_range = [opt.default_fovy, opt.default_fovy] # fix fov as zero123 doesn't support changing fov

            # very important to keep the image's content
            opt.guidance_scale = 3 
            opt.lambda_guidance = 0.02
            
        else:
            # use stable-diffusion when providing both text and image
            opt.guidance = 'stable-diffusion'
        
        opt.t_range = [0.02, 0.50]
        opt.lambda_orient = 10
        
        # latent warmup is not needed, we hardcode a 100-iter rgbd loss only warmup.
        opt.warmup_iters = 0 
        
        # make shape init more stable
        opt.progressive_view = True 
        opt.progressive_level = True

    # default parameters for finetuning
    if opt.dmtet:
        opt.h = int(opt.h * opt.dmtet_reso_scale)
        opt.w = int(opt.w * opt.dmtet_reso_scale)

        opt.t_range = [0.02, 0.50] # ref: magic3D

        # assume finetuning
        opt.warmup_iters = 0
        opt.progressive_view = False
        opt.progressive_level = False

        if opt.guidance != 'zero123':
            # smaller fovy (zoom in) for better details
            opt.fovy_range = [opt.fovy_range[0] - 10, opt.fovy_range[1] - 10] 

    # record full range for progressive view expansion
    if opt.progressive_view:
        # disable as they disturb progressive view
        opt.jitter_pose = False
        opt.uniform_sphere_rate = 0 
        # back up full range
        opt.full_radius_range = opt.radius_range
        opt.full_theta_range = opt.theta_range    
        opt.full_phi_range = opt.phi_range
        opt.full_fovy_range = opt.fovy_range

    if opt.backbone == 'vanilla':
        from nerf.network import NeRFNetwork
    elif opt.backbone == 'grid':
        from nerf.network_grid import NeRFNetwork
    elif opt.backbone == 'grid_tcnn':
        from nerf.network_grid_tcnn import NeRFNetwork
    elif opt.backbone == 'grid_taichi':
        opt.cuda_ray = False
        opt.taichi_ray = True
        import taichi as ti
        from nerf.network_grid_taichi import NeRFNetwork
        taichi_half2_opt = True
        taichi_init_args = {"arch": ti.cuda, "device_memory_GB": 4.0}
        if taichi_half2_opt:
            taichi_init_args["half2_vectorization"] = True
        ti.init(**taichi_init_args)
    else:
        raise NotImplementedError(f'--backbone {opt.backbone} is not implemented!')

    print(opt)

    if opt.seed is not None:
        seed_everything(int(opt.seed))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeRFNetwork(opt).to(device)

    if opt.dmtet and opt.init_ckpt != '':
        # load pretrained weights to init dmtet
        state_dict = torch.load(opt.init_ckpt, map_location=device)
        model.load_state_dict(state_dict['model'], strict=False)
        if opt.cuda_ray:
            model.mean_density = state_dict['mean_density']
        model.init_tet()

    print(model)

    if opt.test:
        guidance = None # no need to load guidance model at test

        trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.ckpt)

        if opt.gui:
            from nerf.gui import NeRFGUI
            gui = NeRFGUI(opt, trainer)
            gui.render()

        else:
            test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=100).dataloader()
            trainer.test(test_loader)

            if opt.save_mesh:
                trainer.save_mesh()

    else:

        train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=100).dataloader()

        if opt.optim == 'adan':
            from optimizer import Adan
            # Adan usually requires a larger LR
            optimizer = lambda model: Adan(model.get_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
        else: # adam
            optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        if opt.backbone == 'vanilla':
            scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
        else:
            scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed
            # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        if opt.guidance == 'stable-diffusion':
            from guidance.sd_utils import StableDiffusion
            guidance = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key, opt.t_range)
        elif opt.guidance == 'zero123':
            from guidance.zero123_utils import Zero123
            guidance = Zero123(device, opt.fp16, opt.vram_O, opt.t_range)
        elif opt.guidance == 'clip':
            from guidance.clip_utils import CLIP
            guidance = CLIP(device)
        else:
            raise NotImplementedError(f'--guidance {opt.guidance} is not implemented.')

        trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, scheduler_update_every_step=True)

        trainer.default_view_data = train_loader._data.get_default_view_data()

        if opt.gui:
            from nerf.gui import NeRFGUI
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()

        else:
            valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=5).dataloader()

            max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            trainer.train(train_loader, valid_loader, max_epoch)

            # also test at the end
            test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=100).dataloader()
            trainer.test(test_loader)

            if opt.save_mesh:
                trainer.save_mesh()

if __name__ == '__main__':
    create_3d()
