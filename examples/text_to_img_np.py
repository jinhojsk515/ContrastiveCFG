import argparse
from pathlib import Path

from munch import munchify
from torchvision.utils import save_image

from latent_diffusion_np import get_solver
from latent_sdxl_np import get_solver as get_solver_sdxl
from utils.log_util import create_workdir, set_seed
from tqdm import tqdm
import torch


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Latent Diffusion")
    parser.add_argument("--workdir", type=Path, default="examples/workdir/t2i")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--null_prompt", type=str, default="")
    parser.add_argument("--pos_prompt", type=str, default="")
    parser.add_argument("--neg_prompt", type=str, default="")
    parser.add_argument("--cfg_guidance", type=float, default=7.5)
    parser.add_argument("--tau", type=float, default=0.2)                       # for CCFG
    parser.add_argument("--naive_positive_cfg", cation='store_true')       # use naive CFG instead of CCFG for positive prompt
    parser.add_argument("--method", type=str, default='ddim_np_ccfg', choices=['ddim_np_naive', 'ddim_np_ccfg'])
    parser.add_argument("--model", type=str, default='sd15', choices=["sd15", "sdxl"])
    parser.add_argument("--NFE", type=int, default=50)
    parser.add_argument("--n_sample", type=int, default=1)
    parser.add_argument("--minibatch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    create_workdir(args.workdir)

    solver_config = munchify({'num_sampling': args.NFE})

    if args.model == "sdxl":
        output_img_paths = []
        solver = get_solver_sdxl(args.method,
                                 solver_config=solver_config,
                                 device=args.device)
        print('solver type: ', solver.__class__.__name__)
        idx = 0
        for _ in tqdm(range(args.n_sample // args.minibatch)):
            zT = torch.randn(args.minibatch, 4, 128, 128).to(solver.device)
            result = solver.sample(prompt1=[args.null_prompt, args.pos_prompt, args.neg_prompt],
                                   prompt2=[args.null_prompt, args.pos_prompt, args.neg_prompt],
                                   cfg_guidance=args.cfg_guidance,
                                   target_size=(1024, 1024),
                                   zT=zT, quiet=True, coeff={'tau': args.tau, 'naive_positive_cfg': args.naive_positive_cfg})
            for i in range(zT.shape[0]):
                save_image(result[i], args.workdir.joinpath(f'result/generated_{idx}.png'), normalize=True)
                output_img_paths.append(str(args.workdir.joinpath(f'result/generated_{idx}.png')))
                idx += 1
    else:
        output_img_paths = []
        solver = get_solver(args.method,
                            solver_config=solver_config,
                            device=args.device)
        print('solver type: ', solver.__class__.__name__)
        idx = 0
        for _ in tqdm(range(args.n_sample // args.minibatch)):
            zT = torch.randn(args.minibatch, 4, 64, 64).to(solver.device)
            result = solver.sample(prompt=[args.null_prompt, args.pos_prompt, args.neg_prompt],
                                   cfg_guidance=args.cfg_guidance,
                                   zT=zT, quiet=True, coeff={'tau': args.tau, 'naive_positive_cfg': args.naive_positive_cfg})
            for i in range(zT.shape[0]):
                save_image(result[i], args.workdir.joinpath(f'result/generated_{idx}.png'), normalize=True)
                output_img_paths.append(str(args.workdir.joinpath(f'result/generated_{idx}.png')))
                idx += 1


if __name__ == "__main__":
    main()
