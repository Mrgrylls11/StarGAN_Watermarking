# --- START OF FILE main.py ---

import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import torch # Add this import

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    # Create watermark output dir if specified and applying watermark
    if config.apply_watermark and config.watermark_output_dir and not os.path.exists(config.watermark_output_dir):
         os.makedirs(config.watermark_output_dir)
    # Create intermediate noisy dir if specified
    if config.save_noisy_intermediate and config.watermark_intermediate_dir and not os.path.exists(config.watermark_intermediate_dir):
         os.makedirs(config.watermark_intermediate_dir)


    # Data loader.
    celeba_loader = None
    rafd_loader = None

    # Check for dataset validity before creating loaders
    if config.dataset not in ['CelebA', 'RaFD', 'Both']:
        raise ValueError(f"Invalid dataset choice: {config.dataset}. Choose from 'CelebA', 'RaFD', 'Both'.")

    if config.dataset in ['CelebA', 'Both']:
        # Check if CelebA paths exist
        if not os.path.exists(config.celeba_image_dir):
            raise FileNotFoundError(f"CelebA image directory not found: {config.celeba_image_dir}")
        if not os.path.exists(config.attr_path):
            raise FileNotFoundError(f"CelebA attribute file not found: {config.attr_path}")

        celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers)
    if config.dataset in ['RaFD', 'Both']:
         # Check if RaFD path exists
        if not os.path.exists(config.rafd_image_dir):
            raise FileNotFoundError(f"RaFD image directory not found: {config.rafd_image_dir}")

        rafd_loader = get_loader(config.rafd_image_dir, None, None,
                                 config.rafd_crop_size, config.image_size, config.batch_size,
                                 'RaFD', config.mode, config.num_workers)

    # --- Add this check ---
    if config.apply_watermark:
        if not config.watermark_path or not os.path.exists(config.watermark_path):
             raise FileNotFoundError(f"Watermark image not found at specified path: {config.watermark_path}")
        if not config.watermark_output_dir:
             print("Warning: Applying watermark but --watermark_output_dir not set. Watermarked images won't be saved separately during testing.")
    # ---------------------

    # Solver for training and testing StarGAN.
    # Make sure the device is explicitly set for the solver if needed elsewhere
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    solver = Solver(celeba_loader, rafd_loader, config, device) # Pass device

    if config.mode == 'train':
        if config.dataset in ['CelebA', 'RaFD']:
            solver.train()
        elif config.dataset in ['Both']:
            solver.train_multi()
    elif config.mode == 'test':
        if config.dataset in ['CelebA', 'RaFD']:
            solver.test()
        elif config.dataset in ['Both']:
            solver.test_multi()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset') # Default 178
    parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution') # Default 128
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')

    # Training configuration.
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both'])
    # parser.add_argument('--batch_size', type=int, default=16) # Default 16 # Keep 1 for watermarking simplicity for now
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (set to 1 for easier watermark integration)')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D') # Default 200k
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr') # Default 100k
    parser.add_argument('--g_lr', type=float, default=0.00005, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.00005, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step') # Default 200k

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1) # Default 1
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True) # Default True

    # --- Watermark Configuration ---
    parser.add_argument('--apply_watermark', type=str2bool, default=False, help='Apply watermark to generated images')
    parser.add_argument('--watermark_path', type=str, default='watermark/watermarks/logo.png', help='Path to the watermark image file')
    parser.add_argument('--watermark_size_percent', type=float, default=5, help='Watermark size as a percentage of image width')
    parser.add_argument('--watermark_position', type=str, default='bottom-right',
                        choices=['top-left', 'top-right', 'bottom-left', 'bottom-right', 'center'],
                        help='Position to place the watermark')
    parser.add_argument('--watermark_output_dir', type=str, default='stargan/watermarked_results', help='Directory to save watermarked results during testing')
    parser.add_argument('--save_noisy_intermediate', type=str2bool, default=False, help='Save intermediate noisy images before watermarking')
    parser.add_argument('--watermark_intermediate_dir', type=str, default='stargan/intermediate_noisy', help='Directory for intermediate noisy images')
    # -----------------------------

    # Directories.
    # Make default paths more robust if needed, e.g., using os.path.join
    default_data_dir = 'data'
    default_stargan_dir = 'stargan_results' # Renamed to avoid conflict

    parser.add_argument('--celeba_image_dir', type=str, default=os.path.join(default_data_dir, 'celeba/images'))
    parser.add_argument('--attr_path', type=str, default=os.path.join(default_data_dir, 'celeba/list_attr_celeba.txt'))
    parser.add_argument('--rafd_image_dir', type=str, default=os.path.join(default_data_dir, 'RaFD/train')) # Adjust if you have test split

    parser.add_argument('--log_dir', type=str, default=os.path.join(default_stargan_dir, 'logs'))
    parser.add_argument('--model_save_dir', type=str, default=os.path.join(default_stargan_dir, 'models'))
    parser.add_argument('--sample_dir', type=str, default=os.path.join(default_stargan_dir, 'samples'))
    parser.add_argument('--result_dir', type=str, default=os.path.join(default_stargan_dir, 'results')) # Original results dir

    # Step size.
    parser.add_argument('--log_step', type=int, default=10) # Default 10
    parser.add_argument('--sample_step', type=int, default=1000) # Default 1000
    parser.add_argument('--model_save_step', type=int, default=10000) # Default 10k
    parser.add_argument('--lr_update_step', type=int, default=1000) # Default 1k

    config = parser.parse_args()

    # --- Add Print Statement for Watermark ---
    if config.apply_watermark:
        print("Watermarking Enabled:")
        print(f"  Path: {config.watermark_path}")
        print(f"  Size: {config.watermark_size_percent}%")
        print(f"  Position: {config.watermark_position}")
        print(f"  Output Dir: {config.watermark_output_dir}")
        print(f"  Save Noisy: {config.save_noisy_intermediate}")
        if config.save_noisy_intermediate:
            print(f"  Noisy Dir: {config.watermark_intermediate_dir}")
    else:
        print("Watermarking Disabled.")
    # ---------------------------------------

    print(config) # Print all configs
    main(config)
# --- END OF FILE main.py ---