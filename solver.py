# --- START OF FILE solver.py ---

from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import cv2 # Import OpenCV

# --- Import Watermark Utilities ---
from watermark_utils import tensor_to_cv2, cv2_to_tensor, apply_robust_watermark_cv2
# ---------------------------------


class Solver(object):
    """Solver for training and testing StarGAN."""

    # --- Modified __init__ ---
    def __init__(self, celeba_loader, rafd_loader, config, device=None): # Add device argument
        """Initialize configurations."""

        # Data loader.
        self.celeba_loader = celeba_loader
        self.rafd_loader = rafd_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        # Use the device passed from main.py or determine automatically
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Solver using device: {self.device}")


        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir # Original non-watermarked results

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # --- Watermark Configuration ---
        self.apply_watermark = config.apply_watermark
        self.watermark_path = config.watermark_path
        self.watermark_size_percent = config.watermark_size_percent
        self.watermark_position = config.watermark_position
        self.watermark_img = None
        self.watermark_output_dir = config.watermark_output_dir # For saving watermarked test results
        self.save_noisy_intermediate = config.save_noisy_intermediate
        self.watermark_intermediate_dir = config.watermark_intermediate_dir


        if self.apply_watermark:
            try:
                # Load watermark with alpha channel if available
                self.watermark_img = cv2.imread(self.watermark_path, cv2.IMREAD_UNCHANGED)
                if self.watermark_img is None:
                    raise FileNotFoundError(f"Failed to load watermark image from {self.watermark_path}")
                print(f"✅ Watermark image '{os.path.basename(self.watermark_path)}' loaded successfully ({self.watermark_img.shape}).")
                # Pre-convert watermark to BGR if it's grayscale
                if len(self.watermark_img.shape) == 2:
                     self.watermark_img = cv2.cvtColor(self.watermark_img, cv2.COLOR_GRAY2BGR)
                     print("ℹ️ Converted grayscale watermark to BGR.")
                elif self.watermark_img.shape[2] == 3:
                     print("ℹ️ Loaded BGR watermark.")
                elif self.watermark_img.shape[2] == 4:
                     print("ℹ️ Loaded BGRA watermark (with alpha).")

            except Exception as e:
                print(f"❌ Error loading watermark image: {e}")
                print("⚠️ Watermarking will be disabled.")
                self.apply_watermark = False
        # -----------------------------

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()
    # -------------------------

    # --- Add Watermark Application Method ---
    def _apply_watermark_batch(self, tensor_batch, iteration=None, save_prefix=""):
        """Applies watermark to a batch of tensors."""
        if not self.apply_watermark or self.watermark_img is None:
            return tensor_batch

        watermarked_tensors = []
        # Process batch item by item (OpenCV operations are not easily vectorized on GPU batches)
        for i in range(tensor_batch.size(0)):
            single_tensor = tensor_batch[i]
            # Ensure tensor is on CPU for conversion
            image_cv2 = tensor_to_cv2(single_tensor.detach().cpu())

            noisy_save_path = None
            if self.save_noisy_intermediate and self.watermark_intermediate_dir and iteration is not None:
                 noisy_filename = f"{save_prefix}noisy_{iteration}_{i}.png"
                 noisy_save_path = os.path.join(self.watermark_intermediate_dir, noisy_filename)
                 # The saving now happens inside apply_robust_watermark_cv2

            # Apply watermark using the utility function
            watermarked_cv2 = apply_robust_watermark_cv2(
                image_cv2,
                self.watermark_img,
                self.watermark_size_percent,
                self.watermark_position,
                save_noisy_path=noisy_save_path
            )

            # Convert back to tensor and move to original device
            watermarked_tensor = cv2_to_tensor(watermarked_cv2).to(self.device)
            watermarked_tensors.append(watermarked_tensor)

        # Stack tensors back into a batch
        return torch.stack(watermarked_tensors)
    # -------------------------------------


    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD']:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)
        elif self.dataset in ['Both']:
            self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector.
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        # Ensure model exists before loading
        if not os.path.exists(G_path):
             print(f"❌ Generator checkpoint not found at {G_path}. Cannot restore.")
             return False # Indicate failure
        if not os.path.exists(D_path):
             print(f"❌ Discriminator checkpoint not found at {D_path}. Cannot restore.")
             return False # Indicate failure

        try:
            self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
            self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
            print("✅ Models restored successfully.")
            return True # Indicate success
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False # Indicate failure


    def build_tensorboard(self):
        """Build a tensorboard logger."""
        try:
            from logger import Logger # Assuming logger.py uses tensorboardX or similar
            self.logger = Logger(self.log_dir)
            print("✅ TensorBoard logger initialized.")
        except ImportError:
            print("⚠️ TensorBoard logger (logger.py) not found or import error. Disabling TensorBoard.")
            self.use_tensorboard = False
        except Exception as e:
            print(f"⚠️ Error initializing TensorBoard logger: {e}. Disabling TensorBoard.")
            self.use_tensorboard = False


    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        # Add small epsilon to prevent potential division by zero if range is exactly [-1, 1]
        out = (x + 1.00001) / 2.00001
        return out.clamp_(0, 1)


    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim).to(self.device) # Ensure output is on the correct device
        # Ensure labels are LongTensor for indexing
        out[np.arange(batch_size), labels.long()] = 1
        return out


    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        hair_color_indices = []
        if dataset == 'CelebA':
             # Ensure selected_attrs is not None before iterating
            if selected_attrs:
                for i, attr_name in enumerate(selected_attrs):
                    if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                        hair_color_indices.append(i)
            else:
                 print("Warning: selected_attrs is None in create_labels for CelebA.")


        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    # Ensure attribute index i is valid
                    if i < c_trg.shape[1]:
                         # Reverse attribute value: 1 if 0, 0 if 1
                         c_trg[:, i] = 1.0 - c_trg[:, i] # Assuming values are 0.0 or 1.0
                    else:
                         print(f"Warning: Attribute index {i} out of bounds for c_trg shape {c_trg.shape} in CelebA label creation.")

            elif dataset == 'RaFD':
                 # Ensure c_org has data before using its size
                if c_org.size(0) > 0:
                    c_trg = self.label2onehot(torch.ones(c_org.size(0)).to(self.device) * i, c_dim) # Ensure tensor creation is on device
                else:
                    # Handle empty batch case if necessary
                    c_trg = torch.empty(0, c_dim).to(self.device)


            c_trg_list.append(c_trg.to(self.device)) # Ensure final tensor is on device
        return c_trg_list


    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        # Ensure target is on the same device as logit
        target = target.to(self.device)

        if dataset == 'CelebA':
             # Ensure target is float for BCEWithLogitsLoss
            return F.binary_cross_entropy_with_logits(logit, target.float(), reduction='sum') / logit.size(0)
        elif dataset == 'RaFD':
             # Ensure target is long for CrossEntropyLoss
            return F.cross_entropy(logit, target.long())


    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader
        else: # Should not happen if checked in main.py, but as safeguard
             print(f"Error: Invalid dataset '{self.dataset}' specified for single training.")
             return

        if data_loader is None:
            print(f"Error: Data loader for {self.dataset} is None. Cannot train.")
            return
        if len(data_loader) == 0:
             print(f"Error: Data loader for {self.dataset} is empty. Cannot train.")
             return


        # Fetch fixed inputs for debugging.
        try:
            data_iter = iter(data_loader)
            x_fixed, c_org = next(data_iter)
            x_fixed = x_fixed.to(self.device)
            c_fixed_list = self.create_labels(c_org.to(self.device), self.c_dim, self.dataset, self.selected_attrs) # Ensure c_org is on device
        except StopIteration:
            print("Error: Could not fetch fixed data batch, dataset might be too small or empty.")
            return # Cannot proceed without fixed data for visualization

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            if self.restore_model(self.resume_iters):
                 start_iters = self.resume_iters
            else:
                 print(f"Could not restore model from iteration {self.resume_iters}. Starting from scratch.")


        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except StopIteration: # Changed from generic except
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)
            except Exception as e: # Catch other potential loader errors
                print(f"Error fetching data at iteration {i+1}: {e}")
                continue # Skip this iteration


            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            if self.dataset == 'CelebA':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            elif self.dataset == 'RaFD':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

            # Compute loss with fake images.
            x_fake = self.G(x_real, c_trg)

            # --- Apply watermark to fake images before D ---
            x_fake_input_d = x_fake.detach() # Detach first
            if self.apply_watermark:
                # Pass iteration number for potential noisy image saving
                x_fake_input_d = self._apply_watermark_batch(x_fake_input_d, iteration=(i+1), save_prefix="d_train_")
            # ---------------------------------------------

            out_src, out_cls_fake = self.D(x_fake_input_d) # Use potentially watermarked fake
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            # Use the non-watermarked x_fake for interpolation
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src_hat, _ = self.D(x_hat) # Pass the interpolated (non-watermarked) image through D for GP
            d_loss_gp = self.gradient_penalty(out_src_hat, x_hat)


            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg) # Generate fake image

                # --- Apply watermark before G's D pass ---
                x_fake_input_g = x_fake
                if self.apply_watermark:
                     # Pass iteration number for potential noisy image saving
                    x_fake_input_g = self._apply_watermark_batch(x_fake_input_g, iteration=(i+1), save_prefix="g_train_")
                # -----------------------------------------

                out_src, out_cls = self.D(x_fake_input_g) # Pass potentially watermarked fake to D
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                # Target-to-original domain (Reconstruction)
                # Use the *original* non-watermarked x_fake for reconstruction input
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst)) # Compare reconstruction to original real image

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                # Add keys safely
                log_items = {k: v for k, v in loss.items() if v is not None}
                for tag, value in log_items.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard and self.logger: # Check if logger exists
                    for tag, value in log_items.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed] # Start with the original fixed images
                    for c_fixed in c_fixed_list:
                        generated_img = self.G(x_fixed, c_fixed)
                        # --- Apply watermark to sample ---
                        if self.apply_watermark:
                            # Pass iteration number for potential noisy image saving
                            generated_img = self._apply_watermark_batch(generated_img, iteration=(i+1), save_prefix="sample_")
                        # ---------------------------------
                        x_fake_list.append(generated_img)

                    # Concatenate along the width dimension (dim=3)
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    try:
                         save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                         print('Saved real and fake sampled images into {}...'.format(sample_path))
                    except Exception as e:
                         print(f"❌ Error saving sample image {sample_path}: {e}")


            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                try:
                    torch.save(self.G.state_dict(), G_path)
                    torch.save(self.D.state_dict(), D_path)
                    print('Saved model checkpoints into {}...'.format(self.model_save_dir))
                except Exception as e:
                     print(f"❌ Error saving model checkpoints at iteration {i+1}: {e}")


            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


    # train_multi needs similar watermark integration if used.
    # Focus on train() first.

    def train_multi(self):
        """Train StarGAN with multiple datasets."""
        # Data iterators.
        if self.celeba_loader is None or self.rafd_loader is None:
            print("Error: Both CelebA and RaFD loaders are required for multi-dataset training.")
            return
        celeba_iter = iter(self.celeba_loader)
        rafd_iter = iter(self.rafd_loader)

        # Fetch fixed inputs for debugging.
        try:
            x_fixed, c_org = next(celeba_iter) # Use CelebA for fixed visualization
            x_fixed = x_fixed.to(self.device)
            c_celeba_list = self.create_labels(c_org.to(self.device), self.c_dim, 'CelebA', self.selected_attrs)
            c_rafd_list = self.create_labels(c_org.to(self.device), self.c2_dim, 'RaFD') # Use c_org shape, but generate RaFD labels
            zero_celeba = torch.zeros(x_fixed.size(0), self.c_dim).to(self.device)           # Zero vector for CelebA.
            zero_rafd = torch.zeros(x_fixed.size(0), self.c2_dim).to(self.device)             # Zero vector for RaFD.
            mask_celeba = self.label2onehot(torch.zeros(x_fixed.size(0)), 2).to(self.device)  # Mask vector: [1, 0].
            mask_rafd = self.label2onehot(torch.ones(x_fixed.size(0)), 2).to(self.device)     # Mask vector: [0, 1].
        except StopIteration:
            print("Error: Could not fetch fixed data batch from CelebA loader for multi-training.")
            return

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            if self.restore_model(self.resume_iters):
                 start_iters = self.resume_iters
            else:
                 print(f"Could not restore model from iteration {self.resume_iters}. Starting from scratch.")

        # Start training.
        print('Start training multi-dataset...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            # Choose dataset for this iteration (alternate or random)
            # Simple alternation for this example:
            dataset = 'CelebA' if i % 2 == 0 else 'RaFD'

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            data_iter = celeba_iter if dataset == 'CelebA' else rafd_iter
            current_loader = self.celeba_loader if dataset == 'CelebA' else self.rafd_loader

            try:
                x_real, label_org = next(data_iter)
            except StopIteration:
                # Reset iterator if it ends
                data_iter = iter(current_loader)
                x_real, label_org = next(data_iter)
            except Exception as e:
                 print(f"Error fetching data for {dataset} at iteration {i+1}: {e}")
                 continue # Skip this iteration


            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            # Create combined labels for the generator/discriminator
            if dataset == 'CelebA':
                c_org_specific = label_org.clone()
                c_trg_specific = label_trg.clone()
                zero = torch.zeros(x_real.size(0), self.c2_dim).to(self.device) # RaFD part is zero
                mask = self.label2onehot(torch.zeros(x_real.size(0)), 2).to(self.device) # Mask is [1, 0]
                c_org = torch.cat([c_org_specific, zero, mask], dim=1)
                c_trg = torch.cat([c_trg_specific, zero, mask], dim=1)
                cls_label_org = label_org # For D's classification loss (CelebA part)
                cls_label_trg = label_trg # For G's classification loss (CelebA part)
                cls_dim_offset = 0        # Index offset for D's classification output
                cls_target_dim = self.c_dim
            elif dataset == 'RaFD':
                c_org_specific = self.label2onehot(label_org, self.c2_dim)
                c_trg_specific = self.label2onehot(label_trg, self.c2_dim)
                zero = torch.zeros(x_real.size(0), self.c_dim).to(self.device) # CelebA part is zero
                mask = self.label2onehot(torch.ones(x_real.size(0)), 2).to(self.device) # Mask is [0, 1]
                c_org = torch.cat([zero, c_org_specific, mask], dim=1)
                c_trg = torch.cat([zero, c_trg_specific, mask], dim=1)
                cls_label_org = label_org # For D's classification loss (RaFD part - needs index)
                cls_label_trg = label_trg # For G's classification loss (RaFD part - needs index)
                cls_dim_offset = self.c_dim # Index offset for D's classification output
                cls_target_dim = self.c2_dim

            x_real = x_real.to(self.device)             # Input images.
            c_org = c_org.to(self.device)               # Original domain labels (combined).
            c_trg = c_trg.to(self.device)               # Target domain labels (combined).
            # cls_label_org/trg are already on device from label_org/trg

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls_all = self.D(x_real)
            # Select the classification output corresponding to the current dataset
            out_cls = out_cls_all[:, cls_dim_offset : cls_dim_offset + cls_target_dim]
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, cls_label_org, dataset) # Use specific dataset loss

            # Compute loss with fake images.
            x_fake = self.G(x_real, c_trg)

            # --- Apply watermark to fake images before D ---
            x_fake_input_d = x_fake.detach() # Detach first
            if self.apply_watermark:
                x_fake_input_d = self._apply_watermark_batch(x_fake_input_d, iteration=(i+1), save_prefix=f"d_train_{dataset}_")
            # ---------------------------------------------

            out_src_fake, _ = self.D(x_fake_input_d) # D only needs source loss for fake
            d_loss_fake = torch.mean(out_src_fake)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            # Use the non-watermarked x_fake for interpolation
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src_hat, _ = self.D(x_hat) # Pass the interpolated (non-watermarked) image through D for GP
            d_loss_gp = self.gradient_penalty(out_src_hat, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss[f'D/loss_real_{dataset}'] = d_loss_real.item()
            loss[f'D/loss_fake_{dataset}'] = d_loss_fake.item()
            loss[f'D/loss_cls_{dataset}'] = d_loss_cls.item()
            loss[f'D/loss_gp_{dataset}'] = d_loss_gp.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg) # Generate fake image

                # --- Apply watermark before G's D pass ---
                x_fake_input_g = x_fake
                if self.apply_watermark:
                    x_fake_input_g = self._apply_watermark_batch(x_fake_input_g, iteration=(i+1), save_prefix=f"g_train_{dataset}_")
                # -----------------------------------------

                out_src, out_cls_all = self.D(x_fake_input_g) # Pass potentially watermarked fake to D
                # Select the classification output corresponding to the target domain of the current dataset
                out_cls = out_cls_all[:, cls_dim_offset : cls_dim_offset + cls_target_dim]
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, cls_label_trg, dataset) # Use target label and dataset

                # Target-to-original domain (Reconstruction)
                # Use the *original* non-watermarked x_fake for reconstruction input
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst)) # Compare reconstruction to original real image

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss[f'G/loss_fake_{dataset}'] = g_loss_fake.item()
                loss[f'G/loss_rec_{dataset}'] = g_loss_rec.item()
                loss[f'G/loss_cls_{dataset}'] = g_loss_cls.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training info.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}], Dataset [{}]".format(et, i+1, self.num_iters, dataset)
                 # Add keys safely
                log_items = {k: v for k, v in loss.items() if v is not None}
                for tag, value in log_items.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard and self.logger:
                    for tag, value in log_items.items():
                        # Log with dataset prefix to distinguish in TensorBoard
                        self.logger.scalar_summary(tag, value, i+1)


            # Translate fixed images for debugging (do this less often maybe).
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed] # Start with original CelebA fixed image

                    # Generate CelebA variations
                    for c_fixed_celeba in c_celeba_list:
                        c_trg_combined = torch.cat([c_fixed_celeba, zero_rafd, mask_celeba], dim=1)
                        generated_img = self.G(x_fixed, c_trg_combined)
                        if self.apply_watermark:
                            generated_img = self._apply_watermark_batch(generated_img, iteration=(i+1), save_prefix="sample_celeba_")
                        x_fake_list.append(generated_img)

                    # Generate RaFD variations
                    for c_fixed_rafd in c_rafd_list:
                        c_trg_combined = torch.cat([zero_celeba, c_fixed_rafd, mask_rafd], dim=1)
                        generated_img = self.G(x_fixed, c_trg_combined)
                        if self.apply_watermark:
                            generated_img = self._apply_watermark_batch(generated_img, iteration=(i+1), save_prefix="sample_rafd_")
                        x_fake_list.append(generated_img)

                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    try:
                         save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                         print('Saved real and fake sampled images into {}...'.format(sample_path))
                    except Exception as e:
                        print(f"❌ Error saving multi-dataset sample image {sample_path}: {e}")


            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                try:
                    torch.save(self.G.state_dict(), G_path)
                    torch.save(self.D.state_dict(), D_path)
                    print('Saved model checkpoints into {}...'.format(self.model_save_dir))
                except Exception as e:
                     print(f"❌ Error saving multi-dataset model checkpoints at iteration {i+1}: {e}")


            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        if not self.restore_model(self.test_iters):
             print("Aborting test due to model loading failure.")
             return

        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader
        else:
             print(f"Error: Invalid dataset '{self.dataset}' for single test.")
             return

        if data_loader is None:
             print(f"Error: Data loader for {self.dataset} is None. Cannot test.")
             return
        if len(data_loader) == 0:
              print(f"Error: Data loader for {self.dataset} is empty. Cannot test.")
              return

        print(f"Starting test for {self.dataset} dataset...")
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                # Ensure c_org is on the correct device before creating labels
                c_trg_list = self.create_labels(c_org.to(self.device), self.c_dim, self.dataset, self.selected_attrs)

                # Translate images.
                x_fake_list = [x_real]
                x_fake_watermarked_list = [x_real] # Keep original real image

                for c_trg in c_trg_list:
                    generated_img = self.G(x_real, c_trg)
                    x_fake_list.append(generated_img) # Add non-watermarked version

                    # --- Apply watermark to test result ---
                    watermarked_img = generated_img # Start with generated
                    if self.apply_watermark:
                         # Pass iteration number (image index) for potential noisy image saving
                         watermarked_img = self._apply_watermark_batch(generated_img, iteration=i, save_prefix=f"test_{self.dataset}_")
                    x_fake_watermarked_list.append(watermarked_img) # Add potentially watermarked version
                    # ------------------------------------

                # Save the non-watermarked translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                try:
                    save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                    print('Saved real and non-watermarked fake images into {}...'.format(result_path))
                except Exception as e:
                    print(f"❌ Error saving non-watermarked result image {result_path}: {e}")

                # --- Save the watermarked translated images (if enabled) ---
                if self.apply_watermark and self.watermark_output_dir:
                    x_concat_wm = torch.cat(x_fake_watermarked_list, dim=3)
                    result_path_wm = os.path.join(self.watermark_output_dir, '{}-wm-images.jpg'.format(i+1))
                    try:
                         os.makedirs(self.watermark_output_dir, exist_ok=True) # Ensure dir exists
                         save_image(self.denorm(x_concat_wm.data.cpu()), result_path_wm, nrow=1, padding=0)
                         print('Saved real and watermarked fake images into {}...'.format(result_path_wm))
                    except Exception as e:
                         print(f"❌ Error saving watermarked result image {result_path_wm}: {e}")
                elif self.apply_watermark:
                     print("Watermarking applied but --watermark_output_dir not set. Skipping saving watermarked images.")
                # --------------------------------------------------------


    def test_multi(self):
        """Translate images using StarGAN trained on multiple datasets."""
        # Load the trained generator.
        if not self.restore_model(self.test_iters):
             print("Aborting multi-test due to model loading failure.")
             return

        if self.celeba_loader is None:
            print("Error: CelebA loader is required for multi-dataset testing.")
            return
        if len(self.celeba_loader) == 0:
             print("Error: CelebA loader is empty. Cannot perform multi-test.")
             return

        print("Starting multi-dataset test (using CelebA as source)...")
        with torch.no_grad():
             # Use CelebA loader as the source for testing domain transfer
            for i, (x_real, c_org) in enumerate(self.celeba_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                # Ensure c_org is on the correct device
                c_org_dev = c_org.to(self.device)
                c_celeba_list = self.create_labels(c_org_dev, self.c_dim, 'CelebA', self.selected_attrs)
                c_rafd_list = self.create_labels(c_org_dev, self.c2_dim, 'RaFD') # Generate RaFD targets

                # Prepare masks and zero vectors
                zero_celeba = torch.zeros(x_real.size(0), self.c_dim).to(self.device)
                zero_rafd = torch.zeros(x_real.size(0), self.c2_dim).to(self.device)
                mask_celeba = self.label2onehot(torch.zeros(x_real.size(0)), 2).to(self.device)
                mask_rafd = self.label2onehot(torch.ones(x_real.size(0)), 2).to(self.device)

                # Translate images.
                x_fake_list = [x_real]              # List for non-watermarked results
                x_fake_watermarked_list = [x_real] # List for watermarked results

                # Generate CelebA variations
                for c_celeba in c_celeba_list:
                    c_trg_combined = torch.cat([c_celeba, zero_rafd, mask_celeba], dim=1)
                    generated_img = self.G(x_real, c_trg_combined)
                    x_fake_list.append(generated_img) # Add non-watermarked

                    watermarked_img = generated_img
                    if self.apply_watermark:
                        watermarked_img = self._apply_watermark_batch(generated_img, iteration=i, save_prefix=f"test_multi_celeba_")
                    x_fake_watermarked_list.append(watermarked_img) # Add potentially watermarked

                # Generate RaFD variations
                for c_rafd in c_rafd_list:
                    c_trg_combined = torch.cat([zero_celeba, c_rafd, mask_rafd], dim=1)
                    generated_img = self.G(x_real, c_trg_combined)
                    x_fake_list.append(generated_img) # Add non-watermarked

                    watermarked_img = generated_img
                    if self.apply_watermark:
                        watermarked_img = self._apply_watermark_batch(generated_img, iteration=i, save_prefix=f"test_multi_rafd_")
                    x_fake_watermarked_list.append(watermarked_img) # Add potentially watermarked


                # Save the non-watermarked translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-multi-images.jpg'.format(i+1))
                try:
                     save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                     print('Saved real and non-watermarked multi-fake images into {}...'.format(result_path))
                except Exception as e:
                     print(f"❌ Error saving non-watermarked multi-result image {result_path}: {e}")

                # --- Save the watermarked translated images (if enabled) ---
                if self.apply_watermark and self.watermark_output_dir:
                    x_concat_wm = torch.cat(x_fake_watermarked_list, dim=3)
                    result_path_wm = os.path.join(self.watermark_output_dir, '{}-multi-wm-images.jpg'.format(i+1))
                    try:
                         os.makedirs(self.watermark_output_dir, exist_ok=True) # Ensure dir exists
                         save_image(self.denorm(x_concat_wm.data.cpu()), result_path_wm, nrow=1, padding=0)
                         print('Saved real and watermarked multi-fake images into {}...'.format(result_path_wm))
                    except Exception as e:
                        print(f"❌ Error saving watermarked multi-result image {result_path_wm}: {e}")
                elif self.apply_watermark:
                     print("Watermarking applied but --watermark_output_dir not set. Skipping saving watermarked images.")
                # --------------------------------------------------------


# --- END OF FILE solver.py ---