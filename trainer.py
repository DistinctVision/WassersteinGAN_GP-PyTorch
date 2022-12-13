# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging
from typing import Optional
import math
import os
import yaml
from pathlib import Path
import numpy as np

import torch.utils.data
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm

import wgangp_pytorch.models as models
from wgangp_pytorch.models import discriminator
from wgangp_pytorch.utils import calculate_gradient_penalty
from wgangp_pytorch.utils import init_torch_seeds
from wgangp_pytorch.utils import select_device
from wgangp_pytorch.utils import weights_init
from dataset import FmaDatasetReader, MlAudioDataset, MlAudioBatchCollector, DataNormalizer, save_spectogram_as_fig

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


class Trainer(object):
    def __init__(self, args, data_slice: Optional[int] = None):
        self.args = args

        # Set random initialization seed, easy to reproduce.
        init_torch_seeds(args.manualSeed)

        logger.info("Load training dataset")

        self.config = yaml.safe_load(open(Path('data') / 'data.yaml', 'r'))
        batch_size = int(self.config['training']['batch_size'])
        self.fma_reader = FmaDatasetReader(self.config)
        if data_slice is not None:
            self.fma_reader.slice_tracks(data_slice)

        self.ml_audio_dataset = MlAudioDataset(self.fma_reader, self.config)
        self.ml_audio_batch_collector = MlAudioBatchCollector(self.ml_audio_dataset, batch_size)
        self.data_normalizer = DataNormalizer(self.config['datasets']['ranges']['amplitude'],
                                              self.config['datasets']['ranges']['phase'])

        # Construct network architecture model of generator and discriminator.
        self.device = select_device(args.device, batch_size=batch_size)
        if args.pretrained:
            logger.info(f"Using pre-trained model `{args.arch}`")
            self.generator = models.__dict__[args.arch](pretrained=True).to(self.device)
        else:
            logger.info(f"Creating model `{args.arch}`")
            self.generator = models.__dict__[args.arch]().to(self.device)
        logger.info(f"Creating discriminator model")
        self.discriminator = discriminator().to(self.device)

        self.generator = self.generator.apply(weights_init)
        self.discriminator = self.discriminator.apply(weights_init)

        # Parameters of pre training model.
        self.start_epoch = math.floor(args.start_iter / len(self.ml_audio_batch_collector))
        self.epochs = math.ceil(args.iters / len(self.ml_audio_batch_collector))
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

        logger.info(f"Model training parameters:\n"
                    f"\tIters is {int(args.iters)}\n"
                    f"\tEpoch is {int(self.epochs)}\n"
                    f"\tOptimizer Adam\n"
                    f"\tBetas is (0.5, 0.999)\n"
                    f"\tLearning rate {args.lr}")

    def run(self):
        args = self.args

        # Load pre training model.
        if args.netD != "":
            self.discriminator.load_state_dict(torch.load(args.netD))
        if args.netG != "":
            self.generator.load_state_dict(torch.load(args.netG))

        self.discriminator.train()
        self.generator.train()

        # Start train PSNR model.
        logger.info(f"Training for {self.epochs} epochs")

        fixed_noise = torch.randn(args.batch_size, 100, 1, 1, device=self.device)

        for epoch in range(self.start_epoch, self.epochs):
            progress_bar = tqdm(enumerate(self.ml_audio_batch_collector.iter_level(2)),
                                total=len(self.ml_audio_batch_collector))
            for i, data in progress_bar:

                real_images_raw = data[0].astype(np.float32)
                real_images_raw = real_images_raw[:, :, :64, :64]
                real_images_raw = self.data_normalizer.normalize(real_images_raw)
                real_images = torch.from_numpy(real_images_raw).to(self.device)

                batch_size = real_images.size(0)
                noise = torch.randn(batch_size, 100, 1, 1, device=self.device)

                ##############################################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ##############################################
                # Set discriminator gradients to zero.
                self.discriminator.zero_grad()

                # Train with real
                real_output = self.discriminator(real_images)
                errD_real = torch.mean(real_output)
                D_x = real_output.mean().item()

                # Generate fake image batch with G
                fake_images = self.generator(noise)

                # Train with fake
                fake_output = self.discriminator(fake_images.detach())
                errD_fake = torch.mean(fake_output)
                D_G_z1 = fake_output.mean().item()

                # Calculate W-div gradient penalty
                gradient_penalty = calculate_gradient_penalty(self.discriminator,
                                                              real_images.data, fake_images.data,
                                                              self.device)

                # Add the gradients from the all-real and all-fake batches
                errD = -errD_real + errD_fake + gradient_penalty * 10
                errD.backward()
                # Update D
                self.optimizer_d.step()

                # Train the generator every n_critic iterations
                if (i + 1) % args.n_critic == 0:
                    ##############################################
                    # (2) Update G network: maximize log(D(G(z)))
                    ##############################################
                    # Set generator gradients to zero
                    self.generator.zero_grad()

                    # Generate fake image batch with G
                    fake_images = self.generator(noise)
                    fake_output = self.discriminator(fake_images)
                    errG = -torch.mean(fake_output)
                    D_G_z2 = fake_output.mean().item()
                    errG.backward()
                    self.optimizer_g.step()

                    progress_bar.set_description(f"[{epoch + 1}/{self.epochs}][{i + 1}/{len(self.ml_audio_batch_collector)}] "
                                                 f"Loss_D: {errD.item():.6f} Loss_G: {errG.item():.6f} "
                                                 f"D(x): {D_x:.6f} D(G(z)): {D_G_z1:.6f}/{D_G_z2:.6f}")

                iters = i + epoch * len(self.ml_audio_batch_collector) + 1
                # The image is saved every 1000 epoch.
                if iters % 1000 == 0:
                    for r_i, real_image_raw in enumerate(real_images_raw):
                        save_spectogram_as_fig(real_image_raw, os.path.join("output",
                                                                            f"real_samples_{iters}_{r_i}.png"), 2 ** 3)
                    fake = self.generator(fixed_noise)
                    fake[fake < -1] = -1
                    fake[fake > 1] = 1
                    fake = fake.detach().cpu().numpy()
                    for f_i, fake_image in enumerate(fake):
                        save_spectogram_as_fig(fake_image, os.path.join("output",
                                                                        f"fake_samples_{iters}_{f_i}.png"), 2 ** 3)

                    # do checkpointing
                    torch.save(self.generator.state_dict(), f"weights/{args.arch}_G_iter_{iters}.pth")
                    torch.save(self.discriminator.state_dict(), f"weights/{args.arch}_D_iter_{iters}.pth")

                if iters == int(args.iters):  # If the iteration is reached, exit.
                    break
