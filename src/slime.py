import os

import pytorch_lightning as pl
import torch
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.stable_difusion import StableDiffusion
from src.utils import (
    calculate_iou,
    get_crops_coords,
    generate_distinct_colors,
    get_colored_segmentation,
    get_boundry_and_eroded_mask,
)
import gc
from PIL import Image
import numpy as np

from sklearn.manifold import TSNE

def visualize_text_embedding(embedding_tensor, token_labels):
    """
    Visualizes text embeddings by reducing dimensionality to 2D using t-SNE.
    
    Args:
    - embedding_tensor (torch.Tensor): A tensor of shape (1, n, dim) containing the text embeddings.
    - token_labels (list of str): A list of length n with the token strings corresponding to each embedding.
    
    Returns:
    - None: Displays a plot of the embeddings in 2D.
    """
    
    # Step 1: Remove the batch dimension (if the shape is (1, n, dim))
    embedding_tensor = embedding_tensor.squeeze(0)  # Now it's of shape (n, dim)
    
    # Step 2: Convert the tensor to a numpy array for t-SNE
    embeddings_np = embedding_tensor.detach().cpu().numpy()
    
    # Step 3: Apply t-SNE to reduce dimensionality from 'dim' to 2D
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    # Step 4: Plot the 2D embeddings
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(token_labels):
        x, y = embeddings_2d[i]
        plt.scatter(x, y)
        plt.annotate(label, (x, y), fontsize=12)

    plt.title('2D Visualization of Text Embeddings')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    # plt.savefig(os.path.join(self.debugger_dir, f'cross_att_e{self.current_epoch}_b{batch_idx}.png'))
    plt.savefig(f'tsne.png')



class Slime(pl.LightningModule):
    def __init__(self, config, learning_rate=0.001):
        super().__init__()
        self.counter = 0
        self.val_counter = 0
        self.config = config
        self.save_hyperparameters(config.__dict__)
        self.learning_rate = learning_rate
        self.max_val_iou = 0
        self.val_ious = []

        self.stable_diffusion = StableDiffusion(
            sd_version="2.1",)
        #     attention_layers_to_use=config.attention_layers_to_use,
        # )

        self.checkpoint_dir = None
        if self.config.train:
            self.num_parts = len(self.config.part_names)
        else:
            self.num_parts = (
                len(
                    [
                        file
                        for file in os.listdir(self.config.checkpoint_dir)
                        if file.endswith(".pth")
                    ]
                )
                + 1
            )
            assert (
                self.num_parts > 0
            ), "a folder path should be passed to --checkpoints_dir, which contains the text embeddings!"

        self.prepare_text_embeddings()
        # del self.stable_diffusion.tokenizer
        # del self.stable_diffusion.text_encoder
        torch.cuda.empty_cache()

        self.embeddings_to_optimize = []
        if self.config.train:
            for i in range(1, self.num_parts):
                embedding = self.text_embedding[:, i : i + 1].clone()
                embedding.requires_grad_(True)
                self.embeddings_to_optimize.append(embedding)

        self.token_ids = list(range(self.num_parts))
        self.debugger_dir = os.path.join(self.config.output_dir,"debugger")

    def prepare_text_embeddings(self):
        if self.config.text_prompt is None:
            text_prompt = " ".join(["part" for _ in range(self.num_parts)])
        else:
            text_prompt = self.config.text_prompt
        (
            self.uncond_embedding,
            self.text_embedding,
        ) = self.stable_diffusion.get_text_embeds(text_prompt, "")

        token_labels = [f'token_{i}' for i in range(10)]
        visualize_text_embedding(self.text_embedding, token_labels)
        
    def on_fit_start(self) -> None:
        self.checkpoint_dir = os.path.join(
            self.config.output_dir, "checkpoints", self.logger.log_dir.split("/")[-1]
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.stable_diffusion.setup(self.device)
        self.uncond_embedding, self.text_embedding = self.uncond_embedding.to(
            self.device
        ), self.text_embedding.to(self.device)

    def training_step(self, batch, batch_idx):
        image, mask, img_idx = batch
        num_pixels = torch.zeros(self.num_parts, dtype=torch.int64).to(self.device)
        values, counts = torch.unique(mask, return_counts=True)
        num_pixels[values.type(torch.int64)] = counts.type(torch.int64)
        num_pixels[0] = 0
        pixel_weights = torch.where(num_pixels > 0, num_pixels.sum() / (num_pixels + 1e-6), 0)
        pixel_weights[0] = 1
        pixel_weights[2] = max(pixel_weights).cpu().numpy()*1000
        mask = mask[0]
        text_embedding = torch.cat(
            [
                self.text_embedding[:, 0:1],
                *list(map(lambda x: x.to(self.device), self.embeddings_to_optimize)),
                self.text_embedding[:, 1 + len(self.embeddings_to_optimize) :],
            ],
            dim=1,
        )
        t_embedding = torch.cat([self.uncond_embedding, text_embedding])
        (
            sd_loss,
            _,
            sd_cross_attention_maps2,
            sd_self_attention_maps,
        ) = self.stable_diffusion.train_step(
            t_embedding,
            image,
            t=torch.tensor(self.config.train_t),
            attention_output_size=self.config.train_mask_size,
            token_ids=self.token_ids,
            train=True,
            average_layers=True,
            apply_softmax=False,
        )
        # imgs, all_attention_maps = self.stable_diffusion.prompt_to_img("dog")
        loss1 = F.cross_entropy(
            sd_cross_attention_maps2[None, ...],
            mask[None, ...].type(torch.long),
            weight=pixel_weights,
        )

        VISUALIZE_LOSS1 = True
        if VISUALIZE_LOSS1:
            ca_map = sd_cross_attention_maps2.cpu().detach().numpy()
            N = len(np.unique(mask.cpu().numpy()))
            N_y = int(np.ceil((N+1)/2))
            fig, axes = plt.subplots(2, N_y, figsize=(15, N+1))
            l = loss1.cpu().detach().numpy()
            weights = pixel_weights.cpu().numpy()
            fig.suptitle(f"sd_cross_attention_maps2 - b:{batch_idx}, e:{self.current_epoch}, Loss: {l:.2f}")
            for idx in range(N):
                cax=axes[idx//N_y][idx%N_y].imshow(ca_map[idx])
                axes[idx//N_y][idx%N_y].set_title(f"label {idx}, W:{weights[idx]:.2f}")
                plt.colorbar(cax, cmap = 'jet')
            
            m = mask.cpu().detach().numpy()
            cax=axes[N//N_y][N%N_y].imshow(m)
            axes[N//N_y][N%N_y].set_title(f"mask")
            cmap = plt.get_cmap('Set1', len(np.unique(m))) 
            plt.colorbar(cax, cmap = cmap)


            fig.savefig(os.path.join(self.debugger_dir, f'cross_att_e{self.current_epoch}_b{batch_idx}.png'))

        sd_cross_attention_maps2 = sd_cross_attention_maps2.softmax(dim=0)
        small_sd_cross_attention_maps2 = F.interpolate(
            sd_cross_attention_maps2[None, ...], 64, mode="bilinear"
        )[0]
        self_attention_map = (
            sd_self_attention_maps[None, ...]
            * small_sd_cross_attention_maps2.flatten(1, 2)[..., None, None]
        ).sum(dim=1)
        one_shot_mask = (
            torch.zeros(
                self.num_parts,
                mask.shape[0],
                mask.shape[1],
            )
            .to(mask.device)
            .scatter_(0, mask.unsqueeze(0).type(torch.int64), 1.0)
        )
        loss2 = F.mse_loss(self_attention_map, one_shot_mask) * self.num_parts
        sd_self_attention_maps = None
        small_sd_cross_attention_maps2 = None
        self_attention_map = None

        loss = (
            10* loss1
            + self.config.sd_loss_coef * sd_loss
            + self.config.self_attention_loss_coef * loss2
        )
        print(f"[CA_loss {loss1:.2f}] + [SD_loss_coef {self.config.sd_loss_coef:.2f}] * [SD_loss {sd_loss:.2f}] +"+\
              f"[SA_loss_coef {self.config.self_attention_loss_coef:.2f}] * [SA_loss {loss2:.2f}] = {loss:.2f}")

        self.test_t_embedding = t_embedding
        final_mask = self.get_patched_masks(
            image,
            self.config.train_mask_size,
            img_idx
        )

        sd_cross_attention_maps2 = None
        ious = []
        for idx, part_name in enumerate(self.config.part_names):
            part_mask = torch.where(mask == idx, 1, 0).type(torch.uint8)
            if torch.all(part_mask == 0):
                continue
            iou = calculate_iou(
                torch.where(final_mask == idx, 1, 0).type(torch.uint8), part_mask
            )
            ious.append(iou)
            self.log(f"train {part_name} iou", iou, on_step=True, sync_dist=True)
        mean_iou = sum(ious) / len(ious)

        self.log("loss2", loss2.detach().cpu(), on_step=True, sync_dist=True)
        self.log("sd_loss", sd_loss.detach().cpu(), on_step=True, sync_dist=True)
        self.log("loss1", loss1.detach().cpu(), on_step=True, sync_dist=True)
        self.log("train mean iou", mean_iou.cpu(), on_step=True, sync_dist=True)
        self.log("loss", loss.detach().cpu(), on_step=True, sync_dist=True)

        return loss

    def visualize_cross_attention(self, sd_attention, axes, fig, N_channels, crop_idx):

        # Iterate over the axes and images
        for idx in range(N_channels):
            sd_cross_att = sd_attention[idx].to('cpu').numpy()
            # sd_self_attention_maps
            # sd_cross_attention_maps2
            axes[crop_idx, idx].imshow(sd_cross_att)  # Display image with grayscale colormap
            axes[crop_idx, idx].axis('off')  # Hide axis
            axes[crop_idx, idx].set_title(f'channel {idx}')
        # plt.imshow(sd_cross_attention_maps2[0].to('cpu').numpy(), cmap='gray')  # 'data' is your 2D array or image
        fig.suptitle(f'crop {crop_idx}', fontsize=16, fontweight='bold')

    def get_patched_masks(self, image, output_size, img_idx):
        VISUALIZE = not self.config.train
        if VISUALIZE:
            img_np = image.permute(2,3,1,0).to('cpu').numpy().squeeze()
            plt.imshow(img_np)
            plt.savefig(os.path.join(self.debugger_dir, f'{img_idx}_img.png'))
            plt.close()

        crops_coords = get_crops_coords(
            image.shape[2:],
            self.config.patch_size,
            self.config.num_patchs_per_side,
        )

        final_attention_map = torch.zeros(
            self.num_parts,
            output_size,
            output_size,
        ).to(self.device)

        aux_attention_map = (
            torch.zeros(
                self.num_parts,
                output_size,
                output_size,
                dtype=torch.uint8,
            )
            + 1e-7
        ).to(self.device)

        ratio = 512 // output_size
        mask_patch_size = self.config.patch_size // ratio
        for crop_idx, crop_coord in enumerate(crops_coords):
            y_start, y_end, x_start, x_end = crop_coord
            mask_y_start, mask_y_end, mask_x_start, mask_x_end = (
                y_start // ratio,
                y_end // ratio,
                x_start // ratio,
                x_end // ratio,
            )
            cropped_image = image[:, :, y_start:y_end, x_start:x_end]
            with torch.no_grad():
                (
                    _,
                    _,
                    sd_cross_attention_maps,
                    sd_self_attention_maps,
                ) = self.stable_diffusion.train_step(
                    self.test_t_embedding,
                    cropped_image,
                    t=torch.tensor(self.config.test_t),
                    generate_new_noise=True,
                    attention_output_size=64,
                    token_ids=self.token_ids,
                    train=False,
                )

                if VISUALIZE:
                    if crop_idx==0:
                        N_channels = len(sd_cross_attention_maps)
                        N_crops = len(crops_coords)
                        fig_cross_att, axes = plt.subplots(N_crops, N_channels, figsize=(15, N_channels))

                    self.visualize_cross_attention(sd_cross_attention_maps, axes, fig_cross_att, N_channels, crop_idx)
                    if crop_idx == N_crops-1:
                        fig_cross_att.savefig(os.path.join(self.debugger_dir, f'{img_idx}_sd_cross_att.png'))
                        plt.close(fig_cross_att)



                sd_cross_attention_maps2 = sd_cross_attention_maps.flatten(1, 2)

                max_values = sd_cross_attention_maps2.max(dim=1).values
                min_values = sd_cross_attention_maps2.min(dim=1).values
                passed_indices = torch.where(max_values >= self.config.patch_threshold)[0]
                
                if len(passed_indices) > 0:
                    sd_cross_attention_maps2 = sd_cross_attention_maps2[passed_indices]
                    sd_cross_attention_maps2[0] = torch.where(
                        sd_cross_attention_maps2[0] > sd_cross_attention_maps2[0].mean(),
                        sd_cross_attention_maps2[0],0,)
                    
                    for idx, mask_id in enumerate(passed_indices):
                        avg_self_attention_map = (
                            sd_cross_attention_maps2[idx][..., None, None]
                            * sd_self_attention_maps
                        ).sum(dim=0)

                        avg_self_attention_map = F.interpolate(
                            avg_self_attention_map[None, None, ...],
                            mask_patch_size,
                            mode="bilinear",)[0, 0]

                        avg_self_attention_map_min = avg_self_attention_map.min()
                        avg_self_attention_map_max = avg_self_attention_map.max()
                        coef = (avg_self_attention_map_max - avg_self_attention_map_min) / (max_values[mask_id] - min_values[mask_id])

                        if torch.isnan(coef) or coef == 0:
                            coef = 1e-7
                        final_attention_map[mask_id, mask_y_start:mask_y_end,mask_x_start:mask_x_end,] += \
                              (avg_self_attention_map / coef) + \
                              (min_values[mask_id] - avg_self_attention_map_min / coef)
                        
                        aux_attention_map[
                            mask_id,
                            mask_y_start:mask_y_end,
                            mask_x_start:mask_x_end,] += \
                                torch.ones_like(avg_self_attention_map, dtype=torch.uint8)
                    
                    if VISUALIZE:
                        if crop_idx==0:
                            fig_final_map, axes_final_map = plt.subplots(N_crops, N_channels, figsize=(15, N_channels))
                            fig_aux_map, axes_aux_map = plt.subplots(N_crops, N_channels, figsize=(15, N_channels))

                        self.visualize_cross_attention(final_attention_map, axes_final_map, fig_final_map, N_channels, crop_idx)
                        self.visualize_cross_attention(aux_attention_map, axes_aux_map, fig_aux_map, N_channels, crop_idx)
                        if crop_idx == N_crops-1:
                            fig_final_map.savefig(os.path.join(self.debugger_dir, f'{img_idx}_att_map.png'))
                            fig_aux_map.savefig(os.path.join(self.debugger_dir, f'{img_idx}_aux_att_map.png'))                        
                            plt.close(fig_final_map)
                            plt.close(fig_aux_map)


        final_attention_map /= aux_attention_map
        final_mask = final_attention_map.argmax(0)

        if VISUALIZE:
            fig_final_attention_map, axes = plt.subplots(1, N_channels, figsize=(15, N_channels))
            for idx in range(N_channels):
                cax=axes[idx].imshow(final_attention_map[idx].to('cpu').numpy())
                fig_final_attention_map.colorbar(cax)
            fig_final_attention_map.savefig(os.path.join(self.debugger_dir, f'{img_idx}_att_map_norm.png'))

            fig = plt.figure(figsize=(6, 6))  # Set figure size
            ax = fig.add_subplot(111)  # 1 row, 1 column, 1st subplot
            cax = ax.imshow(final_mask.to('cpu').numpy())
            fig.colorbar(cax)
            plt.savefig(os.path.join(self.debugger_dir, f'{img_idx}_final_mask.png'))

        return final_mask

    def on_validation_start(self):
        text_embedding = torch.cat(
            [
                self.text_embedding[:, 0:1],
                *list(
                    map(
                        lambda x: x.to(self.device).detach(),
                        self.embeddings_to_optimize,
                    )
                ),
                self.text_embedding[:, 1 + len(self.embeddings_to_optimize) :],
            ],
            dim=1,
        )
        self.test_t_embedding = torch.cat([self.uncond_embedding, text_embedding])

    def on_validation_epoch_start(self):
        self.val_ious = []

    def validation_step(self, batch, batch_idx):
        image, mask, img_idx = batch
        mask = mask[0]
        final_mask = self.get_patched_masks(
            image,
            self.config.test_mask_size,
            img_idx
        )
        ious = []
        for idx, part_name in enumerate(self.config.part_names):
            part_mask = torch.where(mask == idx, 1, 0).type(torch.uint8)
            if torch.all(part_mask == 0):
                continue
            iou = calculate_iou(
                torch.where(final_mask == idx, 1, 0).type(torch.uint8), part_mask
            )
            ious.append(iou)
            self.log(f"val {part_name} iou", iou.cpu(), on_step=True, sync_dist=True)
        mean_iou = sum(ious) / len(ious)
        self.val_ious.append(mean_iou)
        self.log("val mean iou", mean_iou.cpu(), on_step=True, sync_dist=True)
        return torch.tensor(0.0)

    def on_validation_epoch_end(self):
        epoch_mean_iou = sum(self.val_ious) / len(self.val_ious)
        if epoch_mean_iou >= self.max_val_iou:
            self.max_val_iou = epoch_mean_iou
            for i, embedding in enumerate(self.embeddings_to_optimize):
                torch.save(
                    embedding,
                    os.path.join(self.checkpoint_dir, f"embedding_{i}.pth"),
                )
        torch.cuda.empty_cache()
        gc.collect()

    def on_test_start(self) -> None:
        self.stable_diffusion.setup(self.device)
        uncond_embedding, text_embedding = self.uncond_embedding.to(
            self.device
        ), self.text_embedding.to(self.device)
        self.stable_diffusion.change_hooks(
            attention_layers_to_use=self.config.attention_layers_to_use
        )  # detach attention layers
        embeddings_to_optimize = []
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.config.checkpoint_dir
        for i in range(self.num_parts - 1):
            embedding = torch.load(
                os.path.join(self.checkpoint_dir, f"embedding_{i}.pth")
            )
            embeddings_to_optimize.append(embedding)
        text_embedding = torch.cat(
            [
                text_embedding[:, 0:1],
                *list(map(lambda x: x.to(self.device), embeddings_to_optimize)),
                text_embedding[:, 1 + len(embeddings_to_optimize) :],
            ],
            dim=1,
        )
        self.test_t_embedding = torch.cat([uncond_embedding, text_embedding])
        if self.config.save_test_predictions:
            self.distinct_colors = generate_distinct_colors(self.num_parts - 1)
            self.test_results_dir = os.path.join(
                self.config.output_dir,
                "test_results",
                self.logger.log_dir.split("/")[-1],
            )
            os.makedirs(self.test_results_dir, exist_ok=True)

    def test_step(self, batch, batch_idx):
        image, mask, img_idx = batch
        img_idx = img_idx.to('cpu').numpy()[0]
        mask_provided = not torch.all(mask == 0)
        mask = mask[0]
        final_mask = self.get_patched_masks(
            image,
            self.config.test_mask_size,
            img_idx
        )
        if self.config.save_test_predictions:
            eroded_final_mask, final_mask_boundary = get_boundry_and_eroded_mask(
                final_mask.cpu()
            )
            colored_image = get_colored_segmentation(
                torch.tensor(eroded_final_mask),
                torch.tensor(final_mask_boundary),
                image[0].cpu(),
                self.distinct_colors,
            )
            for i in range(image.shape[0]):
                output_name = f"{batch_idx * image.shape[0] + i+img_idx}.png"
                output_name = os.path.join(self.test_results_dir, output_name)
                print(output_name)
                Image.fromarray((255 * colored_image).type(torch.uint8).numpy()).save(output_name)

        if mask_provided:
            for idx, part_name in enumerate(self.config.part_names):
                part_mask = torch.where(mask == idx, 1, 0).type(torch.uint8)
                if torch.all(part_mask == 0):
                    continue
                iou = calculate_iou(
                    torch.where(final_mask == idx, 1, 0).type(torch.uint8), part_mask
                )
                # self.ious[part_name].append(iou.cpu())
                self.log(
                    f"test {part_name} iou", iou.cpu(), on_step=True, sync_dist=True
                )

        return torch.tensor(0.0)

    def on_test_end(self) -> None:
        print("max val mean iou: ", self.max_val_iou)

    def configure_optimizers(self):
        parameters = [{"params": self.embeddings_to_optimize, "lr": self.config.lr}]
        optimizer = getattr(optim, self.config.optimizer)(
            parameters,
            lr=self.config.lr,
        )
        return optimizer
