# from src.arguments import init_args
# from src.slime import Slime
# from sklearn.manifold import TSNE

import torch
from torchvision import transforms

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel
from diffusers import AutoencoderKL
from diffusers import DDPMScheduler
from diffusers import UNet2DModel
from diffusers import StableDiffusionPipeline
from diffusers import PNDMScheduler
import matplotlib.pyplot as plt
import gc
import numpy as np

# Define device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.memory._record_memory_history()

# Load models
sd_version = "1.4"
if sd_version == "2.1":
    sd_model_key = "stabilityai/stable-diffusion-2-1-base"
    # model_key = "stabilityai/stable-diffusion-2-base"
elif sd_version == "1.5":
    sd_model_key = "runwayml/stable-diffusion-v1-5"
elif sd_version == "1.4":
    sd_model_key = "CompVis/stable-diffusion-v1-4"

# UNet2DConditionModel
# Model Name	                    Encoder Hidden States Size	Hidden Dimension
# CompVis/stable-diffusion-v1-4	        77	                        512
# stabilityai/stable-diffusion-2-base	77	                        768
# stabilityai/stable-diffusion-2-1	    77	                        768

#CLIPTokenizer
# Model Name	                Sequence Length	 Hidden Size	Output Shape
# openai/clip-vit-base-patch32	    77	            512	        (batch_size, 77, 512)
# openai/clip-vit-large-patch14	    77	            768	        (batch_size, 77, 768)


# CLIP_dim = "openai/clip-vit-base-patch32"
CLIP_dim = "openai/clip-vit-large-patch14"
text_encoder = CLIPTextModel.from_pretrained(CLIP_dim).to(device)
tokenizer = CLIPTokenizer.from_pretrained(CLIP_dim)
unet = UNet2DConditionModel.from_pretrained(sd_model_key, subfolder="unet").to(device).half()
# unet = UNet2DModel.from_pretrained("google/ddpm-cat-256").to(device)
# unet = UNet2DModel.from_pretrained("google/ddpm-celebahq-256").to(device)

# "google/ddpm-celebahq-256"
vae = AutoencoderKL.from_pretrained(sd_model_key, subfolder="vae").to(device).half()
# scheduler = DDPMScheduler(num_train_timesteps=1000)
if 0:
    scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256", num_train_timesteps=2000)
else:
    # scheduler = PNDMScheduler.from_config(unet_global.config)
    scheduler = PNDMScheduler(\
        beta_start=0.00085,  # Starting value of beta
        beta_end=0.012,      # End value of beta
        beta_schedule="scaled_linear",  # Type of beta schedule
        skip_prk_steps=True,  # Optional, whether to skip the Pseudo-Reversible Kinetics steps,
        steps_offset = 1
    )

pipe = StableDiffusionPipeline.from_pretrained(sd_model_key, torch_dtype=torch.float16)
pipe = pipe.to(device)


# Function 1: Add noise to an existing image
def add_noise(image_tensor, noise_level):
    """
    Adds Gaussian noise to an image latent representation.
    
    Args:
    - image_tensor (torch.Tensor): The latent image tensor (encoded by VAE).
    - noise_level (float): The level of noise to add.
    
    Returns:
    - noisy_image (torch.Tensor): The noisy image tensor.
    """
    # Generate Gaussian noise
    noise = torch.randn_like(image_tensor).to(device)
    
    # Add noise to the image tensor
    noisy_image = image_tensor + noise_level * noise
    
    return noisy_image

def plot_latent_space(latent, file_name):
    if 0:
        from PIL import Image
        image_path = r"/home/benf22/repos/slime/SLiMe/src/aaa.jpg"
        image_array = Image.open(image_path)

        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to the expected input size
            transforms.ToTensor(),         # Convert to tensor and scale to [0, 1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
        ])

        image_tensor = transform(image_array).unsqueeze(0).to('cuda')  # Add batch dimension
        with torch.no_grad():  # Disable gradient calculation
            # emmbeded_obj = vae.encode(image_tensor)
            # emmbeded = emmbeded_obj.latent_dist.sample()
        
            emmbeded_ = torch.randn((1,4,64,64)).to('cuda')
            emmbeded_2 = (emmbeded_-emmbeded_.min())/(emmbeded_.max() - emmbeded_.min())
            reproduced_image = vae.decode(emmbeded_2).sample
        reproduced_image = (reproduced_image/2 + 0.5).clamp(0,1)
        reproduced_np = reproduced_image.squeeze().permute(1,2,0).cpu().detach().numpy()
        reproduced_np_norm = (reproduced_np*255).astype(np.uint8)
        fig, axs = plt.subplots(1, 3, figsize=(14, 6))

        # axs[0].imshow(image_array)
        # axs[1].imshow((image_tensor/2 +0.5).clamp(0,1).permute(2,3,1,0).squeeze().cpu().detach().numpy())
        axs[2].imshow(reproduced_np_norm)
        fig.savefig('test_image')

    N = latent.shape[0]
    fig, axs = plt.subplots(1, N, figsize=(14, 6))

    with torch.no_grad():
        latent = 1 / vae.config.scaling_factor * latent

        intermidiate_image = vae.decode(latent).sample # .clamp(0, 1)
    # if intermidiate_image.min() < 0 or intermidiate_image.max() > 1:
        # decoded = (intermidiate_image - intermidiate_image.min()) / (intermidiate_image.max() - intermidiate_image.min())  # Normalize to [0, 1]
    decoded = ((intermidiate_image + 1 ) / 2).clamp(0, 1)

    # generated_image_np = decoded.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze()
    generated_image_np = decoded.detach().permute(0,2,3,1).cpu().numpy()

    # scaled_img = (generated_image_np*255).astype(np.uint8)
    scaled_img = generated_image_np

    for i in range(N):
        ax = axs[i] if type(axs)==np.ndarray else axs
        ax.imshow(scaled_img[i].astype(np.float32))
    # axs[1].imshow(scaled_img)

    # fig.tight_layout() 
    fig.savefig(file_name)
    
def show_words_on_emmbeding_space():
    pass
    # tsne = TSNE(n_components=2, random_state=42, perplexity = 2)  # Set random_state for reproducibility

    # words = ["dog", "cat", "mouse", "bear", "cow", "sheep", "ant", "bird",\
    #         "plus", "minus", "limit", "function", "algebra", "math", "integral"]
    # all_emmbedings = np.zeros((0, 768*3))
    # for this_word in words:
    #     inputs_ = tokenizer(this_word, return_tensors="pt").to(device)
    #     with torch.no_grad():
    #         text_embeddings_ = text_encoder(input_ids=inputs_.input_ids).last_hidden_state

    #     text_embeddings_np = text_embeddings_.view(-1,1).permute(1,0).cpu().numpy()
    #     # all_emmbedings.append(text_embeddings_np)
    #     all_emmbedings = np.vstack((all_emmbedings, text_embeddings_np))

    # twoD = tsne.fit_transform(all_emmbedings)

    # fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # axs[0].scatter(twoD[:, 0], twoD[:, 1], alpha=0.7)
    # for i, word in enumerate(words):
    #     axs[0].text(twoD[i, 0], twoD[i, 1], word)
    # axs[0].grid()
    # # plt.imshow(scaled_img)
    # fig.savefig( f'TSNE_text_emmbeding.png')


# Function 2: Remove noise to generate an image conditioned on a text prompt
def denoise_image(text_prompt, num_steps=100):
    """
    Performs the backward pass to denoise the image using UNet and text conditioning.
    
    Args:
    - noisy_latent (torch.Tensor): The noisy latent image.
    - text_prompt (str): The text prompt to condition the denoising process.
    - num_steps (int): Number of timesteps to perform denoising.
    
    Returns:
    - final_image (torch.Tensor): The generated image.
    """

    guidance_scale = 10 #7.5
    do_classifier_free_guidance = guidance_scale > 1.0


    # Step 1: Tokenize the text prompt
    ##################################
    if 0:
        inputs = tokenizer(text_prompt, return_tensors="pt", padding="max_length",max_length=tokenizer.model_max_length, truncation=True).to(device) 

        with torch.no_grad():
            # text_embeddings = text_encoder(input_ids=inputs.input_ids).last_hidden_state
            text_embeddings = text_encoder(input_ids=inputs.input_ids, attention_mask=None)[0]
    else:
        text_embeddings = pipe._encode_prompt(\
                                text_prompt,
                                device,
                                num_images_per_prompt = 1,
                                do_classifier_free_guidance = do_classifier_free_guidance,
                                negative_prompt = None,
                                prompt_embeds=None,
                                negative_prompt_embeds=None)
    

    # Step 2: Inits before the loop
    ##################################
    latent_input = torch.randn(1, unet.config.in_channels, unet.config.sample_size, unet.config.sample_size, dtype = text_embeddings.dtype ).to(device)  # Example latent size for demonstration
    scheduler.set_timesteps(num_steps, device=device)

    latents = []
    all_predicted_noise = np.zeros((0, unet.config.in_channels *unet.config.sample_size* unet.config.sample_size))
    # tsne1 = TSNE(n_components=2, random_state=42, perplexity = 2)  # Set random_state for reproducibility
    # tsne2 = TSNE(n_components=2, random_state=42, perplexity = 15)  # Set random_state for reproducibility

    # Step 3: Iteratively denoise the image using UNet
    ##################################
    for i, timestep in enumerate(scheduler.timesteps):
        print(f"starting iteration = {i}, timestep = {timestep}")
        # Get the scheduler's timestep (scaled between 0 and 1)
        # timestep = torch.tensor([t / num_steps], dtype=torch.float32, device=device)
        # timestep = torch.tensor([t], dtype=torch.float32, device=device)
        # timestep = scheduler.config.num_train_timesteps - t*20 - 1

        # Step 3.1 - Predict the noise using the UNet model, conditioned on the text embeddings
        ##################################
        latent_unet_input = torch.cat([latent_input] * 2) if do_classifier_free_guidance else latent_input
        latent_unet_input = scheduler.scale_model_input(latent_unet_input, timestep)

        with torch.no_grad():
            predicted_noise = unet(latent_unet_input, timestep, encoder_hidden_states=text_embeddings).sample
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = predicted_noise.chunk(2)
            predicted_noise = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # predicted_noise_np = predicted_noise.view(-1,1).permute(1,0).cpu().numpy()
        # all_emmbedings.append(text_embeddings_np)
        # all_predicted_noise = np.vstack((all_predicted_noise, predicted_noise_np))
        plot_latent_space(predicted_noise, f'predicted_noise_{i}.png')

        # Step 3.2 - Denoise the latent representation (progressively remove noise)
        ##################################
        latent_output_t  = scheduler.step(predicted_noise, timestep, latent_input).prev_sample #pred_original_sample
        latent_output_T0 = scheduler.step(predicted_noise, torch.tensor([0]).to("cuda"), latent_input).prev_sample #pred_original_sample

        # latents.append(latent_input)

        if i%10==0:
            plot_latent_space(latent_output_t, f'intermidiate_image_{i}.png')
            plot_latent_space(latent_output_T0, f'intermidiate_image_{i}_T0predicted.png')

            #########
            # latents_np = latents.detach().numpy()

            # twoD = tsne1.fit_transform(latents_np)

            # fig, axs = plt.subplots(1, 2, figsize=(14, 6))
            # axs[0].scatter(twoD[:, 0], twoD[:, 1], alpha=0.7)
            # axs[0].grid()

        latent_input = latent_output_t

        # torch.cuda.empty_cache()
        # gc.collect()
        # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")

    ### Step 4: Visualize the final image
    # predicted_noise_2d = tsne2.fit_transform(all_predicted_noise)
    # plt.figure(figsize=(8, 6))
    # plt.scatter(predicted_noise_2d[:, 0], predicted_noise_2d[:, 1], alpha=0.7)
    # txt = [str(x) for x in range(len(predicted_noise_2d))]
    # for i, word in enumerate(txt):
    #     plt.text(predicted_noise_2d[i, 0], predicted_noise_2d[i, 1], word)
    # plt.grid()
    # plt.savefig( f'predicted_noise_2d.png')

    # Step 3: Decode the latent representation back into an image using the VAE decoder
    final_image = vae.decode(latent_output_t).sample
    
    return final_image


def genertae_image(text_prompt):

    # Perform backward pass (denoise) to generate image conditioned on the text prompt
    generated_image = denoise_image(text_prompt)
    # generated_image = model.stable_diffusion.denoise_image(image_tensor, text_prompt, num_steps=50)

    # Visualize the final image

def main():
    # config = init_args()

    # model = Slime(config=config)
 
    # Example Usage:
    # Assume `image_tensor` is an image encoded in the latent space, which can be encoded using VAE.
    # Example latent image tensor
    # text_prompt = "A beautiful landscape with mountains and a river."
    text_prompt = "rabbit on a chair"
    torch.manual_seed(torch.randint(0, 10000, (1,)).item())
    # image_tensor = torch.zeros(1, 4, 64, 64).to(device)  # Example latent size for demonstration

    # Add noise to the image
    # noisy_image = add_noise(image_tensor, noise_level=0.8)

    if 1:
        genertae_image(text_prompt)
    else:
        # pipe = StableDiffusionPipeline.from_pretrained(sd_model_key, torch_dtype=torch.float16)
        # pipe = pipe.to(device)
        image = pipe(text_prompt).images[0]
        plt.imshow(image)
        # axs[1].imshow(scaled_img)

        # fig.tight_layout() 
        plt.savefig('pipeline_generator.png')

    if 0:
        show_words_on_emmbeding_space()



if __name__ == "__main__":
    main()
