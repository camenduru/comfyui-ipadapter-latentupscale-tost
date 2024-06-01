import os, json, requests, runpod

import  sys
sys.path.append('/content/TotoroUI/IPAdapter')

import torch
import numpy as np
from PIL import Image
import totoro
import scipy
from latent_resizer import LatentResizer
import gc

import nodes, IPAdapterPlus
from totoro import model_management

discord_token = os.getenv('com_camenduru_discord_token')
web_uri = os.getenv('com_camenduru_web_uri')
web_token = os.getenv('com_camenduru_web_token')

def download_file(url, save_dir='/content/TotoroUI/models'):
    os.makedirs(save_dir, exist_ok=True)
    file_name = url.split('/')[-1]
    file_path = os.path.join(save_dir, file_name)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

with torch.no_grad():
    device = model_management.get_torch_device()
    vae_device = model_management.vae_offload_device()
    model_up = LatentResizer.load_model('/content/TotoroUI/models/sd15_resizer.pt', device, torch.float16)
    model_patcher, clip, vae, clipvision = totoro.sd.load_checkpoint_guess_config("/content/TotoroUI/models/dreamshaper_8.safetensors", output_vae=True, output_clip=True, embedding_directory=None)
    IPAdapterPlus_model = IPAdapterPlus.IPAdapterUnifiedLoader().load_models(model_patcher, 'PLUS (high strength)', lora_strength=0.0, provider="CPU", ipadapter=None)

def upscale(latent, upscale, model, device, vae_device):
  samples = latent.to(device=device, dtype=torch.float16)
  model.to(device=device)
  latent_out = (model(0.13025 * samples, scale=upscale) / 0.13025)
  latent_out = latent_out.to(device="cpu")
  model.to(device=vae_device)
  return ({"samples": latent_out},)

# mask_from_colors() and conditioning_combine_multiple() from https://github.com/cubiq/ComfyUI_essentials/blob/main/essentials.py
def mask_from_colors(image, threshold_r, threshold_g, threshold_b, remove_isolated_pixels, fill_holes):
    red = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] < threshold_b)).float()
    green = ((image[..., 0] < threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] < threshold_b)).float()
    blue = ((image[..., 0] < threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] >= 1-threshold_b)).float()
    cyan = ((image[..., 0] < threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] >= 1-threshold_b)).float()
    magenta = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] > 1-threshold_b)).float()
    yellow = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] < threshold_b)).float()
    black = ((image[..., 0] <= threshold_r) & (image[..., 1] <= threshold_g) & (image[..., 2] <= threshold_b)).float()
    white = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] >= 1-threshold_b)).float()
    if remove_isolated_pixels > 0 or fill_holes:
        colors = [red, green, blue, cyan, magenta, yellow, black, white]
        color_names = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'white']
        processed_colors = {}
        for color_name, color in zip(color_names, colors):
            color = color.cpu().numpy()
            masks = []
            for i in range(image.shape[0]):
                mask = color[i]
                if remove_isolated_pixels > 0:
                    mask = scipy.ndimage.binary_opening(mask, structure=np.ones((remove_isolated_pixels, remove_isolated_pixels)))
                if fill_holes:
                    mask = scipy.ndimage.binary_fill_holes(mask)
                mask = torch.from_numpy(mask)
                masks.append(mask)
            processed_colors[color_name] = torch.stack(masks, dim=0).float()
        red = processed_colors['red']
        green = processed_colors['green']
        blue = processed_colors['blue']
        cyan = processed_colors['cyan']
        magenta = processed_colors['magenta']
        yellow = processed_colors['yellow']
        black = processed_colors['black']
        white = processed_colors['white']
        del colors, processed_colors
    return (red, green, blue, cyan, magenta, yellow, black, white,)

def conditioning_combine_multiple(conditioning_1, conditioning_2, conditioning_3=None, conditioning_4=None, conditioning_5=None):
    c = conditioning_1 + conditioning_2
    if conditioning_3 is not None:
        c += conditioning_3
    if conditioning_4 is not None:
        c += conditioning_4
    if conditioning_5 is not None:
        c += conditioning_5
    return (c,)

@torch.inference_mode()
def generate(input):
    values = input["input"]

    red_part = values['red_part']
    red_positive_prompt = values['red_positive_prompt']
    red_negative_prompt = values['red_negative_prompt']
    red_threshold = values['red_threshold']
    red_image_weight = values['red_image_weight']
    red_prompt_weight = values['red_prompt_weight']
    green_part = values['green_part']
    green_positive_prompt = values['green_positive_prompt']
    green_negative_prompt = values['green_negative_prompt']
    green_threshold = values['green_threshold']
    green_image_weight = values['green_image_weight']
    green_prompt_weight = values['green_prompt_weight']
    black_part = values['black_part']
    black_positive_prompt = values['black_positive_prompt']
    black_negative_prompt = values['black_negative_prompt']
    black_threshold = values['black_threshold']
    black_image_weight = values['black_image_weight']
    black_prompt_weight = values['black_prompt_weight']
    color_mask = values['color_mask']
    seed = values['seed']
    steps = values['steps']
    cfg = values['cfg']
    sampler_name = values['sampler_name']
    scheduler = values['scheduler']
    width = values['width']
    height = values['height']
    image_denoise = values['image_denoise']
    latent_upscale = values['latent_upscale']
    latent_upscale_size = values['latent_upscale_size']
    latent_upscale_denoise = values['latent_upscale_denoise']

    red_part = download_file(red_part)
    green_part = download_file(green_part)
    black_part = download_file(black_part)
    color_mask = download_file(color_mask)

    output1_image, output1_mask = nodes.LoadImage().load_image(str(red_part))
    output2_image, output2_mask = nodes.LoadImage().load_image(str(green_part))
    output3_image, output3_mask = nodes.LoadImage().load_image(str(black_part))
    color_image, color_mask = nodes.LoadImage().load_image(str(color_mask))
    red, green, blue, cyan, magenta, yellow, black, white = mask_from_colors(image=color_image, threshold_r=red_threshold, threshold_g=green_threshold, threshold_b=black_threshold, remove_isolated_pixels=0, fill_holes=False)

    tokens_1 = clip.tokenize(red_positive_prompt)
    cond_1, pooled_1 = clip.encode_from_tokens(tokens_1, return_pooled=True)
    cond_1 = [[cond_1, {"pooled_output": pooled_1}]]
    n_tokens_1 = clip.tokenize(red_negative_prompt)
    n_cond_1, n_pooled_1 = clip.encode_from_tokens(n_tokens_1, return_pooled=True)
    n_cond_1 = [[n_cond_1, {"pooled_output": n_pooled_1}]]
    params_1, positive_1, negative_1 = IPAdapterPlus.IPAdapterRegionalConditioning().conditioning(output1_image, image_weight=red_image_weight, prompt_weight=red_prompt_weight, weight_type='linear', start_at=0.0, end_at=1.0, mask=red, positive=cond_1, negative=n_cond_1)

    tokens_2 = clip.tokenize(green_positive_prompt)
    cond_2, pooled_2 = clip.encode_from_tokens(tokens_2, return_pooled=True)
    cond_2 = [[cond_2, {"pooled_output": pooled_2}]]
    n_tokens_2 = clip.tokenize(green_negative_prompt)
    n_cond_2, n_pooled_2 = clip.encode_from_tokens(n_tokens_2, return_pooled=True)
    n_cond_2 = [[n_cond_2, {"pooled_output": n_pooled_2}]]
    params_2, positive_2, negative_2 = IPAdapterPlus.IPAdapterRegionalConditioning().conditioning(output2_image, image_weight=green_image_weight, prompt_weight=green_prompt_weight, weight_type='linear', start_at=0.0, end_at=1.0, mask=green, positive=cond_2, negative=n_cond_2)

    tokens_3 = clip.tokenize(black_positive_prompt)
    cond_3, pooled_3 = clip.encode_from_tokens(tokens_3, return_pooled=True)
    cond_3 = [[cond_3, {"pooled_output": pooled_3}]]
    n_tokens_3 = clip.tokenize(black_negative_prompt)
    n_cond_3, n_pooled_3 = clip.encode_from_tokens(n_tokens_3, return_pooled=True)
    n_cond_3 = [[n_cond_3, {"pooled_output": n_pooled_3}]]
    params_3, positive_3, negative_3 = IPAdapterPlus.IPAdapterRegionalConditioning().conditioning(output3_image, image_weight=black_image_weight, prompt_weight=black_prompt_weight, weight_type='linear', start_at=0.0, end_at=1.0, mask=black, positive=None, negative=None)
    positive = conditioning_combine_multiple(conditioning_1=positive_1, conditioning_2=positive_2, conditioning_3=cond_3)
    negative = conditioning_combine_multiple(conditioning_1=negative_1, conditioning_2=negative_2, conditioning_3=n_cond_3)
    ipadapter_params = IPAdapterPlus.IPAdapterCombineParams().combine(params_1=params_1, params_2=params_2, params_3=params_3)
    ip_model_patcher = IPAdapterPlus.IPAdapterAdvanced().apply_ipadapter(IPAdapterPlus_model[0], IPAdapterPlus_model[1], start_at=0.0, end_at=1.0, weight=1.0, weight_style=1.0, weight_composition=1.0, expand_style=False, weight_type="linear", combine_embeds="concat", embeds_scaling='V only', ipadapter_params=ipadapter_params[0])
    latent = {"samples":torch.zeros([1, 4, height // 8, width // 8])}
    sample = nodes.common_ksampler(
        model=ip_model_patcher[0], 
        seed=seed, 
        steps=steps, 
        cfg=cfg, 
        sampler_name=sampler_name, 
        scheduler=scheduler, 
        positive=positive[0], 
        negative=negative[0],
        latent=latent, 
        denoise=image_denoise)

    if latent_upscale:
        with torch.inference_mode():
            sample = sample[0]["samples"].to(torch.float16)
            vae.first_stage_model.cuda()
            decoded = vae.decode_tiled(sample).detach()
        print(torch.cuda.memory_cached(device=None))
        model_management.cleanup_models()
        gc.collect()
        model_management.soft_empty_cache()
        print(torch.cuda.memory_cached(device=None))
        final_image = Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0])
        final_image.save("/content/TotoroUI/models/final_image.png")
        latent_up = upscale(sample, latent_upscale_size, model_up, device, vae_device)
        sample_up = nodes.common_ksampler(
            model=ip_model_patcher[0], 
            seed=seed,
            steps=steps,
            cfg=cfg, 
            sampler_name=sampler_name, 
            scheduler=scheduler, 
            positive=positive[0], 
            negative=negative[0],
            latent=latent_up[0], 
            denoise=latent_upscale_denoise)
        with torch.inference_mode():
            sample_up = sample_up[0]["samples"].to(torch.float16)
            vae.first_stage_model.cuda()
            decoded_up = vae.decode_tiled(sample_up).detach()
        print(torch.cuda.memory_cached(device=None))
        model_management.cleanup_models()
        gc.collect()
        model_management.soft_empty_cache()
        print(torch.cuda.memory_cached(device=None))
        final_up_image = Image.fromarray(np.array(decoded_up*255, dtype=np.uint8)[0])
        final_up_image.save("/content/TotoroUI/models/final_up_image.png")
        result = "/content/TotoroUI/models/final_up_image.png"
    else:
        with torch.inference_mode():
            sample = sample[0]["samples"].to(torch.float16)
            vae.first_stage_model.cuda()
            decoded = vae.decode_tiled(sample).detach()
        print(torch.cuda.memory_cached(device=None))
        model_management.cleanup_models()
        gc.collect()
        model_management.soft_empty_cache()
        print(torch.cuda.memory_cached(device=None))
        final_image = Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0])
        final_image.save("/content/TotoroUI/models/final_image.png")
        result = "/content/TotoroUI/models/final_image.png"

    response = None
    try:
        source_id = values['source_id']
        del values['source_id']
        source_channel = values['source_channel']     
        del values['source_channel']
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        files = {default_filename: open(result, "rb").read()}
        payload = {"content": f"{json.dumps(values)} <@{source_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{source_channel}/messages",
            data=payload,
            headers={"authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if os.path.exists(result):
            os.remove(result)

    if response and response.status_code == 200:
        try:
            payload = {"jobId": job_id, "result": response.json()['attachments'][0]['url']}
            requests.post(f"{web_uri}/api/notify", data=json.dumps(payload), headers={'Content-Type': 'application/json', "authorization": f"{web_token}"})
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            return {"result": response.json()['attachments'][0]['url']}
    else:
        return {"result": "ERROR"}

runpod.serverless.start({"handler": generate})
