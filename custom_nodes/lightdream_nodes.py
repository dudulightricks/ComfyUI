import torch
from enum import Enum
import math

import comfy.sample
import comfy.utils
import comfy.model_management

from custom_nodes.lightdream_model.lightdream import LightDreamModel
from custom_nodes.lightdream_model.lightdream_pipeline import LightdreamPipeline
from custom_nodes.lightdream_model.noise_scheduler_factory import create_scheduler, NoiseScheduler
import latent_preview
from custom_nodes.lightdream_latentformat import LightdreamFormat
from custom_nodes.lightdream_model.models_library import fetch


class LightdreamVersion(Enum):
    LT_1_0 = "lt-1-0"

class LightdreamCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model_version": ([e.value for e in LightdreamVersion], ),
                             }}
    RETURN_TYPES = ("MODEL", "TEXT_ENCODER")
    FUNCTION = "load_checkpoint"

    CATEGORY = "lightricks/lightdream"

    def load_checkpoint(self, model_version: str):
        assert model_version == LightdreamVersion.LT_1_0.value, "lt-1-0 is the only one supported currently."
        model_path = fetch("lightdream/ld-v1")
        model = LightDreamModel.load(
            path=model_path,
            device="cuda",
        )
        return model.unet, model.text_encoder


class LightdreamTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True}), "text_encoder": ("TEXT_ENCODER", )}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "lightricks/lightdream"

    def encode(self, text_encoder, text):
        device = comfy.model_management.get_torch_device()
        text_embeddings, last_tokens, attn_mask = self._make_text_embeddings_and_last_tokens(
            text_encoder=text_encoder, prompt=[text], num_images_per_prompt=1, device=device
        )
        cond = (text_embeddings, last_tokens, attn_mask)
        return (cond, )

    def _make_text_embeddings_and_last_tokens(self, text_encoder, prompt, num_images_per_prompt, device):
        text_embeddings, sequence_lengths_per_prompt = text_encoder.encode(texts=prompt)
        text_embeddings = text_embeddings.to(device)
        sequence_lengths_per_prompt = sequence_lengths_per_prompt.to(device)

        t5_max_sequence_length = text_encoder.max_sequence_length
        sequence_lengths_per_prompt.clamp_(0, t5_max_sequence_length)

        batch_size = len(prompt)
        batch_range = torch.arange(batch_size, device=device)
        last_tokens = text_embeddings[batch_range, sequence_lengths_per_prompt - 1]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        last_tokens = last_tokens.repeat(1, num_images_per_prompt, 1)
        last_tokens = last_tokens.view(bs_embed * num_images_per_prompt, 1, -1)

        return text_embeddings, last_tokens, self._compute_attention_mask(text_embeddings, sequence_lengths_per_prompt)

    def _compute_attention_mask(self, text_embeddings, sequence_lengths):
        max_seq_len = text_embeddings.size(1)
        range_tensor = torch.arange(max_seq_len, device=sequence_lengths.device)
        return range_tensor.unsqueeze(0) < sequence_lengths.unsqueeze(1)

class LightdreamEmptyLatentImage:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 256}),
                              "height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 256}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "lightricks/lightdream"

    def generate(self, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 3, height, width])
        return ({"samples":latent}, )

class LightdreamKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.5, "round": 0.01}),
                    "scheduler": ([e.value for e in NoiseScheduler], ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "deviation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "lightricks/lightdream"

    def sample(self, model, seed, steps, cfg, scheduler, positive, negative, latent_image, deviation=1.0):
        latent = latent_image
        device = comfy.model_management.get_torch_device()
        latent_image = latent["samples"]
        latent_image = latent_image.to(device)

        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"].to(device)

        preview_format = "JPEG"
        if preview_format not in ["JPEG", "PNG"]:
            preview_format = "JPEG"

        previewer = None
        previewer = latent_preview.get_previewer(device, LightdreamFormat())

        pbar = comfy.utils.ProgressBar(steps)

        def callback(step, timestep, latents):
            preview_bytes = None
            if previewer:
                preview_bytes = previewer.decode_latent_to_preview_image(preview_format, latents)
            pbar.update_absolute(step + 1, steps, preview_bytes)

        out = latent.copy()

        scheduler = create_scheduler(NoiseScheduler(scheduler))
        pipeline = LightdreamPipeline(
            device=device, unet=model, scheduler=scheduler
        )
        latents = pipeline(
            positive=positive,
            negative=negative,
            height=latent_image.shape[-2],
            width=latent_image.shape[-1],
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=None,
            initial_noise=noise,
            callback=callback,
            input_image=latent_image,
            input_mask=noise_mask,
            deviation=deviation,
        )

        out["samples"] = latents
        return (out,)


class LightdreamLatentDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT", )}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    CATEGORY = "lightricks/lightdream"

    def decode(self, samples):
        latents = (samples["samples"] / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        latents = latents.permute(0, 2, 3, 1).float()
        return (latents, )


class LightdreamLatentEecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "pixels": ("IMAGE", )}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "lightricks/lightdream"

    def encode(self, pixels):
        latents = ((pixels * 2) - 1).clamp(-1, 1)
        latents = latents.permute(0, 3, 1, 2)
        return ({"samples":latents}, )

class LightdreamEncodeForInpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "pixels": ("IMAGE", ), "mask": ("MASK", ), "grow_mask_by": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "lightricks/lightdream/inpaint"

    def encode(self, pixels, mask, grow_mask_by=6):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

        pixels = pixels.clone()
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
            mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

        #grow mask by a few pixels to keep things seamless in latent space
        if grow_mask_by == 0:
            mask_erosion = mask
        else:
            kernel_tensor = torch.ones((1, 1, grow_mask_by, grow_mask_by))
            padding = math.ceil((grow_mask_by - 1) / 2)

            mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask.round(), kernel_tensor, padding=padding), 0, 1)

        latents = ((pixels * 2) - 1).clamp(-1, 1)
        latents = latents.permute(0, 3, 1, 2)

        return ({"samples":latents, "noise_mask": (mask_erosion[:,:,:x,:y].round())}, )


NODE_CLASS_MAPPINGS = {
    "LightdreamCheckpointLoader": LightdreamCheckpointLoader,
    "LightdreamTextEncode": LightdreamTextEncode,
    "LightdreamEmptyLatentImage": LightdreamEmptyLatentImage,
    "LightdreamLatentDecode": LightdreamLatentDecode,
    "LightdreamKSampler": LightdreamKSampler,
    "LightdreamLatentEecode": LightdreamLatentEecode,
    "LightdreamEncodeForInpaint": LightdreamEncodeForInpaint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LightdreamCheckpointLoader": "Lightdream Load Checkpoint",
    "LightdreamTextEncode": "Lightdream Text Encode (Prompt)",
    "LightdreamEmptyLatentImage": "Lightdream Empty Latent Image",
    "LightdreamLatentDecode": "Lightdream Latent Decode",
    "LightdreamKSampler": "Lightdream KSampler",
    "LightdreamLatentEecode": "Lightdream Latent Encode",
    "LightdreamEncodeForInpaint": "Lightdream Encode (for Inpainting)",
}
