import inspect
from typing import Callable, List, Optional, Union

import torch
from diffusers.configuration_utils import FrozenDict
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging

from custom_nodes.lightdream_model.scheduling_ddim import DDIMScheduler
from custom_nodes.lightdream_model.lightdream import LightDreamUnetTorchScriptModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LightdreamPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Lightricks' model ("Lightdream").
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        text_encoder ([`T5Encoder`]): Text encoder with built in tokenizer.
        unet ([`ConditionalUNet`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDPMScheduler`], [`DDIMScheduler`], [`PNDMScheduler`], [`EulerAncestralDiscreteScheduler`],
             [`EulerDiscreteScheduler`], or [`LMSDiscreteScheduler`].
             Details:
             - DDPMScheduler works with variance_type="learned_range" and beta_schedule="squaredcos_cap_v2".
             - DDIM works with beta_schedule="squaredcos_cap_v2".
             - Rest of the schedulers works with use beta_schedule="scaled_linear".
    """
    def __init__(
        self, device: torch.device, unet: LightDreamUnetTorchScriptModel,
        scheduler: Union[DDPMScheduler,
                         DDIMScheduler,
                         PNDMScheduler,
                         LMSDiscreteScheduler,
                         EulerDiscreteScheduler,
                         EulerAncestralDiscreteScheduler,
                         DPMSolverMultistepScheduler,
                        ]
    ):
        super().__init__()

        self._device = device
        self.register_modules(unet=unet, scheduler=scheduler)
        self.vae_scale_factor = 1
        self._unet = unet.to(self.device)
        self._scheduler = scheduler

    @property
    def device(self) -> torch.device:
        return self._device

    def _concat_embeddings(self, positive, negative):
        return torch.cat([negative[0], positive[0]]),\
               torch.cat([negative[1], positive[1]]).squeeze(),\
               torch.cat([negative[2], positive[2]])

    def decode_latents(self, latents):
        latents = (latents / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        return latents.cpu().permute(0, 2, 3, 1).float().numpy()

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is
            None) or (callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    # pylint: disable=too-many-arguments
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = (1, ) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
    @torch.no_grad()
    def __call__(
        self,
        positive,
        negative,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        initial_noise: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        input_image: Optional[torch.FloatTensor] = None,
        input_mask: Optional[torch.FloatTensor] = None,
        deviation: float = 1.0,
        **kwargs
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            t5_embeddings:
                embeddings of the prompt. Will be used instead of the original prompt if not None.
            last_tokens:
                last_tokens of the prompt. Will be used instead of the original prompt if not None.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            initial_noise (`torch.FloatTensor`, *optional*):
                Pre-generated noise, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.lightdream_model.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
        Examples:
        Returns:
            [`~pipelines.lightdream_model.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.lightdream_model.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.generator.model_resolution * self.vae_scale_factor
        width = width or self.unet.config.generator.model_resolution * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1
        device = self.device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        text_embeddings, last_tokens, attention_mask = self._concat_embeddings(positive, negative)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        torch.manual_seed(0)

        offset = self.scheduler.config.get("steps_offset", 0) if deviation < 1 else 0
        init_timestep_pos = int(num_inference_steps * deviation) + offset
        init_timestep_pos = min(init_timestep_pos, num_inference_steps)
        init_timestep_pos = max(init_timestep_pos, 1)
        first_timestep = self.scheduler.timesteps[-init_timestep_pos]
        batch_first_timestep = torch.tensor([first_timestep] * batch_size * num_images_per_prompt, device=device)
        t_start = max(num_inference_steps - init_timestep_pos + offset, 0)
        timesteps = self.scheduler.timesteps[t_start:].to(device)

        # 5. Prepare latent variables
        num_channels_latents = 3
        original_noise = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            initial_noise,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        epsilon = 0.00001
        if deviation < 1 - epsilon and input_image is not None:
            latents = self.scheduler.add_noise(input_image, original_noise, batch_first_timestep)
        else:
            latents = original_noise

        with self.progress_bar(iterable=range(num_inference_steps)) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                batch_t = t
                if not torch.is_tensor(batch_t):
                    batch_t = torch.tensor([batch_t], dtype=torch.long, device=self.device)
                elif torch.is_tensor(batch_t) and len(batch_t.shape) == 0:
                    batch_t = batch_t.unsqueeze(0).to(self.device)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                batch_t = batch_t * torch.ones(latent_model_input.shape[0], dtype=batch_t.dtype, device=batch_t.device)

                if "noise_aug" in kwargs:
                    kwargs["noise_aug"] = kwargs.get('noise_aug') * torch.ones(
                        latent_model_input.shape[0], dtype=batch_t.dtype, device=batch_t.device
                    ).unsqueeze(1)
                if "low_res" in kwargs:
                    kwargs["low_res"] = kwargs.get("low_res").repeat(latent_model_input.shape[0], 1, 1, 1)
                # predict the noise residual
                alphas_cumprod_batch = self.scheduler.add_noise(
                    torch.ones(1, dtype=torch.float32, device=device),
                    torch.zeros(1, dtype=torch.float32, device=device),
                    batch_t
                )

                noise_pred = self._unet(
                    x=latent_model_input,
                    timestep=alphas_cumprod_batch * 1000,
                    text_encoder_seq=text_embeddings,
                    text_encoder_last=last_tokens,
                    attention_mask=attention_mask,
                    **kwargs
                )
                if "low_res" in kwargs:
                    kwargs["low_res"] = kwargs.get("low_res")[0].unsqueeze(0)

                # perform guidance
                if do_classifier_free_guidance:
                    uncond_pred, cond_pred = torch.split(noise_pred, len(noise_pred) // 2, dim=0)
                    noise_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)

                if not isinstance(
                    self.scheduler, DDPMScheduler
                ) and not self.scheduler.config.prediction_type == "v_prediction":
                    # ddpm requires both noise and variance together.
                    noise_pred, _ = noise_pred.chunk(2, dim=1)

                # compute the previous noisy sample x_t -> x_t-1
                if isinstance(self.scheduler, DDIMScheduler):
                    extra_step_kwargs['use_clipped_model_output'] = True
                step_result = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
                latents = step_result.prev_sample

                if input_mask is not None:
                    if i < len(timesteps) - 1:
                        batch_t_next = timesteps[i + 1]
                        noised_image = self.scheduler.add_noise(input_image, original_noise, batch_t_next)
                    else:
                        noised_image = input_image
                    latents = latents * input_mask + noised_image * (1 - input_mask)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        preview_content = step_result.pred_original_sample \
                            if hasattr(step_result, 'pred_original_sample') else latents
                        callback(i, t, preview_content)

        return latents


def extract_samples_embeddings(data_loader, count, device):
    r"""
    Function extract sample embeddings from the given data_loader, with the given count.
    """
    t5_embeddings, last_tokens = None, None
    for i, batch in enumerate(data_loader):
        if t5_embeddings is not None:
            t5_embeddings = torch.cat([t5_embeddings, batch["t5_embeddings"]])
            last_token = batch["t5_embeddings"][:, batch["sequence_lengths"] - 1]
            last_tokens = torch.cat([last_tokens, last_token])
        else:
            t5_embeddings = batch["t5_embeddings"]
            last_tokens = t5_embeddings[:, batch["sequence_lengths"] - 1]
        if i == (count - 1):
            break
    t5_embeddings = t5_embeddings.to(device)
    last_tokens = last_tokens.to(device)

    t5_embeddings = torch.cat([torch.zeros_like(t5_embeddings), t5_embeddings])
    last_tokens = torch.cat([torch.zeros_like(last_tokens), last_tokens])
    return t5_embeddings, last_tokens
