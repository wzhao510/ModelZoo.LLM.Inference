
import numpy as np
import random
import os
from diffusers import OnnxStableDiffusionPipeline, OnnxRuntimeModel
from diffusers import DDIMScheduler, PNDMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer



SchedulerMap = {
    "DDIM" : DDIMScheduler,
    "PNDM" : PNDMScheduler,
    "EulerA" : EulerAncestralDiscreteScheduler,
}


class Text2ImgModelSess:
    def __init__(self, 
                params,
                text_encoder_sess,
                unet_sess,
                vae_decoder_sess):
        self.params = params
        self.text_encoder = OnnxRuntimeModel(model=text_encoder_sess)
        self.unet = OnnxRuntimeModel(model=unet_sess)
        self.vae_decoder = OnnxRuntimeModel(model=vae_decoder_sess)
        self.tokenizer = CLIPTokenizer.from_pretrained(params["fp16_path"], subfolder="tokenizer")

        self.generator = np.random
        self.num_inference_steps = params["num_inference_steps"]

        # para get and set
        img_info = params["outputs_size"].split("#")
        self.height = int(img_info[0])
        self.width = int(img_info[1])

        self.scheduler_name = params["scheduler"]
        scheduler_class =  SchedulerMap.get(params["scheduler"], None)
        if scheduler_class is None:
            raise ValueError(f"scheduler: {self.scheduler_name} not in {SchedulerMap.keys()}")
        self.scheduler = scheduler_class.from_pretrained(os.path.join(params["fp16_path"], "scheduler/scheduler_config.json"))
        random_int = random.randint(0, 2**32 - 1) if params["seed"] == -1 else params["seed"]
        np.random.seed(random_int)

        self.pipe = OnnxStableDiffusionPipeline(
            vae_encoder=None,
            vae_decoder=self.vae_decoder,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )

        self.pipe.set_progress_bar_config(disable=True)
    
    def truncated_prompt_embeds(self, prompt):
        text_input_ids = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="np",
            ).input_ids
        prompt_embeds = self.text_encoder(input_ids=text_input_ids.astype(np.int32))[0]
        return prompt_embeds
        
    def __call__(self, prompts, height=None, width=None, prompt_embeds=None):
        if height is None:
            height = self.height
        if width is None:
            width = self.width
        if prompt_embeds is not None:
            prompts = None
        
        image = self.pipe(prompts,
                            height=height,
                            width=width,
                            generator=self.generator,
                            num_inference_steps=self.num_inference_steps,
                            prompt_embeds=prompt_embeds)
        return image

        

