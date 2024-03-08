import os
import torch
import numpy as np
from diffusers import OnnxStableDiffusionPipeline, OnnxRuntimeModel
from diffusers import DDIMScheduler, PNDMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer



SchedulerMap = {
    "DDIM" : DDIMScheduler,
    "PNDM" : PNDMScheduler,
    "EulerA" : EulerAncestralDiscreteScheduler,
}


class Text2ImgModel:
    def __init__(self, params, provider= "MACAExecutionProvider"):
        self.provider = provider
        self.scheduler_name = params["scheduler"]
        self.seed = params["seed"]
        self.model_dir =  params["ori_path"]
        self._build_model(params)
        self.generator = np.random
        

        scheduler_class =  SchedulerMap.get(params["scheduler"], None)
        if scheduler_class is None:
            raise ValueError(f"scheduler: {self.scheduler_name} not in {SchedulerMap.keys()}") 
        self.scheduler = scheduler_class.from_pretrained(os.path.join(params["ori_path"], "scheduler/scheduler_config.json"))
        
        random_int = random.randint(0, 2**32 - 1) if params["seed"] == -1 else params["seed"]
        np.random.seed(params["seed"])
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
    
    def check_model(self, params):
        return True

    def _build_model(self, params):
        model_dir =  params["ori_path"]
        self.tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
        self.text_encoder = OnnxRuntimeModel(model=OnnxRuntimeModel.load_model(os.path.join(model_dir, f"text_encoder/{params['text_encoder']}"), provider=self.provider))
        self.vae_decoder = OnnxRuntimeModel(model=OnnxRuntimeModel.load_model(os.path.join(model_dir, f"vae_decoder/{params['vae_decoder']}"), provider=self.provider))
        self.unet = OnnxRuntimeModel(model=OnnxRuntimeModel.load_model(os.path.join(model_dir, f"unet/{params['unet']}"), provider=self.provider))
        
    
    def refresh_para(self, params):
        status = 0

        if params["ori_path"] != self.model_dir:
            if not check_model(params):
                return -1
            self.model_dir =  params["ori_path"]
            self._build_model(params)
            status = 1

        if params["scheduler"] != self.scheduler_name:
            scheduler_class =  SchedulerMap.get(params["scheduler"], None)
            if scheduler_class is None:
                 print(f"scheduler: {self.scheduler_name} not in {SchedulerMap.keys()}.Plese set Again")
                 return -1
            self.scheduler_name = params["scheduler"]
            self.scheduler = scheduler_class.from_pretrained(os.path.join(params["ori_path"], "scheduler/scheduler_config.json"))
            status = 1

        if params["seed"] != self.seed:
            if params["seed"] < -1 or params["seed"] > 2**32:
                print(f'seed: {params["seed"]} not in [-1, 2**32] range')
                return -1
            random_int = random.randint(0, 2**32 - 1) if params["seed"] == -1 else params["seed"]
            self.seed = params["seed"]
            np.random.seed(params["seed"])
        
        return status

    def infer(self, prompts, params):
        status_ret = self.refresh_para(params)
        if status_ret == -1:
            return None
        elif status_ret == 1:
            self.pipe = OnnxStableDiffusionPipeline(
                        vae_encoder=None,
                        vae_decoder=self.vae_decoder,
                        text_encoder=self.text_encoder,
                        tokenizer=self.tokenizer,
                        unet=self.unet,
                        scheduler=self.scheduler,
                        safety_checker=None,
                        feature_extractor=None,
                        requires_safety_checker=False)
        image = self.pipe(prompts, generator=self.generator, num_inference_steps=params["num_inference_steps"])
        return image
