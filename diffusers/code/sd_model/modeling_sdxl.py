from transformers import PretrainedConfig
from pathlib import Path
from diffusers.image_processor import VaeImageProcessor as DiffusersVaeImageProcessor
import onnxruntime as ort
from abc import abstractmethod
import numpy as np
from typing import Any, List, Dict, Optional, Union
import torch
import PIL
from transformers import CLIPFeatureExtractor, CLIPTokenizer
from diffusers import (
    DDIMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)

_ORT_TO_NP_TYPE = {
    "tensor(bool)": np.bool_,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(int16)": np.int16,
    "tensor(uint16)": np.uint16,
    "tensor(int32)": np.int32,
    "tensor(uint32)": np.uint32,
    "tensor(int64)": np.int64,
    "tensor(uint64)": np.uint64,
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
}

CONFIG_NAME = "config.json"
DIFFUSION_MODEL_UNET_SUBFOLDER = "unet"
DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER = "text_encoder"
DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER = "vae_decoder"
DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER = "vae_encoder"
DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER = "text_encoder_2"
ONNX_WEIGHTS_NAME = "model.onnx"

class _ORTDiffusionModelPart:
    CONFIG_NAME = "config.json"

    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.input_names = {input_key.name: idx for idx, input_key in enumerate(self.session.get_inputs())}
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}
        config_path = Path(session._model_path).parent / self.CONFIG_NAME
        self.config = PretrainedConfig.from_json_file(config_path).to_dict() if config_path.is_file() else {}
        self.input_dtype = {inputs.name: _ORT_TO_NP_TYPE[inputs.type] for inputs in self.session.get_inputs()}

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class ORTModelTextEncoder(_ORTDiffusionModelPart):
    def forward(self, input_ids: np.ndarray):
        onnx_inputs = {
            "input_ids": input_ids,
        }
        outputs = self.session.run(None, onnx_inputs)
        return outputs

class ORTModelUnet(_ORTDiffusionModelPart):
    def __init__(self, session: ort.InferenceSession):
        super().__init__(session)

    def forward(
        self,
        sample: np.ndarray,
        timestep: np.ndarray,
        encoder_hidden_states: np.ndarray,
        text_embeds: Optional[np.ndarray] = None,
        time_ids: Optional[np.ndarray] = None,
        timestep_cond: Optional[np.ndarray] = None,
    ):
        onnx_inputs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }

        if text_embeds is not None:
            onnx_inputs["text_embeds"] = text_embeds
        if time_ids is not None:
            onnx_inputs["time_ids"] = time_ids
        if timestep_cond is not None:
            onnx_inputs["timestep_cond"] = timestep_cond
        outputs = self.session.run(None, onnx_inputs)
        return outputs


class ORTModelVaeDecoder(_ORTDiffusionModelPart):
    def forward(self, latent_sample: np.ndarray):
        onnx_inputs = {
            "latent_sample": latent_sample,
        }
        outputs = self.session.run(None, onnx_inputs)
        return outputs


class ORTModelVaeEncoder(_ORTDiffusionModelPart):
    def forward(self, sample: np.ndarray):
        onnx_inputs = {
            "sample": sample,
        }
        outputs = self.session.run(None, onnx_inputs)
        return outputs

class VaeImageProcessor(DiffusersVaeImageProcessor):
    # Adapted from diffusers.VaeImageProcessor.denormalize
    @staticmethod
    def denormalize(images: np.ndarray):
        """
        Denormalize an image array to [0,1].
        """
        return np.clip(images / 2 + 0.5, 0, 1)

    # Adapted from diffusers.VaeImageProcessor.preprocess
    def preprocess(
        self,
        image: Union[torch.FloatTensor, PIL.Image.Image, np.ndarray],
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> np.ndarray:
        """
        Preprocess the image input. Accepted formats are PIL images, NumPy arrays or PyTorch tensors.
        """
        supported_formats = (PIL.Image.Image, np.ndarray, torch.Tensor)

        do_convert_grayscale = getattr(self.config, "do_convert_grayscale", False)
        # Expand the missing dimension for 3-dimensional pytorch tensor or numpy array that represents grayscale image
        if do_convert_grayscale and isinstance(image, (torch.Tensor, np.ndarray)) and image.ndim == 3:
            if isinstance(image, torch.Tensor):
                # if image is a pytorch tensor could have 2 possible shapes:
                #    1. batch x height x width: we should insert the channel dimension at position 1
                #    2. channnel x height x width: we should insert batch dimension at position 0,
                #       however, since both channel and batch dimension has same size 1, it is same to insert at position 1
                #    for simplicity, we insert a dimension of size 1 at position 1 for both cases
                image = image.unsqueeze(1)
            else:
                # if it is a numpy array, it could have 2 possible shapes:
                #   1. batch x height x width: insert channel dimension on last position
                #   2. height x width x channel: insert batch dimension on first position
                if image.shape[-1] == 1:
                    image = np.expand_dims(image, axis=0)
                else:
                    image = np.expand_dims(image, axis=-1)

        if isinstance(image, supported_formats):
            image = [image]
        elif not (isinstance(image, list) and all(isinstance(i, supported_formats) for i in image)):
            raise ValueError(
                f"Input is in incorrect format: {[type(i) for i in image]}. Currently, we only support {', '.join(supported_formats)}"
            )

        if isinstance(image[0], PIL.Image.Image):
            if self.config.do_convert_rgb:
                image = [self.convert_to_rgb(i) for i in image]
            elif do_convert_grayscale:
                image = [self.convert_to_grayscale(i) for i in image]
            if self.config.do_resize:
                height, width = self.get_height_width(image[0], height, width)
                image = [self.resize(i, height, width) for i in image]
            image = self.reshape(self.pil_to_numpy(image))
        else:
            if isinstance(image[0], torch.Tensor):
                image = [self.pt_to_numpy(elem) for elem in image]
                image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)
            else:
                image = self.reshape(np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0))

            if do_convert_grayscale and image.ndim == 3:
                image = np.expand_dims(image, 1)

            # don't need any preprocess if the image is latents
            if image.shape[1] == 4:
                return image

            if self.config.do_resize:
                height, width = self.get_height_width(image, height, width)
                image = self.resize(image, height, width)

        # expected range [0,1], normalize to [-1,1]
        do_normalize = self.config.do_normalize
        if image.min() < 0 and do_normalize:
            warnings.warn(
                "Passing `image` as torch tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] "
                f"when passing as pytorch tensor or numpy Array. You passed `image` with value range [{image.min()},{image.max()}]",
                FutureWarning,
            )
            do_normalize = False

        if do_normalize:
            image = self.normalize(image)

        if getattr(self.config, "do_binarize", False):
            image = self.binarize(image)

        return image

    # Adapted from diffusers.VaeImageProcessor.postprocess
    def postprocess(
        self,
        image: np.ndarray,
        output_type: str = "pil",
        do_denormalize: Optional[List[bool]] = None,
    ):
        if not isinstance(image, np.ndarray):
            raise ValueError(
                f"Input for postprocessing is in incorrect format: {type(image)}. We only support np array"
            )
        if output_type not in ["latent", "np", "pil"]:
            deprecation_message = (
                f"the output_type {output_type} is outdated and has been set to `np`. Please make sure to set it to one of these instead: "
                "`pil`, `np`, `pt`, `latent`"
            )
            warnings.warn(deprecation_message, FutureWarning)
            output_type = "np"

        if output_type == "latent":
            return image

        if do_denormalize is None:
            do_denormalize = [self.config.do_normalize] * image.shape[0]

        image = np.stack(
            [self.denormalize(image[i]) if do_denormalize[i] else image[i] for i in range(image.shape[0])], axis=0
        )

        image = image.transpose((0, 2, 3, 1))

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return image

    def get_height_width(
        self,
        image: [PIL.Image.Image, np.ndarray],
        height: Optional[int] = None,
        width: Optional[int] = None,
    ):
        """
        This function return the height and width that are downscaled to the next integer multiple of
        `vae_scale_factor`.

        Args:
            image(`PIL.Image.Image`, `np.ndarray`):
                The image input, can be a PIL image, numpy array or pytorch tensor. if it is a numpy array, should have
                shape `[batch, height, width]` or `[batch, height, width, channel]` if it is a pytorch tensor, should
                have shape `[batch, channel, height, width]`.
            height (`int`, *optional*, defaults to `None`):
                The height in preprocessed image. If `None`, will use the height of `image` input.
            width (`int`, *optional*`, defaults to `None`):
                The width in preprocessed. If `None`, will use the width of the `image` input.
        """
        height = height or (image.height if isinstance(image, PIL.Image.Image) else image.shape[-2])
        width = width or (image.width if isinstance(image, PIL.Image.Image) else image.shape[-1])
        # resize to integer multiple of vae_scale_factor
        width, height = (x - x % self.config.vae_scale_factor for x in (width, height))
        return height, width

    # Adapted from diffusers.VaeImageProcessor.numpy_to_pt
    @staticmethod
    def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
        """
        Convert a NumPy image to a PyTorch tensor.
        """
        if images.ndim == 3:
            images = images[..., None]

        images = torch.from_numpy(images)
        return images

    # Adapted from diffusers.VaeImageProcessor.pt_to_numpy
    @staticmethod
    def pt_to_numpy(images: torch.FloatTensor) -> np.ndarray:
        """
        Convert a PyTorch tensor to a NumPy image.
        """
        images = images.cpu().float().numpy()
        return images

    @staticmethod
    def reshape(images: np.ndarray) -> np.ndarray:
        """
        Reshape inputs to expected shape.
        """
        if images.ndim == 3:
            images = images[..., None]

        return images.transpose(0, 3, 1, 2)

    # TODO : remove after diffusers v0.21.0 release
    def resize(
        self,
        image: [PIL.Image.Image, np.ndarray, torch.Tensor],
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> [PIL.Image.Image, np.ndarray, torch.Tensor]:
        """
        Resize image.
        """
        if isinstance(image, PIL.Image.Image):
            image = image.resize((width, height), resample=PIL_INTERPOLATION[self.config.resample])
        elif isinstance(image, torch.Tensor):
            image = torch.nn.functional.interpolate(image, size=(height, width))
        elif isinstance(image, np.ndarray):
            image = self.numpy_to_pt(image)
            image = torch.nn.functional.interpolate(image, size=(height, width))
            image = self.pt_to_numpy(image)
        return image

class ORTStableDiffusionXLPipelineBase:
    def __init__(
        self,
        vae_decoder_session: ort.InferenceSession,
        text_encoder_session: ort.InferenceSession,
        unet_session: ort.InferenceSession,
        config: Dict[str, Any],
        tokenizer: CLIPTokenizer,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        feature_extractor: Optional[CLIPFeatureExtractor] = None,
        vae_encoder_session: Optional[ort.InferenceSession] = None,
        text_encoder_2_session: Optional[ort.InferenceSession] = None,
        tokenizer_2: Optional[CLIPTokenizer] = None,
    ):
        self._internal_dict = config

        self.vae_decoder = ORTModelVaeDecoder(vae_decoder_session)
        self.vae_decoder_model_path = Path(vae_decoder_session._model_path)

        self.unet = ORTModelUnet(unet_session)
        self.unet_model_path = Path(unet_session._model_path)

        self.text_encoder_model_path = Path(text_encoder_session._model_path)
        self.text_encoder = ORTModelTextEncoder(text_encoder_session)

        self.vae_encoder_model_path = Path(vae_encoder_session._model_path)
        self.vae_encoder = ORTModelVaeEncoder(vae_encoder_session)

        self.text_encoder_2_model_path = Path(text_encoder_2_session._model_path)
        self.text_encoder_2 = ORTModelTextEncoder(text_encoder_2_session)

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.scheduler = scheduler

        sub_models = {
            DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER: self.text_encoder,
            DIFFUSION_MODEL_UNET_SUBFOLDER: self.unet,
            DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER: self.vae_decoder,
            DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER: self.vae_encoder,
            DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER: self.text_encoder_2,
        }

        for name in sub_models.keys():
                self._internal_dict[name] = (
                    ("diffusers", "OnnxRuntimeModel") if sub_models[name] is not None else (None, None)
                )
        self._internal_dict.pop("vae", None)
        
        if "block_out_channels" in self.vae_decoder.config:
            self.vae_scale_factor = 2 ** (len(self.vae_decoder.config["block_out_channels"]) - 1)
        else:
            self.vae_scale_factor = 8

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        
        self.watermark = None
