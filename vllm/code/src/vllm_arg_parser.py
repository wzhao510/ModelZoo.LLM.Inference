#!/usr/bin/env python3  
"""  
vLLM参数解析工具脚本  
先解析vLLM参数，再解析自定义参数  
"""  
import sys  
from typing import Optional, Callable  
from argparse import Namespace  
from vllm.engine.arg_utils import EngineArgs  
from vllm.transformers_utils.config import get_config  
from vllm.utils.argparse_utils import FlexibleArgumentParser  
  

def _extract_model_from_args(args: list[str]) -> Optional[str]:  
    """从命令行参数中提取--model的值"""  
    try:  
        for i, arg in enumerate(args):  
            if arg == "--model":  
                if i + 1 < len(args) and not args[i + 1].startswith("-"):  
                    return args[i + 1]  
            elif arg.startswith("--model="):  
                return arg.split("=", 1)[1]  
        return None  
    except Exception:  
        return None  
  
def _load_model_config_defaults(model_path: str) -> dict:  
    """从模型路径的config.json加载vLLM相关配置"""  
    try:  
        # 获取模型的配置  
        hf_config = get_config(  
            model_path,  
            trust_remote_code=True,  
        )
          
        defaults = {}  

        if hasattr(hf_config, 'torch_dtype') and hf_config.torch_dtype is not None:  
            dtype_map = {  
                'torch.float16': 'half',  
                'torch.bfloat16': 'bfloat16',   
                'torch.float32': 'float'  
            }  
            torch_dtype_str = str(hf_config.torch_dtype)  
            if torch_dtype_str in dtype_map:  
                defaults['dtype'] = dtype_map[torch_dtype_str]  

        if hasattr(hf_config, 'max_position_embeddings'):  
            defaults['max_model_len'] = hf_config.max_position_embeddings  
              
        return defaults  
          
    except Exception as e:  
        print(f"Warning: Failed to load model config from {model_path}: {e}")  
        return {}
    
def create_vllm_parser(  
    description: str = "vLLM Script",  
    add_custom_args: Optional[Callable[[FlexibleArgumentParser], None]] = None,  
    include_async_args: bool = False,
    gpu_memory_utilization: float = 0.85 
) -> FlexibleArgumentParser:  
    """  
    创建包含vLLM参数的解析器  
      
    Args:  
        description: 脚本描述  
        add_custom_args: 添加自定义参数的函数  
        include_async_args: 是否包含异步引擎参数  
      
    Returns:  
        配置好的FlexibleArgumentParser  
    """  
    parser = FlexibleArgumentParser(description=description)  
      
    # 添加vLLM引擎参数  
    if include_async_args:  
        from vllm.engine.arg_utils import AsyncEngineArgs  
        parser = AsyncEngineArgs.add_cli_args(parser)  
    else:  
        parser = EngineArgs.add_cli_args(parser)  
    # 从命令行参数获取--model  
    model_path = _extract_model_from_args(sys.argv[1:])  
    # 从config.json加载默认配置  
    if model_path:  
        model_defaults = _load_model_config_defaults(model_path)  
        if model_defaults:  
            parser.set_defaults(**model_defaults)  

    parser.set_defaults(gpu_memory_utilization=gpu_memory_utilization)  
    
    # 添加自定义参数  
    if add_custom_args:  
        add_custom_args(parser)  
      
    return parser