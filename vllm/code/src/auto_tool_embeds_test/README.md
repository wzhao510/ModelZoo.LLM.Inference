# Project README

## GPU Configuration

You can specify which GPU to use by setting the CUDA_VISIBLE_DEVICESi environment variable. For example, to use GPU 3,4:

  `export CUDA_VISIBLE_DEVICES=3,4`
  
If not specified, the system will use GPU 0 by default.

## Quick Start

To test different functionalities, please use the following commands:

- **To run all tests at once**:  
  Run `bash auto_test_prompt_tool.sh`

- **To test prompt embeddings**:  
  Run `bash prompt_embeds/run_prompt_embeds.sh`

- **To test tool call**:  
  Run `bash tool_calling/run_tool_calling.sh`

- **To test structure output**:  
  Run `bash structured_output/run_structured_output.sh`

## Log Files

The logs for each test are saved in the following locations:

- **Prompt embeddings test logs**: `./prompt_embeds.log`
- **Tool call test logs**: `./tool_calling.log`
- **Tool call structure output**: `./structured_output.log`

You can monitor the logs in real-time or check them after execution for debugging purposes.

