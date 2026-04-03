#!/bin/bash

# 使用 && 连接，只有第一个成功才执行第二个
echo "开始运行脚本..."
bash prompt_embeds/run_prompt_embeds.sh && bash tool_calling/run_tool_calling.sh
