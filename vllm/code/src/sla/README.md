# 参数说明
```bash
# 参数                      默认值                       说明

--model	                    None	                    模型路径
--backend	                vllm	                    后端引擎（暂不支持sglang 此参数无效）
--host	                    127.0.0.1	                服务器主机地址
--port	                    9527	                    服务器端口
--dataset-name	            burstgpt	                数据集名称：burstgpt 或 random
--combinations	            ["128/128", "2048/128"]	    random模式使用的输入/输出长度组合
--mandatory	                [16, 8]	                    强制测试的并发点列表,预估最佳bs所在范围
--max-concurrency-limit	    20	                        最大并发限制（搜索范围上界）
--find-precise-bs           False                       是否要精准找到最佳并发
--con-times                 10                          num-prompt是max-concurrency的con-times倍

# file params
--model-name	            ""	                        模型名称（为空时从model路径自动提取）
--result-jsonl	            {model}_{mode}_{YYMMDD}.jsonl 结果jsonl文件路径
--log-file	                {model}_{YYYYMMDD_HHMM}.log	日志文件路径
--bench-output-jsonl	    ""	                        bench输出JSONL文件路径
--dataset-jsonl	            ""	                        数据集JSONL文件路径
--sla-path	                ./sla.sh	                sla.sh脚本路径
--dataset-path	            ./250910_BurstGPT.csv	    默认数据集CSV路径

# sla params
--ttft-ms-max	            2005.0	                    TTFT最大延迟（毫秒）
--tpot-ms-max	            51.0	                    TPOT最大延迟（毫秒）
--qps-field	                request_throughput	        QPS字段名称

# debug params
--print-cmd	                False	                    打印执行的命令（调试用）
--dry-run	                False	                    干跑模式（不实际执行测试,调试用）
```
# 使用示例
- random
```bash 
# 使用随机输入输出  寻找300/300 1024/1024输入组合 在ttfs<2s tpot<50ms 的最佳并发  
# 预估最佳并发范围是8到32  上限是40 使用默认--con-times 10倍并发测试  
# 测试完毕后会自动将结果保存到{model}_{mode}_{YYMMDD}.jsonl和{model}_{YYYYMMDD_HHMM}.log中
python sla.py --model /path/to/model --find-precise-bs true --con-times 5 --dataset-name random \
--combinations 300/300 1024/1024  --mandatory 8 32 --max-concurrency-limit 40 --port 8000
```
- burstgpt
```bash 
# 使用burstgpt数据集 在ttfs<2s tpot<50ms 的最佳并发 （不需要指定输入输出长度）
# 预估最佳并发范围是8到32  上限是40 使用默认--con-times 10倍并发测试  
# 测试完毕后会自动将结果保存到{model}_{mode}_{YYMMDD}.jsonl和{model}_{YYYYMMDD_HHMM}.log中
python sla.py --model /path/to/model --find-precise-bs true --con-times 5 \
--dataset-name burstgpt  --mandatory 8 32 --max-concurrency-limit 40 --port 8000 
```