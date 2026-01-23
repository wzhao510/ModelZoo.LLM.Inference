import json
import time
import sys
import argparse
import subprocess
import socket
from openai import OpenAI
from typing import Dict, List, Any, Optional

class VLLMServerManager:

    def __init__(self, model: str, port: int = 8000, tensor_parallel: int = 1, 
                 gpu_memory_utilization = 0.85, 
                 log_file: str = "vllm_server.log", tool_call_parser: str = "llama3_json",
                 enable_auto_tool_choice: bool = True):
        self.model = model
        self.port = port
        self.tensor_parallel = tensor_parallel
        self.gpu_memory_utilization = gpu_memory_utilization
        self.log_file = log_file
        self.tool_call_parser = tool_call_parser
        self.enable_auto_tool_choice = enable_auto_tool_choice
        self.process = None
        self.base_url = f"http://localhost:{port}/v1"
        
    def start_server(self):
        print(f"Starting vLLM server...")
        print(f"Model path: {self.model}")
        print(f"Port: {self.port}")
        print(f"Tensor parallel: {self.tensor_parallel}")
        print(f"Log file: {self.log_file}")
        print(f"Tool call parser: {self.tool_call_parser}")
        
        cmd = [
            "vllm", "serve", self.model,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tensor_parallel),
            "--tool-call-parser", self.tool_call_parser,
            "--gpu-memory-utilization", str(self.gpu_memory_utilization)
        ]
        
        if self.enable_auto_tool_choice:
            cmd.append("--enable-auto-tool-choice")
        
        # 启动服务器进程
        try:
            with open(self.log_file, 'w') as log_f:
                self.process = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True)
            
            print("Waiting for server to start (max 30 minutes)...")
            if self.wait_for_port_ready(timeout_minutes=30):
                # 额外检查API是否完全就绪
                if self.check_api_ready():
                    print(f"Server started successfully: {self.base_url}")
                    return True
                else:
                    print("Server port is open but API is not ready")
                    return False
            else:
                print("Server failed to start within 30 minutes")
                return False
                
        except Exception as e:
            print(f"Error starting server: {e}")
            return False
    
    def wait_for_port_ready(self, timeout_minutes: int = 30) -> bool:
        """使用socket持续检查端口是否开放"""
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        check_interval = 5  # 每5秒检查一次
        
        last_status_time = start_time
        status_interval = 30  # 每30秒打印一次状态
        
        while time.time() - start_time < timeout_seconds:
            # 检查进程是否还在运行
            if self.process.poll() is not None:
                print(f"Server process exited with code: {self.process.poll()}")
                return False
            
            # 检查端口是否开放
            if self.is_port_open("localhost", self.port):
                print("Server port is now open")
                return True
            
            # 定期打印状态
            current_time = time.time()
            if current_time - last_status_time >= status_interval:
                elapsed = int(current_time - start_time)
                remaining = int(timeout_seconds - elapsed)
                print(f"Waiting for server... {elapsed}s elapsed, {remaining}s remaining")
                last_status_time = current_time
            
            time.sleep(check_interval)
        
        return False
    
    def is_port_open(self, host: str, port: int) -> bool:
        """检查端口是否开放"""
        try:
            with socket.create_connection((host, port), timeout=5):
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False
    
    def check_api_ready(self, max_retries: int = 10) -> bool:
        """检查API是否完全就绪"""
        for i in range(max_retries):
            try:
                client = OpenAI(base_url=self.base_url, api_key="dummy", timeout=10)
                models = client.models.list()
                if models.data:
                    return True
            except Exception:
                if i < max_retries - 1:
                    time.sleep(3)
                continue
        return False
    
    def stop_server(self):
        """停止vLLM服务器"""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()

class ToolCallValidator:
    """Tool Call功能验证器"""
    
    def __init__(self, base_url: str = "http://localhost:8000/v1"):
        self.client = OpenAI(base_url=base_url, api_key="dummy", timeout=30)
        self.test_results = []
    
    def define_test_tools(self) -> List[Dict]:
        """定义测试用的工具集"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "calculate_expression",
                    "description": "计算数学表达式并返回结果",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "数学表达式"}
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取指定城市的当前天气信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "城市名称"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
    
    def execute_tool_function(self, function_name: str, arguments: Dict) -> str:
        """执行工具函数"""
        if function_name == "calculate_expression":
            expr = arguments["expression"]
            return f"计算结果: {expr} = 42 (模拟)"
        
        elif function_name == "get_weather":
            location = arguments["location"]
            unit = arguments.get("unit", "celsius")
            unit_text = "摄氏度" if unit == "celsius" else "华氏度"
            
            weather_data = {
                "北京": "晴朗 25度", "上海": "多云 28度", "广州": "阵雨 32度",
                "深圳": "晴间多云 30度", "New York": "晴朗 20度", "London": "阴天 15度"
            }
            
            weather = weather_data.get(location, "晴朗 25度")
            return f"{location}天气: {weather} ({unit_text})"
        
        return f"未知工具: {function_name}"
    
    def run_single_test(self, test_case: Dict) -> Dict[str, Any]:
        """运行单个测试用例"""
        print(f"Test {test_case['id']}: {test_case['name']}")
        print(f"  Input: {test_case['user_input']}")
        
        try:
            response = self.client.chat.completions.create(
                model=self.client.models.list().data[0].id,
                messages=[{"role": "user", "content": test_case["user_input"]}],
                tools=self.define_test_tools(),
                tool_choice=test_case.get("tool_choice", "auto"),
                temperature=0.1,
                max_tokens=500
            )
            
            message = response.choices[0].message
            result = {
                "test_id": test_case["id"],
                "test_name": test_case["name"],
                "user_input": test_case["user_input"],
                "has_tool_calls": bool(message.tool_calls),
                "direct_response": message.content,
                "tool_calls": [],
                "success": False
            }
            
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    function = tool_call.function
                    arguments = json.loads(function.arguments)
                    tool_result = self.execute_tool_function(function.name, arguments)
                    
                    result["tool_calls"].append({
                        "tool_name": function.name,
                        "arguments": arguments,
                        "tool_result": tool_result
                    })
                    
                    print(f"  Tool call: {function.name}")
                    print(f"    Arguments: {arguments}")
                    print(f"    Result: {tool_result}")
                
                result["success"] = self.verify_complete_workflow(
                    test_case["user_input"], message, result["tool_calls"]
                )
            else:
                print(f"  Direct response: {message.content}")
                if test_case.get("expected_tool") is None:
                    result["success"] = True
                else:
                    print(f"  Expected tool call: {test_case.get('expected_tool')}")
            
            return result
            
        except Exception as e:
            print(f"  Test failed: {str(e)}")
            return {"test_id": test_case["id"], "error": str(e), "success": False}
    
    def verify_complete_workflow(self, user_input: str, first_message: Any, tool_calls: List[Dict]) -> bool:
        """验证完整的工作流程"""
        try:
            messages = [
                {"role": "user", "content": user_input},
                first_message
            ]
            
            for i, tool_call in enumerate(first_message.tool_calls):
                messages.append({
                    "role": "tool",
                    "content": tool_calls[i]["tool_result"],
                    "tool_call_id": tool_call.id
                })
            
            response = self.client.chat.completions.create(
                model=self.client.models.list().data[0].id,
                messages=messages,
                max_tokens=300
            )
            
            final_message = response.choices[0].message
            print(f"  Final response: {final_message.content}")
            
            return bool(final_message.content and len(final_message.content.strip()) > 5)
                
        except Exception as e:
            print(f"  Workflow verification failed: {str(e)}")
            return False
    
    def run_validation_suite(self):
        """运行验证测试套件"""
        print("Starting Tool Call validation tests")
        print("=" * 50)
        
        test_cases = [
            {"id": "T01", "name": "Math calculation", "user_input": "计算 125 + 368", "expected_tool": "calculate_expression"},
            {"id": "T02", "name": "Complex calculation", "user_input": "(25 * 4 - 18) / 2", "expected_tool": "calculate_expression"},
            {"id": "T03", "name": "Weather query", "user_input": "北京天气怎么样？", "expected_tool": "get_weather"},
            {"id": "T04", "name": "English weather", "user_input": "Weather in New York?", "expected_tool": "get_weather"},
            {"id": "T05", "name": "Weather with unit", "user_input": "上海天气，华氏度", "expected_tool": "get_weather"},
            {"id": "T06", "name": "Force tool call", "user_input": "调用get_weather查广州", "expected_tool": "get_weather", 
             "tool_choice": {"type": "function", "function": {"name": "get_weather"}}},
            {"id": "T07", "name": "No tool needed", "user_input": "请自我介绍", "expected_tool": None},
        ]
        
        for test_case in test_cases:
            result = self.run_single_test(test_case)
            self.test_results.append(result)
        
        return self.generate_test_report()
    
    def generate_test_report(self):
        """生成测试报告"""
        print("\n" + "=" * 50)
        print("Test Report")
        print("=" * 50)
        
        total = len(self.test_results)
        success = sum(1 for r in self.test_results if r.get("success"))
        rate = (success / total) * 100 if total > 0 else 0
            
        print("\nDetailed results:")
        for result in self.test_results:
            status = "PASS" if result.get("success") else "FAIL"
            print(f"{result['test_id']}. {result['test_name']}: {status}")
            
            if "error" in result:
                print(f"   Error: {result['error']}")
        
        print(f"\nConclusion: {'PASS - Tool call functionality is working' if rate >= 70 else 'FAIL - Tool call issues detected'}")
        print(f"Total tests: {total}")
        print(f"Successful: {success}")
        print(f"正确率：{rate:.1f}%")

        return rate >= 70

def get_model_parser_config(model: str) -> tuple:
    """根据模型路径获取对应的tool-call解析器配置"""
    model_lower = model.lower()
    
    if 'qwen' in model_lower:
        return 'hermes', True
    elif 'llama' in model_lower:
        return 'llama3_json', True
    elif 'mistral' in model_lower:
        return 'mistral', True  
    elif 'internlm' in model_lower:
        return 'internlm', True
    elif 'hermes' in model_lower:
        return 'hermes', True
    else:
        return 'llama3_json', True

def main():

    parser = argparse.ArgumentParser(description="vLLM Tool Call validation script")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--tensor_parallel_size", "-tp", type=int, default=1, help="tp")
    parser.add_argument("--log", default="vllm_server.log", help="Server log file")
    
    # Tool Call parameters
    parser.add_argument("--tool-call-parser", help="Tool call parser (llama3_json/hermes/mistral/internlm)")
    parser.add_argument("--disable-auto-tool-choice", action="store_true", help="Disable auto tool choice")
    parser.add_argument('--gpu-memory-utilization',
                        type=float,
                        default=0.85,
                        help='the fraction of GPU memory to be used for '
                        'the model executor, which can range from 0 to 1.'
                        'If unspecified, will use the default value of 0.85.')

    args = parser.parse_args()
    
    # Auto detect model configuration
    auto_parser, auto_enable = get_model_parser_config(args.model)
    tool_call_parser = args.tool_call_parser or auto_parser
    enable_auto_tool_choice = not args.disable_auto_tool_choice
    
    print(f"Model configuration: {args.model}")
    print(f"Parser: {tool_call_parser} ({'auto-detected' if not args.tool_call_parser else 'manual'})")
    print(f"Auto tool choice: {'enabled' if enable_auto_tool_choice else 'disabled'}")
    
    # Create server manager
    server_manager = VLLMServerManager(
        model=args.model,
        port=args.port,
        tensor_parallel=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        log_file=args.log,
        tool_call_parser=tool_call_parser,
        enable_auto_tool_choice=enable_auto_tool_choice
    )
    
    try:
        # Start server and run tests
        if server_manager.start_server():
            validator = ToolCallValidator(base_url=server_manager.base_url)
            success = validator.run_validation_suite()
            sys.exit(0 if success else 1)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("User interrupted testing")
    except Exception as e:
        print(f"Testing error: {e}")
        sys.exit(1)
    finally:
        server_manager.stop_server()

if __name__ == "__main__":
    main()