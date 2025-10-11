import os
import random

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("configs", exist_ok=True)

container_name='sglang-benchmark-server'
model_path='/oschina0/kychina/models/DeepSeek-R1-BF16_W8A8/vllm_quant_model/'
port=8000

host_list = [
    "192.168.12.15",
    "192.168.12.17",

    "192.168.12.11",
    "192.168.12.13",
]

num_prefill = 1
num_decoder = 1
num_nodes_per_prefill = 2
num_nodes_per_decoder = 2
prefill_pp_size = 2


decoder_hosts = host_list[:num_decoder * num_nodes_per_decoder]
prefill_hosts = host_list[num_decoder * num_nodes_per_decoder:]

prefill = [prefill_hosts[i:i+num_nodes_per_prefill] for i in range(0, len(prefill_hosts), num_nodes_per_prefill)]
decoder = [decoder_hosts[i:i+num_nodes_per_decoder] for i in range(0, len(decoder_hosts), num_nodes_per_decoder)]

print(f"{prefill=}")
print(f"{decoder=}")

prefill_rank0_ips = []
decode_rank0_ips = []

for ps in prefill:
    prefill_rank0_ips.append(ps[0])
    template = """
export MODEL={model}
export CONTAINER_NAME={name}
export DIST_INIT_ADDR={host}
export NODE_RANK={rank}
export ROLE=prefill
export NNODES={nnodes}
export PREFILL_PP_SIZE={pp}
"""
    for rank, host in enumerate(ps):
        with open(f"configs/{host}.env", "w") as f:
            f.write(
                template.format(
                    model=model_path, name=container_name, host=f'{ps[0]}:5000', rank=rank, nnodes=len(ps), pp=prefill_pp_size
                )
            )

for ds in decoder:
    decode_rank0_ips.append(ds[0])
    template = """
export MODEL={model}
export CONTAINER_NAME={name}
export DIST_INIT_ADDR={host}
export NODE_RANK={rank}
export ROLE=decoder
export NNODES={nnodes}
export PREFILL_PP_SIZE={pp}
"""
    for rank, host in enumerate(ds):
        with open(f"configs/{host}.env", "w") as f:
            f.write(
                template.format(
                    model=model_path, name=container_name, host=f'{ds[0]}:5000', rank=rank, nnodes=len(ds), pp=prefill_pp_size
                )
            )


base_port = 30000
host_ip   = prefill_rank0_ips[0]  

prefill_args = ' '.join([f'http://{ip}:{base_port + i}'
                         for i, ip in enumerate(prefill_rank0_ips)])
decode_args  = ' '.join([f'http://{ip}:{base_port + len(prefill_rank0_ips) + i}'
                         for i, ip in enumerate(decode_rank0_ips)])


prefill_logs = ' '.join([f'prefill_{ip}.log' for ip in prefill_rank0_ips])
decoder_logs = ' '.join([f'decoder_{ip}.log' for ip in decode_rank0_ips])


mini_lb_cmd = f"docker exec {container_name} /bin/bash -c 'source /opt/conda/etc/profile.d/conda.sh;conda activate base;python3 -m sglang.srt.disaggregation.mini_lb --prefill {prefill_args} --decode {decode_args} --host {host_ip} --port {port} 2>&1 | tee $log_dir/mini_lb.log'"
mini_lb_cmd = f'ssh -o StrictHostKeyChecking=no {host_ip} "{mini_lb_cmd}" '

shell_script = f"""#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <log_dir>"
  exit 1
fi

log_dir=$1

prefill_logs="{prefill_logs}"
decoder_logs="{decoder_logs}"

wait_for_logs() {{
  for log in $prefill_logs $decoder_logs; do
    log_path="$log_dir/$log"
    while [ ! -f "$log_path" ]; do
      echo "Waiting for log file $log_path to be created..."
      sleep 10
    done
  done
}}


check_logs() {{
  for log in $prefill_logs $decoder_logs; do
    log_path="$log_dir/$log"
    while ! grep -q "The server is fired up and ready to roll!" "$log_path"; do
      echo "Waiting for server to start in $log_path..."
      sleep 10
    done
  done
}}

wait_for_logs

check_logs

{mini_lb_cmd} 
"""

with open(f"configs/mini_lb.sh", "w") as f:
    f.write(shell_script)


def generate_tmux(hosts):
    tmux_str = []

    for i in range(len(hosts) - 1):
        tmux_str += ['tmux split-window -v']
        tmux_str += ['tmux select-layout even-vertical']

    for i, host in enumerate(hosts):
        tmux_str.append(f'tmux send-keys -t {i} "ssh -o StrictHostKeyChecking=no {host}" C-m')

    return '\n'.join(tmux_str)

prefill_hosts = [j for i in prefill for j in i]
with open('configs/stop_prefill.sh', 'w') as f:
    stop_str = [f'ssh -o StrictHostKeyChecking=no {host} "docker stop {container_name}" &' for host in prefill_hosts]
    f.write('\n'.join(stop_str))

deocder_hosts = [j for i in decoder for j in i]
with open('configs/stop_decoder.sh', 'w') as f:
    stop_str = [f'ssh -o StrictHostKeyChecking=no {host} "docker stop {container_name}" &' for host in deocder_hosts]
    f.write('\n'.join(stop_str))

with open('configs/stop_all.sh', 'w') as f:
    stop_str = [f'ssh -o StrictHostKeyChecking=no {host} "docker stop {container_name}" &' for host in prefill_hosts + deocder_hosts]
    f.write('\n'.join(stop_str))

def generate_launch(hosts):
    launch_str = []
    cwd = os.getcwd()

    for i in range(len(hosts)):
        launch_str.append(f'ssh -o StrictHostKeyChecking=no {hosts[i]} "cd {cwd} && bash template_docker.sh &"')

    return '\n'.join(launch_str)

with open('configs/launch_prefill.sh', 'w') as f:
    f.write(generate_launch(prefill_hosts))

with open('configs/launch_decoder.sh', 'w') as f:
    f.write(generate_launch(deocder_hosts))

with open('configs/launch_all.sh', 'w') as f:
    f.write(generate_launch(prefill_hosts + deocder_hosts))

with open('configs/tmux_prefill.sh', 'w') as f:
    f.write(generate_tmux(prefill_hosts))

with open('configs/tmux_decoder.sh', 'w') as f:
    f.write(generate_tmux(deocder_hosts))

with open('configs/tmux_all.sh', 'w') as f:
    f.write(generate_tmux(prefill_hosts + deocder_hosts))

print("Environment files generated successfully.")

