import os
from pathlib import Path
import datetime

dlrm_bin = "python dlrm.py"

data = "dataset"  # synthetic
data_type = "kaggle"

input_path = Path("./data")

output_path = Path("./output")
output_path.mkdir(parents=True, exist_ok=True)

raw_data_file = input_path / "train.txt"
processed_data_file = input_path / "kaggleAdDisplayChallenge_processed.npz"

nbatches = -1  # Early stop to debug

# Train param
loss_func = "bce"
round_targets = "True"
lr = 0.1
m_batch_size = 128
test_m_batch_size = 16384
print_freq = 1024
test_freq = 1024

# Model param
sparse_size = 16
bot_mlp = "13-512-256-64-16"
top_mlp = "512-256-1"

args = f"  --arch-sparse-feature-dimension={sparse_size}" \
       f" --arch-mlp-bot={bot_mlp}" \
       f" --arch-mlp-top={top_mlp}" \
       f" --data-generation={data}" \
       f" --data-set={data_type}" \
       f" --raw-data-file={raw_data_file}" \
       f" --processed-data-file={processed_data_file}" \
       f" --loss-function={loss_func}" \
       f" --round-targets={round_targets}" \
       f" --learning-rate={lr}" \
       f" --mini-batch-size={m_batch_size}" \
       f" --num-batches={nbatches}" \
       f" --test-freq={test_freq}" \
       f" --test-mini-batch-size={test_m_batch_size}" \
       f" --print-freq={print_freq}" \
       f" --use-gpu" \
       f" --use-quiver"


def output(file, message):
    with open(file, "a") as f:
        print(message, file=f)
    print(message)


def run(gpu: int, ts: str, args: str):
    output_dir = output_path / ts

    output_dir.mkdir(parents=True)

    log_file = output_dir / "dlrm_kaggle.log"
    readme_file = output_dir / "README.txt"

    output(readme_file, args)
    cuda_arg = f"CUDA_VISIBLE_DEVICES={gpu}"

    cmd = f"{cuda_arg} {dlrm_bin} {args} 2>&1 | tee {log_file}"
    output(readme_file, cmd)
    os.system(cmd)

    output(readme_file, f"Done at {datetime.datetime.now()}")


if __name__ == '__main__':
    ts = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")

    run(0, ts, args)
