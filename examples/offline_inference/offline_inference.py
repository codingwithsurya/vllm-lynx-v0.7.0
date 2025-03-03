from vllm import LLM, SamplingParams
import argparse

import os
import sys

# # Set NCCL environment variables to make distributed communication more robust
# os.environ["NCCL_DEBUG"] = "INFO"  # Enable detailed NCCL logging
# os.environ["NCCL_IB_DISABLE"] = "1"  # Disable InfiniBand (use TCP instead)
# os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"  # Avoid docker0 and loopback interfaces
# os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable peer-to-peer operations
# os.environ["NCCL_BLOCKING_WAIT"] = "1"  # Use blocking wait to improve stability


def parse_arguments():
    parser = argparse.ArgumentParser(description='Offline Inference Script')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--seq_length', type=int, default=10, help='Sequence length for each input')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the Mixtral config file')
    return parser.parse_args()

args = parse_arguments()

prompt_token_ids = []
for _ in range(args.batch_size):
    tokens = list(range(args.seq_length))
    prompt_token_ids.append(tokens)


# Sample prompts.
# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
#     "Need to increase the batch size",
#     "I am so sleepy",
#     "AI makes machines go brrrr",
#     "The best movie of all time is",
#     "The best book of all time is",
#     "The best song of all time is",
#     "Python is a popular programming language because",
#     "The largest planet in our solar system is",
#     "The fastest land animal is",
#     "To be or not to be, that is",
#     "A healthy breakfast consists of",
#     "The primary colors are",
#     "The solution to climate change is",
#     "The longest river in the world is",
#     "The inventor of the telephone is",
#     "The first person to walk on the moon was",
#     "Machine learning is a branch of",
#     "The sum of two plus two is",
#     "The speed of light is",
#     "The meaning of life is",
#     "The Great Wall of China was built to",
#     "The most popular sport in the world is",
#     "The atomic number of carbon is",
#     "The most expensive painting ever sold is",
#     "In the year 2100, humans will",
#     "The best way to learn programming is",
#     "The Nobel Prize in Physics was awarded for",
#     "The tallest building in the world is",
# ]
# prompt_token_ids = [
#     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#     [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
#     [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
#     [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
#     [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
#     [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
#     [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
# ]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=100,ignore_eos=True)

# Create an LLM.
#llm = LLM(model="facebook/opt-125m")
# CONFIG_PATH = "/scratch/vgupta345/prowl/mixtral_configs/none-none-2.json"
# CONFIG_PATH = "/scratch/vgupta345/prowl/mixtral_configs/none-none-2.json"
llm = LLM(model="mistralai/Mixtral-8x7B-v0.1", tensor_parallel_size=2, enforce_eager=True, gpu_memory_utilization=0.9, mixtral_config_file=args.config_path, logit_logging_frequency=5, load_format='dummy')
# llm = LLM(model="mistralai/Mixtral-8x7B-Instruct-v0.1", tensor_parallel_size=4, enforce_eager=True, gpu_memory_utilization=0.9, num_experts=8, mixtral_config_file=CONFIG_PATH, logit_logging_frequency=5)
# llm = LLM(model="meta-llama/Meta-Llama-3.1-70B", tensor_parallel_size=4, enforce_eager=True, gpu_memory_utilization=0.9)
# llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", tensor_parallel_size=1, enforce_eager=True, gpu_memory_utilization=0.9)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts=[""] * len(prompt_token_ids),sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    # prompt_token_ids = output.prompt_token_ids
    # print("The prompt length is: ", len(prompt_token_ids))
    generated_text = output.outputs[0].text
    # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
