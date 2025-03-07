# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


import argparse
import os
import subprocess
import tensorrt as trt
import torch
import torchvision
import torchvision.transforms as transforms
import subprocess
from hardware_stats_usage import create_tegrastats_file
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('engine', type=str, default=None, help='Path to the optimized TensorRT engine')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--dataset_path', type=str, default='data/cifar10')
args = parser.parse_args()

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

test_dataset = torchvision.datasets.CIFAR10(
    root=args.dataset_path, train=False, download=True, transform=transform
)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

logger = trt.Logger()
runtime = trt.Runtime(logger)

with open(args.engine, 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())


context = engine.create_execution_context()

input_binding_idx = engine.get_binding_index('input')
output_binding_idx = engine.get_binding_index('output')

input_shape = (args.batch_size, 3, 32, 32)
output_shape = (args.batch_size, 10)

context.set_binding_shape(input_binding_idx, input_shape)

input_buffer = torch.zeros(input_shape, dtype=torch.float32, device=torch.device('cuda'))
output_buffer = torch.zeros(output_shape, dtype=torch.float32, device=torch.device('cuda'))

bindings = [None, None]
bindings[input_binding_idx] = input_buffer.data_ptr()
bindings[output_binding_idx] = output_buffer.data_ptr()

test_accuracy = 0

tegrastats_output = f"/TFG/jetson_dla_tutorial/eval_excels/tegrastats_{args.engine}_batch_size_{args.batch_size}_raw.csv"
filtered_tegrastats_output = f"/TFG/jetson_dla_tutorial/eval_excels/tegrastats_{args.engine}_batch_size_{args.batch_size}.csv"

os.makedirs(os.path.dirname(tegrastats_output), exist_ok=True)

# run tegrastats in the background
tegrastats = subprocess.Popen(
    ["tegrastats", "--interval", "100", "--logfile", tegrastats_output],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)


# run through test dataset
for i in range(10):
    for image, label in iter(test_loader):

        actual_batch_size = int(image.shape[0])

        input_buffer[0:actual_batch_size].copy_(image)

        context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)

        torch.cuda.current_stream().synchronize()

        output = output_buffer[0:actual_batch_size]
        label = label.cuda()

        test_accuracy += int(torch.sum(output.argmax(dim=-1) == label))

        # print(f'Image: {image.shape}, Label: {label}, Prediction: {output.argmax(dim=-1)}')

    test_accuracy /= len(test_dataset)

    print(f'TEST ACCURACY: {test_accuracy}')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tegrastats.terminate()

create_tegrastats_file(tegrastats_output, filtered_tegrastats_output)
print(f"Datos de tegrastats guardados en {filtered_tegrastats_output}")
os.remove(tegrastats_output)
