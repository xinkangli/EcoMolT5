import torch

# 输出 PyTorch 版本
print(f"PyTorch Version: {torch.__version__}")

# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

# 如果 CUDA 可用，输出 CUDA 版本
if cuda_available:
    print(f"CUDA Version: {torch.version.cuda}")
    # 打印可用的 GPU 数量
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    # 打印每个 GPU 的名称
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 测试一个简单的 CUDA 操作
    device = torch.device("cuda")  # 指定使用 GPU
    x = torch.rand(3, 3).to(device)  # 创建一个随机张量并将其转移到 GPU
    y = torch.rand(3, 3).to(device)  # 创建另一个张量并转移到 GPU
    z = x + y  # 在 GPU 上执行加法操作
    print(f"Result of x + y on CUDA: \n{z}")
else:
    print("CUDA is not available. Running on CPU.")
