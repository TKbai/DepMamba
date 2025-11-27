import torch
print(torch.version.cuda)  # 必须输出 12.4
print(torch.cuda.is_available()) # 必须为 True
print(torch.cuda.get_device_capability()) # 5090 应该显示 (10, 0) 或类似