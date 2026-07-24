import sys
try:
    import torch
except ImportError:
    torch = None

print("python  :", sys.version.split()[0])
if torch is not None:
    print("torch   :", torch.__version__)
    if torch.cuda.is_available():
        print("gpu     :", torch.cuda.get_device_name(0))
        print("vram    : %.1f GB" % (torch.cuda.get_device_properties(0).total_memory/1e9))
        print("bf16 ok :", torch.cuda.is_bf16_supported())
    else:
        print("NO GPU — Runtime > Change runtime type > T4 GPU")
else:
    print("torch   : NOT INSTALLED")

# T4 (Turing): bf16 False -> use fp16 in your configs
# L4/A100 (Ampere+): bf16 True -> use bf16, fewer NaN problems