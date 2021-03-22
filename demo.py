import torch
from models import UNet

def dump_op(model):
    img = torch.randn(1,3,224,224)
    with torch.autograd.profiler.profile() as prof:
        out = model(img)
        loss = out.sum()
        loss.backward()
    file_name = "op.txt"
    with open(file_name, "w") as f:
        f.write(prof.key_averages().table(sort_by="self_cpu_time_total"))
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))

if __name__ == "__main__":
    model = UNet(3, 1)
    dump_op(model)