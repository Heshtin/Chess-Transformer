def dense_attn(N, d, H, D, w, M):
    out = 8*N*d*d + 4*N*N*d + 5*H*N*N + 5*N*D
    return out

def local_attn(N, d, H, D, w, M):
    out = 8*N*d*d + 4*N*M*d + 5*N*M*H + 5*N*d
    return out

def global_attn(N, d, H, D, w, M):
    out = 4*N*D*d*(1/M) + 8*D*D*w + 4*D*w*w + 5*M*w*w + 5*N*d
    return out

def MLP(N, d, H, D, w, M):
    out = 16*N*d*d + 9*N*d
    return out

def my_attn(N, d, H, D, w, M):
    out = local_attn(N, d, H, D, w, M) + global_attn(N, d, H, D, w, M) 
    return out

N = 16384
L = 12
d = 768
H = 12
D = 8192
w = 64
assert N % w == 0
M = N // w
#assert D // M == 64
compression_size = D // M
print(f"{compression_size=}")

Dense = int(dense_attn(N, d, H, D, w, M) / (1000000000))
Local = int(local_attn(N, d, H, D, w, M) / (1000000000))
Global = int(global_attn(N, d, H, D, w, M) / (1000000000))
Mine = int((Local + Global))
MLP_out = int(MLP(N, d, H, D, w, M) / (1000000000))
Block = Dense + MLP_out
MyBlock = Mine + MLP_out
Total = Block * L
MyTotal = MyBlock * L

print(f"{Dense=}")
print(f"{Local=}")
print(f"{Global=}")
print(f"{Mine=}")
print(f"{Block=}")
print(f"{MyBlock=}")
print(f"{Total=}")
print(f"{MyTotal=}")

