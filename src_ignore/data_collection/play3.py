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

def bigbird_attn(N, d, H, D, w, M, K):
    out = 8*N*d*d + (4*d+5)*N*M*K + (4*d+10)*N*M + (21*d+10)*N
    return out

def output_layer(d, dict_size):
    out = 2*dict_size*d
    return out


N = 4096
L = 12
d = 768
H = 12
D = 4096
w = 64
K = 8
dict_size = 30000
assert N % w == 0
M = N // w
#assert D // M == 64
compression_size = D // M
print(f"{compression_size=}")

Dense = int(dense_attn(N, d, H, D, w, M) / (1000000000))
Local = int(local_attn(N, d, H, D, w, M) / (1000000000))
Global = int(global_attn(N, d, H, D, w, M) / (1000000000))
Mine = int((Local + Global))
BigBird = int(bigbird_attn(N, d, H, D, w, M, K) / (1000000000))
MLP_out = int(MLP(N, d, H, D, w, M) / (1000000000))
Block = Dense + MLP_out
MyBlock = Mine + MLP_out
BigBirdBlock = BigBird + MLP_out
Output = int(output_layer(d, dict_size) / (1000000000))
Total = Block * L + Output
MyTotal = MyBlock * L + Output
BigBirdTotal = BigBirdBlock * L + Output

print(f"{N=}, {L=}, {d=}, {H=}, {D=}, {w=}, {K=}")
print(f"{Dense=}")
print(f"{Local=}")
print(f"{Global=}")
print(f"{Mine=}")
print(f"{BigBird=}")
print(f"{Block=}")
print(f"{MyBlock=}")
print(f"{BigBirdBlock=}")
print(f"{Total=}")
print(f"{MyTotal=}")
print(f"{BigBirdTotal=}")

