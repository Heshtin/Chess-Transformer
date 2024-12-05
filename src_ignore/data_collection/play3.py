def dense_attn(N, d, H, D, w, M):
    out = 8*N*d*d + 4*N*N*d + 5*H*N*N + 5*N*D
    return out

# def local_attn(N, d, H, D, w, M):
#     out = 8*N*d*d + 4*N*M*d + 5*N*M*H + 5*N*d
#     return out
def local_attn(N, d, H, D, w, M):
    out = 8*N*d*d + 1.25*1.25*4*M*M*w*d + 1.25*1.25*5*M*M*w*H + 5*N*d
    return out
# def global_attn(N, d, H, D, w, M):
#     out = 4*N*D*d*(1/M) + 8*D*D*w + 4*D*w*w + 5*M*w*w + 5*N*d
#     return out

# def global_attn(N, d, H, D, w, M):
#     out = 4*N*D*d*(1/M) + 8*D*D*w + 4*D*w*w + 5*H*w*w + 5*N*d + 2*D*d*N*(H/M) + 2*N*d*d*(H/M)
#                           #6 or 8? (6)
#     return out

def global_attn(N, d, H, D, w, M, k):
    out = 8*D*D*w + 4*D*w*w + 5*M*w*w*H + 4*N*d*k + 2*N*D*(k+1) + 5*N*d
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
H = 10
D = 4096
w = 32
h=16
dict_size = 30000
assert N % w == 0
M = N // w

#assert D // M == 64
compression_size = D // M
print(f"{compression_size=}")
k = D // M
a=0


if a==0:
    bb_N = 4096
    bb_L = 12
    bb_d = 768
    bb_H = 12
    bb_D = 4096
    bb_w = 64
    bb_K = 8
elif a==1:
    bb_N = 12288
    bb_L = 12
    bb_d = 768
    bb_H = 12
    bb_D = 16384
    bb_w = 64
    bb_K = 8
else:
    bb_N = N
    bb_L = L
    bb_d = d
    bb_H = H
    bb_D = D
    bb_w = w
    bb_K = 32
assert bb_N % bb_w == 0
bb_M = bb_N // bb_w




Dense = int(dense_attn(N, d, H, D, w, M) / (1000000000))
Local = int(local_attn(N, d, H, D, w, M) / (1000000000))
Global = int(global_attn(N, d, h, D, w, M, k) / (1000000000))
Mine = int((Local + Global))
BigBird = int(bigbird_attn(bb_N, bb_d, bb_H, bb_D, bb_w, bb_M, bb_K) / (1000000000))
MLP_out = int(MLP(N, d, H, D, w, M) / (1000000000))
bb_MLP_out = int(MLP(bb_N, bb_d, bb_H, bb_D, bb_w, bb_M) / (1000000000))
Block = Dense + MLP_out
MyBlock = Mine + MLP_out
BigBirdBlock = BigBird + bb_MLP_out
Output = int(output_layer(d, dict_size) / (1000000000))
Total = Block * L + Output
MyTotal = MyBlock * L + Output
BigBirdTotal = BigBirdBlock * bb_L + Output

print(f"{N=}, {L=}, {d=}, {H=}, {D=}, {w=}")
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

