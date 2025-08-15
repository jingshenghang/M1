# from mindspeed_llm.tasks.models.ssm.state_space_duality import StateSpaceProcessor, ProcessInputs, StateOptions
import torch
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from einops import rearrange, repeat


nheads_local = 32
ngroups_local = 4
dt_min = 0.001
dt_max = 0.1
headdim = 128
d_state = 128
chunk_size = 256
D_has_hdim = False


path = "/home/data/mamba_tensor_2/"

x = torch.load(path + "x.pt").to(torch.float32).cuda()
A = torch.load(path + "A.pt").to(torch.float32).cuda()
B = torch.load(path + "B.pt").to(torch.float32).cuda()
C = torch.load(path + "C.pt").to(torch.float32).cuda()
D = torch.load(path + "D.pt").to(torch.float32).cuda()
dt = torch.load(path + "dt.pt").to(torch.float32).cuda()
dt_bias = torch.load(path + "dt_bias.pt").to(torch.float32).cuda()


config = {
        'nheads_local': nheads_local,
        'ngroups_local': ngroups_local,
        'dt_min': dt_min,
        'dt_max': dt_max,
        'dt_bias': dt_bias,
        'headdim': headdim,
        'd_state': d_state,
        'chunk_size': chunk_size,
        'D_has_hdim': D_has_hdim
    }

'''
inputs = ProcessInputs(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D
    )

state_opts = StateOptions(
        return_final_state=False
    )

state_space_duality = StateSpaceProcessor(config=config)
y = state_space_duality.process(inputs, state_opts)
print(y)

'''

repeat_group = 8
d_state = 128
# z = ??????
z = None
ngroups = 32
rmsnorm = True
seq_idx = None
cu_seqlens = None
dt_limit_kwargs = {}
ssm_state = None
inference_params = None


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

print("A.shape = {}".format(A.shape))
print("B.shape = {}".format(B.shape))
print("C.shape = {}".format(C.shape))
print("D.shape = {}".format(D.shape))
print("x.shape = {}".format(x.shape))
print("dt.shape = {}".format(dt.shape))
print("dt_bias.shape = {}".format(dt_bias.shape))


# minic the GQA
x = rearrange(x, "b l (xb_group dstate) -> b xb_group l dstate", dstate=d_state)
x = repeat_kv(x, repeat_group)
# x shape: (bsz, n_group, l, dim)

B = rearrange(B, "b l (xb_group dstate) -> b xb_group l dstate", dstate=d_state)
B = repeat_kv(B, repeat_group)


print("--------new---------")

print("A.shape = {}".format(A.shape))
print("B.shape = {}".format(B.shape))
print("C.shape = {}".format(C.shape))
print("D.shape = {}".format(D.shape))
print("x.shape = {}".format(x.shape))
print("dt.shape = {}".format(dt.shape))
print("dt_bias.shape = {}".format(dt_bias.shape))


print("--------new_rerange---------")

print("A.shape = {}".format(A.shape))
print("B.shape = {}".format(rearrange(B, "b g l n -> b l g n").shape))
print("C.shape = {}".format(rearrange(C, "b l (g n) -> b l g n", g=ngroups).shape))
print("D.shape = {}".format(D.shape))
print("x.shape = {}".format(rearrange(x, "b g l p -> b l g p").shape))
print("dt.shape = {}".format(dt.shape))
print("dt_bias.shape = {}".format(dt_bias.shape))


y = mamba_chunk_scan_combined(
    # rearrange(x, "b l (h p) -> b l h p", p=headdim),
    rearrange(x, "b g l p -> b l g p"),
    dt,
    A,
    # rearrange(B, "b l (g n) -> b l g n", g=ngroups),
    rearrange(B, "b g l n -> b l g n"),
    rearrange(C, "b l (g n) -> b l g n", g=ngroups),
    chunk_size=chunk_size,
    D=rearrange(D, "(h p) -> h p", p=headdim) if D_has_hdim else D,
    z=rearrange(z, "b l (h p) -> b l h p", p=headdim) if not rmsnorm else None,
    dt_bias=dt_bias,
    dt_softplus=True,
    seq_idx=seq_idx,
    cu_seqlens=cu_seqlens,
    **dt_limit_kwargs,
    return_final_states=ssm_state is not None,
    return_varlen_states=cu_seqlens is not None and inference_params is not None,
)

print(y)
print(y.mean())
print(y.max())
print(y.min())
