"""Complete self-attention with checkpointed query chunks; no token/key pruning."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _attention(q,k,v):
    return F.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=False)


def query_chunked_transformer(encoder,x,*,query_chunk_size):
    """Single observation [N,E]. Every query still attends to all N keys/values."""
    if type(query_chunk_size) is not int or query_chunk_size<1 or x.ndim!=2 or len(x)<1:
        raise ValueError('Positive query chunk and nonempty single-observation [N,E] required')
    for layer in encoder.layers:
        attn=layer.self_attn
        if not layer.norm_first or not attn.batch_first or layer.activation_relu_or_gelu!=2:
            raise ValueError('Only the existing pre-norm batch-first GELU architecture is supported')
        if any(module.p for module in (layer.dropout,layer.dropout1,layer.dropout2)) or attn.dropout:
            raise ValueError('Stochastic dropout needs separate RNG-equivalence acceptance')
        if not attn._qkv_same_embed_dim or attn.bias_k is not None or attn.bias_v is not None or attn.add_zero_attn:
            raise ValueError('Only the existing shared self-attention QKV structure is supported')
        normalized=layer.norm1(x);q,k,v=F.linear(normalized,attn.in_proj_weight,attn.in_proj_bias).chunk(3,-1)
        n,width=x.shape;heads=attn.num_heads;dim=width//heads
        q,k,v=(tensor.reshape(1,n,heads,dim).transpose(1,2) for tensor in (q,k,v))
        parts=[]
        for block in q.split(query_chunk_size,dim=2):
            # All K/V stay in the graph. Checkpoint discards quadratic activations,
            # then recomputes each query block during its backward pass.
            if torch.is_grad_enabled() and any(t.requires_grad for t in (block,k,v)):
                value=checkpoint(_attention,block,k,v,use_reentrant=False,preserve_rng_state=False)
            else:value=_attention(block,k,v)
            parts.append(value)
        merged=torch.cat(parts,dim=2).transpose(1,2).reshape(n,width)
        x=x+layer.dropout1(F.linear(merged,attn.out_proj.weight,attn.out_proj.bias))
        hidden=layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(x)))))
        x=x+layer.dropout2(hidden)
    return encoder.norm(x) if encoder.norm is not None else x
