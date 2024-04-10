

from einops import pack, unpack
import torch
from torch import nn
from einops import rearrange, repeat
from einops import repeat
from zmq import device

num = 0
attn_mean_cat = None
depth_of_MHA_in_transformer_in_ViT = 3

class TensorConcater:
  def __init__(self):
    self.stored_tensor = None
  
  def clear(self):
    self.stored_tensor = None

  def update(self, new_tensor, axis=1):
    x = new_tensor.cpu()   
    x = torch.atleast_2d(x)
    if self.stored_tensor is None:
      self.stored_tensor = x
    else:
      self.stored_tensor = torch.concat([self.stored_tensor, x])

  def get(self) -> torch.Tensor:
    if self.stored_tensor is None:
      raise ValueError("Concater has not been updated with any tensor yet.")
    return self.stored_tensor


class ViTmod_print(nn.Module):
  def __init__(self, *,
               num_classes, 
               dim, 
               depth, 
               heads, 
               mlp_dim,
               dim_head=64, 
               FF_dropout=0., 
               emb_dropout=0., 
               pathway_number, 
               categories,
               external_matrix=None,
               ):
    super().__init__()
    print("---------Using external matrix---------")
    self.categories = categories
    self.mutil_linear_layers = nn.ModuleList([])

    for i in range(len(pathway_number)):
      self.mutil_linear_layers.append(nn.Sequential(
          nn.LayerNorm(pathway_number[i] + 1),
          nn.Linear(pathway_number[i] + 1, dim),
          nn.LayerNorm(dim)
      ))

    self.cls_token = nn.Parameter(torch.randn(dim))
    self.dropout = nn.Dropout(emb_dropout)
    self.transformer = Transformer_printer(dim, depth, heads, dim_head, mlp_dim, FF_dropout, external_matrix)
    self.mlp_head = nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, num_classes)
    )
    self.concater = TensorConcater()  
    self.to_segment_num = pathway_number

  def forward(self, genes):  
    genes = genes.view(genes.shape[0], 1, -1)  
    pathways = torch.split(genes, self.to_segment_num, dim=-1)  
    added_pathways = self.append_category(pathways)
    embed_avg_list = []
    
    pathway_embedding_list = []
    for idx, layers in enumerate(self.mutil_linear_layers):
      per_segment = layers(added_pathways[idx])  
      pathway_embedding_list.append(per_segment)  
      segment_avg_float = torch.mean(torch.squeeze(per_segment), dim=0)
      embed_avg_list.append(segment_avg_float.item())  
      
    embed_avg_tensor = torch.tensor(embed_avg_list, device='cpu', requires_grad=False)  
    self.concater.update(embed_avg_tensor)  
    
    x = torch.cat(pathway_embedding_list, dim=-2)
    b, n, _ = x.shape  
    cls_token = repeat(self.cls_token, 'd -> b d', b=b)  
    x, ps = pack([cls_token, x], 'b * d')  
    x = self.dropout(x)
    x = self.transformer(x)
    cls_token, _ = unpack(x, ps, 'b * d')
    return self.mlp_head(cls_token)

  def append_category(self, pathways):
    added_pathways = []
    if len(pathways) != len(self.categories):
      raise ValueError("The number of segments and categories should be the same.")

    for pathway, category in zip(pathways, self.categories):
      
      category_tensor = torch.tensor([category], dtype=pathway.dtype, device=pathway.device)

      
      
      
      category_tensor = category_tensor.expand(pathway.shape[0], 1, 1)

      
      modified_pathway = torch.cat((pathway, category_tensor), dim=-1)
      added_pathways.append(modified_pathway)

    return added_pathways



class Attention_printer(nn.Module):
  def __init__(self, dim, heads=8, dim_head=64, dropout=0., pathway_bias=None):
    super().__init__()
    inner_dim = dim_head * heads
    project_out = not (heads == 1 and dim_head == dim)

    self.heads = heads
    self.scale = dim_head ** -0.5

    self.attend = nn.Softmax(dim=-1)
    self.dropout = nn.Dropout(dropout)

    self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

    self.project = nn.Sequential(
        nn.Linear(inner_dim, dim),
        nn.Dropout(dropout)
    ) if project_out else nn.Identity()
    
    self.pathway_bias = None
    if pathway_bias is not None:
        self.pathway_bias = nn.Parameter(pathway_bias, requires_grad=False)
    self.pathway_transform = nn.Linear(dim, dim)
    
  def forward(self, x):
    qkv = self.to_qkv(x).chunk(3, dim=-1)
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

    dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
    if self.pathway_bias is not None:
      dots += self.pathway_bias
    attn = self.attend(dots) 
    attn_dropped = self.dropout(attn)  
    out = torch.matmul(attn_dropped, v)
    out_re = rearrange(out, 'b h n d -> b n (h d)')  
    out_proj = self.project(out_re)  

    global num
    num += 1
    attn_mean = torch.mean(torch.mean(attn, dim=1), dim=1).cpu().detach()  
    global attn_mean_cat
    if num % depth_of_MHA_in_transformer_in_ViT == 0:
      if attn_mean_cat == None:
        attn_mean_cat = attn_mean.clone()
      else:
        attn_mean_cat = torch.cat([attn_mean_cat, attn_mean], dim=0)
    return out_proj



class Transformer_printer(nn.Module):
  def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., pathway_bias=None):
    super().__init__()
    self.pathway_bias = pathway_bias 
    self.layers = nn.ModuleList([])
    for _ in range(depth):
      self.layers.append(nn.ModuleList([
          PreNorm(dim, Attention_printer(dim, heads, dim_head, dropout, pathway_bias)),
          PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
      ]))


  def forward(self, x):
    for ATTN, FF in self.layers:
      x = ATTN(x) + x
      x = FF(x) + x
    return x


class PreNorm(nn.Module):
  def __init__(self, dim, fn):
    super().__init__()
    self.norm = nn.LayerNorm(dim)
    self.fn = fn

  def forward(self, x, **kwargs):
    return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
  def __init__(self, dim, hidden_dim, dropout=0.):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)
