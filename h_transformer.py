import torch.nn as nn
from torch import tensor
import copy
import pickle
import torch
from torch.nn import Dropout, ModuleList
from typing import Optional, Tuple, Union, Callable, Any
from torch import Tensor
from torch.nn import functional as F
from torch.nn.init import constant_, xavier_uniform_, xavier_normal_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear as c_linear
from torch.nn.modules.linear import Linear
from torch.nn import LayerNorm

'''
This is a reproduce of transformer in production enviroment,
worth to mention, all functions start with "_" are not pulically showed,
if you need pull attention out, those function needs to be modified

'''





class transformer_encoder_layer(nn.Module):

    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model:int, n_head:int, dim_forward:int = 2048, dropout:float = 0.1,
                activation:Union[str, Callable[[Tensor], Tensor]] = F.relu,
                layer_norm_eps:float = 1e-8, batch_first: bool = False, norm_first:bool = False,
                device = None, dtype=None):
        h_kwargs = {'device': device, 'dtype':dtype}
        super(transformer_encoder_layer, self).__init__()
        self.self_attn = transformer_multi_attention(d_model, n_head, dropout=dropout,
        batch_first=batch_first, **h_kwargs)
        self.Linear1 = Linear(d_model, dim_forward, **h_kwargs)
        self.dropout = Dropout(dropout)
        self.Linear2 = Linear(dim_forward, d_model, **h_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **h_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **h_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        if activation is F.relu:
            print('choosing the default activation function')
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu:
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super(transformer_encoder_layer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(self, src:Tensor, src_mask:Optional[tensor] = None,
                src_key_padding_mask:Optional[Tensor] = None) -> None:
        if (src.dim() == 3 and not self.norm_first and not self.training and
            self.self_attn.batch_first and
            self.self_attn._QKV_same_dim and self.activation_relu_or_gelu and
            self.norm1.eps == self.norm2.eps and
            src_mask is None and not
            (src.is_nested and src_key_padding_mask is not None)):
            # print('wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww')
            tensor_args = (
                src,
                self.self_attn.direct_embedding,
                self.self_attn.bias_direct_embedding,
                self.self_attn.out_projection.weight,
                self.self_attn.out_projection.bias,
                self.norms1.weight,
                self.norms1.bias,
                self.norms2.weight,
                self.norms2.bias,
                self.Linear1.weight,
                self.Linear1.bias,
                self.Linear.weigth,
                self.Linear.bias,
            )
            if (not torch.overrides.has_torch_function(tensor_args) and
                all([x.is_cuda or 'cpu' in str(x.device) for x in tensor_args]) and
                (not torch.is_grad_enabled() or all([not x.requires_grad for x in tensor_args]))):
                print('checking correct')
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.direct_embedding,
                    self.self_attn.bias_direct_embedding,
                    self.self_attn.out_projection.weight,
                    self.self.attn.out_projection.bias,
                    self.activation_relu_or_gelu == 2,
                    False,
                    self.norm1.eps,
                    self.norm1.weigth,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.Linear1.weight,
                    self.Linear1.bias,
                    self.Linear2.weight,
                    self.Linear2.bias,
                    src_mask if src_mask is not None else src_key_padding_mask,
                )
        x = src
        print(x.dim())
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x


    # the self attention block
    def _sa_block(self, x:Tensor,
                    attn_mask:Optional[Tensor], key_padding_mask:Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                            attn_mask=attn_mask,
                            K_padding_mask=key_padding_mask,
                            attn_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x:Tensor) -> Tensor:
        x = self.Linear2(self.dropout(self.activation(self.Linear1(x))))
        return self.dropout2(x)

#TODO
'''
BERT could be built through the following encoder
'''
class transformer_encoder(nn.Module):
    __constant__ = ['norm']
    def __init__(self, encoder_layer, num_layers, norm = None, enable_nested_tensor=False):
        super(transformer_encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.enabled_nested_tensor = enable_nested_tensor

    def forward(self, src:Tensor, mask:Optional[Tensor] = None, src_key_padding_mask:
                Optional[Tensor] = None) -> None:
        output = src
        convert_to_nested =False
        first_layer = self.layers[0]

        if isinstance(first_layer, transformer_encoder_layer):
            if (not first_layer.norm_first and not first_layer.training and
                first_layer.self_attn.batch_first and
                first_layer.self_attn._QKV_same_dim and
                first_layer.activation_relu_or_gelu and
                first_layer.norm1.eps == first_layer.norm2.eps and
                src.dim() == 3 and self.enable_nested_tensor):
                if src_key_padding_mask is not None and output.is_nested and mask is None:
                    tensor_args = (
                        src,
                        first_layer.self_attn.direct_embedding,
                        first_layer.self_attn.bias_direct_embedding,
                        first_layer.self_attn.out_projection.weight,
                        first_layer.self_attn.out_projection.bias,
                        first_layer.norm1.weight,
                        first_layer.norm1.bias,
                        first_layer.norm2.weight,
                        first_layer.norm2.bias,
                        first_layer.Linear1.weight,
                        first_layer.Linear1.bias,
                        first_layer.Linear2.weight,
                        first_layer.Linear2.bias,
                    )
                    if not torch.overrides.has_torch_function(tensor_args):
                        if not torch.is_grad_enabled() or all([x.requires_grad for x in
                        tensor_args]):
                            if output.is_cuda or 'cpu' in str(output.device):
                                convert_to_nested = True
                                output = torch._nested_tensor_from_mask(output,
                                src_key_padding_mask.logical_not())
        for mod in self.layers:
            if convert_to_nested:
                output = mod(output, src_mask=mask)
            else:
                output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if convert_to_nested:
            output = output.to_padded_tensor(0.)
        if self.norm is not None:
            output = self.norm(output)

        return output


class transformer_decoder_layer(nn.Module):

    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, d_model:int, num_heads:int, dim_forward:int=2048, dropout:float=0.1,
                activation:Union[str, Callable[[Tensor], Tensor]] = F.relu,
                layer_norm_eps:float = 1e-8, batch_first:bool = False, norm_first:bool = False,
                device = None, dtype=None
                ) -> None:
        h_kwargs = {'device':device, 'dtype':dtype}
        super(transformer_decoder_layer, self).__init__()
        self.self_attn = transformer_multi_attention(d_model, num_heads, dropout=dropout,
        batch_first=batch_first, **h_kwargs)
        self.multihead_attn = transformer_multi_attention(d_model, num_heads, dropout=dropout,
        batch_first=batch_first, **h_kwargs)
        self.Linear1 = Linear(d_model, dim_forward, **h_kwargs)
        self.dropout = Dropout(dropout)
        self.Linear2 = Linear(dim_forward, d_model, **h_kwargs)
        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps = layer_norm_eps, **h_kwargs)
        self.norm2 = LayerNorm(d_model, eps = layer_norm_eps, **h_kwargs)
        self.norm3 = LayerNorm(d_model, eps = layer_norm_eps, **h_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(transformer_decoder_layer, self).__setstate__(state)

    def forward(self, target:Tensor, memory:Tensor, target_mask:Optional[Tensor] = None,
                memory_mask:Optional[Tensor] = None, target_key_padding_mask:Optional[Tensor] = None,
                memory_key_padding_mask:Optional[Tensor]= None) -> None:
        x = target
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), target_mask, target_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = self.norm3(x + self._ff_block(x))
        return x

    def _sa_block(self, x:Tensor,
                    attn_mask:Optional[Tensor], key_padding_mask:Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=False)[0]
        return x

    def _mha_block(self, x:Tensor, mem:Tensor,
                    attn_mask:Optional[Tensor], key_padding_mask:Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask = attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    def _ff_block(self, x:Tensor) -> Tensor:
        x = self.Linear2(self.dropout(self.activation(self.Linear1(x))))
        return self.dropour3(x)


class transformer_decoder(nn.Module):

    __constant__ = ['norm']
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(transformer_decoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, target:Tensor, memory:Tensor, target_mask:Optional[Tensor]=None,
                memory_mask:Optional[Tensor]=None, target_key_padding_mask:Optional[Tensor]=None,
                memory_key_padding_mask:Optional[tensor]=None) -> Tensor:
        output = target

        for mod in self.layers:
            output = mod(output, memory, target_mask=target_mask,
                        memory_mask=memory_mask,
                        target_key_padding_mask=target_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)

        return output


class transformer_multi_attention(nn.Module):


    __constants__= ['batch_first']
    """
    params:
        embed_dim: the last dimension's length of the model
        num_heads: how many attention heads you want to split
        dropout: regularizer to avoid overfitting
        K_dim, V_dim and embed_dim are with equal size by default
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True, K_dim=None, V_dim=None,
                KV_add_bias=False, K_bias= False, V_bias=False, add_zero_attn=False,
                batch_first=True, device=None, dtype=None) -> None:
                super(transformer_multi_attention, self).__init__()
                self.embed_dim = embed_dim
                print('the embed_dim is ', embed_dim)
                self.num_heads = num_heads
                self.dropout = dropout
                self.bias = bias
                self.K_dim = K_dim if K_dim is not None else embed_dim
                print('The K_dim is ', self.K_dim)
                self.V_dim = V_dim if V_dim is not None else embed_dim
                self._QKV_same_dim = self.K_dim == embed_dim and self.V_dim == embed_dim
                self.batch_first = batch_first
                self.head_dim = embed_dim // num_heads
                assert self.head_dim * num_heads == self.embed_dim
                parameter_settings = {'dtype': dtype, 'device': device}
                if not self._QKV_same_dim:
                    self.Q_weight_matrix = nn.Parameter(torch.empty(embed_dim, embed_dim))
                    self.K_weight_matrix = nn.Parameter(torch.empty((embed_dim, self.K_dim),
                    **parameter_settings))
                    self.V_weight_matrix = nn.Parameter(torch.empty((embed_dim, self.V_dim),
                    **parameter_settings))
                    self.register_parameter('direct_embedding', None)
                else:
                    self.direct_embedding = nn.Parameter(torch.empty(3*embed_dim, embed_dim))
                    print('QKV now the same size, direct embedding is ', self.direct_embedding.size())
                    self.register_parameter('K_weight_matrix', None)
                    # print('K_weight_matrix is', K_weight_matrix)
                    self.register_parameter('Q_weight_matrix', None)
                    self.register_parameter('V_weight_matrix', None)
                # justify if the bias should be add to the input and output projection layers
                if bias:
                    self.bias_projection = nn.Parameter(torch.empty(3*embed_dim, **parameter_settings))
                    print('the bias_projection is', self.bias_projection.size())
                else:
                    self.register_parameter('bias_direct_embedding', None)
                self.out_projection = c_linear(embed_dim, embed_dim, bias = bias, **parameter_settings)
                print('the out_projection is', self.out_projection)
                # if K, v with differnt size, you will need to make it the same
                if KV_add_bias:
                    self.K_bias = nn.Parameter(torch.empty((1, 1, embed_dim), **parameter_settings))
                    self.V_bias = nn.Parameter(torch.empty((1, 1, embed_dim), **parameter_settings))
                else:
                    self.K_bias = self.V_bias = None
                self.add_zero_attn = add_zero_attn
                self._reset_parameters()

    def _reset_parameters(self):
                if self._QKV_same_dim:
                    xavier_uniform_(self.direct_embedding)
                else:
                    xavier_uniform_(self.K_weight_matrix)
                    xavier_uniform_(self.Q_weight_matrix)
                    xavier_uniform_(self.V_weight_matrix)

                if self.bias_projection is not None:
                    constant_(self.bias_projection, 0.)
                    constant_(self.out_projection.bias, 0.)

                if self.K_bias is not None:
                    xavier_normal_(K_bias)
                if self.V_bias is not None:
                    xavier_normal_(V_bias)

    #TODO: understand the deeper code of this problem
    def __setstate__(self, state):

        if '_QKV_same_dim' not in state:
            state['_QKV_same_dim'] = True
        super(transformer_multi_attention, self).__setstate__(state)

    def forward(self, Q:Tensor, K:Tensor, V:Tensor, K_padding_mask: Optional[Tensor] = None,
                attn_weights:bool = True, attn_mask: Optional[Tensor] = None,
                avg_attn_weights:bool = False) -> Tuple[Tensor, Optional[Tensor]]:
                print('entering forward process')
                is_batched = Q.dim() == 3
                why_not_batch = ''
                if not is_batched:
                    why_not_batch = f'the input fed in was not batched, expect the query dimension \
                    to be 3, yet got{Q.dim()}'
                elif Q is not K or K is not V:
                    why_not_batch = 'The Q, K and V are not using the same Tensor'
                elif self.bias_projection is not None and Q is not self.bias_projection:
                    why_not_batch = f"the query dtype {Q.dtype} is not the same with bias \
                    type{self.bias_projection.dtype}"
                elif self.direct_embedding is not None and Q is not self.direct_embedding:
                    why_not_batch = f"the query dtype {Q.dtype} is not the same with direct \
                    embedding weight \
                    type{self.bias_projection.dtype}"
                elif self.training:
                    why_not_batch = 'training mode is open'
                elif not self.batch_first:
                    why_not_batch = 'the batch is not the first dimension'
                elif self.K_bias is not None:
                    why_not_batch = 'K_bias is now added'
                elif self.V_bias is not None:
                    why_not_batch = 'V_bias is now added'
                elif self.dropout:
                    why_not_batch = f'dropout now is {self.dropout}, required 0'
                elif self.add_zero_attn:
                    why_not_batch = 'the add zero to attention is enabled'
                elif not self._QKV_same_dim:
                    why_not_batch = 'Q, K, V are not have the same dimension'
                elif attn_mask is not None:
                    why_not_batch = 'attention mask is not none'
                elif self.dropout:
                    why_not_batch = f'dropout now is {self.dropout}, required 0'
                elif Q.is_nested and K_padding_mask is not None:
                    why_not_batch = 'K_padding mask is not supported for nested tensors'

                print('why_not_batch is ', why_not_batch)
                if not why_not_batch:
                    print('you are doing batch')
                    tensor_settings = (
                    Q,
                    K,
                    V,
                    self.direct_embedding,
                    self.bias_projection,
                    self.out_projection,
                    self.out_projection.bias
                    )
                    if not all([(x.is_cuda or 'cpu' in str(x.device)) for x in tensor_settings]):
                        why_not_batch = 'some tensor is neither cpu nor cuda'
                    elif torch.is_grad_enabled() and any([x.requires_grad for x in tensor_settings]):
                        why_not_batch = 'grad is enbaled and at least one of query ot the \
                        input/output projection weights or bias requires grad'
                    if not why_not_batch:
                        return torch._native_multi_head_attention(
                        Q,
                        K,
                        V,
                        self.embed_dim,
                        self.num_heads,
                        self.direct_embedding,
                        self.bias_projection,
                        self.out_projection,
                        self.out_projection.bias,
                        K_padding_mask if K_padding_mask is not None else attn_mask,
                        attn_weights,
                        avg_attn_weights,
                        1 if K_padding_mask is not None else 0 if attn_mask is not None else None
                        )

                    assert not any[Q.is_nested, K.is_nested, V.is_nested]

                if self.batch_first and is_batched:
                    if K is V:
                        if Q is K:
                            Q = K = V = Q.transpose(1, 0)
                        else:
                            Q, K = [x.transpose for x in (Q, K)]
                            V = K
                    else:
                        Q, K, V = [x.transpose(1, 0) for x in (Q, K, V)]

                if not self._QKV_same_dim:
                    attn_output, attn_output_weights = F.multi_head_attention_forward(
                    Q, K, V, self.embed_dim, self.num_heads,
                    self.direct_embedding, self.bias_projection,
                    self.K_bias, self.V_bias, self.add_zero_attn,
                    self.dropout, self.out_projection.weight, self.out_projection.bias,
                    training = self.training, key_padding_mask = K_padding_mask,
                    need_weights= attn_weights, use_seperate_proj_weight=True,
                    q_proj_weight = self.Q_weight_matrix,
                    k_proj_weight = self.K_weight_matrix,
                    v_proj_weight = self.V_weight_matrix,
                    average_attn_weights=avg_attn_weights
                    )
                else:
                    print('QKV are the same dimension')
                    attn_output, attn_output_weights = F.multi_head_attention_forward(
                    Q, K, V, self.embed_dim, self.num_heads,
                    self.direct_embedding, self.bias_projection,
                    self.K_bias, self.V_bias, self.add_zero_attn,
                    self.dropout, self.out_projection.weight, self.out_projection.bias,
                    training = self.training, key_padding_mask = K_padding_mask,
                    need_weights= attn_weights, attn_mask=attn_mask,
                    average_attn_weights=avg_attn_weights
                    )
                if self.batch_first and is_batched:
                    print('data is batched and batch_first')
                    return attn_output.transpose(1, 0), attn_output_weights
                else:
                    return attn_output, attn_output_weights


class transformer(nn.Module):
    '''
    Transformer class requires the input feature == model dimension  ==  output feature
    '''
    def __init__(self, d_model:int=512, num_heads:int=8, num_encoder_layers:int=6,
                num_decoder_layers:int=6, dim_forward:int=2048, dropout:float=0.1,
                activation:Union[str, Callable[[Tensor], Tensor]] = F.relu,
                custome_encoder:Optional[Any] =None, custome_decoder:Optional[Any]=None,
                layer_norm_eps:float= 1e-8, batch_first:bool=False, norm_first:bool=False,
                device = None, dtype=None
                ) -> None:
        h_kwargs = {'device': device, 'dtype':dtype}
        super(transformer, self).__init__()
        if custome_encoder is not None:
            self.encoder = custome_encoder
        else:
            encoder_layer = transformer_encoder_layer(d_model, num_heads, dim_forward,
            dropout, activation, layer_norm_eps, batch_first, norm_first, **h_kwargs)
            encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **h_kwargs)
            self.encoder = transformer_encoder(encoder_layer, num_encoder_layers,
            encoder_norm)
        if custome_decoder is not None:
            self.decoder =custome_decoder
        else:
            decoder_layer = transformer_decoder_layer(d_model, num_heads, dim_forward,
            dropout, activation, layer_norm_eps, batch_first, norm_first, **h_kwargs)
            decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **h_kwargs)
            self.decoder = transformer_decoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()
        self.d_model = d_model
        self.num_heads = num_heads

        self.batch_first = batch_first

    def forward(self, src:Tensor, target:Tensor, src_mask:Optional[Tensor]=None,
                target_mask:Optional[Tensor]=None, memory_mask:Optional[Tensor]=None,
                src_key_padding_mask:Optional[Tensor]=None,
                target_key_padding_mask:Optional[Tensor]=None,
                memory_key_padding_mask:Optional[Tensor]=None) -> Tensor:
        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) != target.size(1) and is_batched:
            raise RuntimeError('the batch size of input and output should be the same')
        elif self.batch_first and src.size(0) != target.size(0) and is_batched:
            raise RuntimeError('the batch size of input and output should be the same')

        if src.size(-1) != self.d_model or target.size(-1) != self.d_model:
            raise RuntimeError('the feature number of the src and target must equal to d_model')

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(target, memory, target_mask=target_mask, memory_mask=memory_mask,
                                target_key_padding_mask=target_key_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask)
        # print(output.size())
        output = F.softmax(output, -1)

        return output


    @staticmethod

    def generate_square_subsequent_mask(sz:int) -> Tensor:
        return torch.triu(torch.full((sz, sz), float('inf')), diagonal=1)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation:str) -> Callable[[Tensor], Tensor]:
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu

    return RuntimeError('you should either choose gelu or relu, not{}'.format(activation))


# multi_attn = transformer_multi_attention(10, 2)
encoder_layer = transformer_encoder_layer(10, 2, batch_first=True)
encoder = transformer_encoder(encoder_layer, 4)
src = torch.rand(3, 6, 10)
en_out = encoder(src)

tgr = torch.rand(3, 6, 10)
decoder_layer = transformer_decoder_layer(10, 2, batch_first=True)
decoder = transformer_decoder(decoder_layer, num_layers=4)
out = decoder(tgr, en_out)

transformer = transformer(d_model = 10, num_heads = 2, num_encoder_layers = 4)
out  = transformer(src, tgr)

print(out)





# key = torch.rand(5, 6, 10)
# query = torch.rand(5, 6, 10)
# value = torch.rand(5, 6, 10)

# print(key.size(), query.size(), value.size())
# attn_output, attn_weight_output = multi_attn(key, query, value)
#
#
# print(attn_output, attn_output.size(), attn_weight_output, attn_weight_output.size())

    # w = torch.empty(3, 5)
    # x_w = nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    #
    # if w is not x_w:
    #     print('x is not x_w')
    #
    # a = ''
    #
    # if not a:
    #     print('test')
    #
    # print(w, x_w)
    #
    # b = 3
    # c = 2
    # a = b == 3 and c == 3
    # print(a)
    #
    # __constant__ = ['nima', 'gebi']
    #
    # print(__constant__, type(__constant__))






if __name__ == '__main__':
    # basics()
    pass








































print('This is Mr placeholder')
