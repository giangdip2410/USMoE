r"""
Adaption to act as the MLP layer using an MoE MLP layer in transformer.
"""
import torch
import torch.nn as nn
from custom_layers import FMoE
from linear import FMoELinear
from transformers.activations import ACT2FN
import torch.nn.functional as F



class OlmoeMLP(nn.Module):
    def __init__(self, d_model, d_hidden, activation):
        super().__init__()
        self.hidden_size = d_model
        self.intermediate_size = d_hidden
        self.htoh4 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.h4toh = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = activation

    def forward(self, x):
        return self.h4toh(self.act_fn(self.htoh4(x)))

class FMoETransformerMLP(nn.Module):
    def __init__(self, num_expert=32, top_k=2, norm_topk_prob=False,
                d_model=1024,
                d_hidden=4096,
                activation=torch.nn.GELU(),
                expert_dp_comm="none",
                expert_rank=0,
                **kwargs):
        super().__init__()
        self.num_experts = num_expert
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.gate = nn.Linear(d_model, self.num_experts, bias=False)
        self.experts = nn.ModuleList([OlmoeMLP(d_model, d_hidden, activation) for _ in range(self.num_experts)])
        self.rate = 0.5

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        router_logits_reshape = torch.sigmoid(router_logits) * 2.0 / self.num_experts
        # router_logits.reshape(batch_size, sequence_length, -1)
        routing_weights = routing_weights * (1-self.rate) + router_logits_reshape * self.rate
        # select the best
        router_logits_reshape = routing_weights.reshape(batch_size, -1)
        topk = sequence_length * self.top_k
        _, gate_top_k_idx = torch.topk(
                router_logits_reshape, k=topk, dim=-1, largest=True, sorted=False
            )
        all_idx = torch.zeros_like(router_logits_reshape, dtype=torch.bool)
        all_idx = all_idx.scatter_(1, gate_top_k_idx, True)
        all_idx = all_idx.reshape(batch_size * sequence_length, -1)
        #routing_weights = F.softmax(router_logits_reshape.reshape(-1, self.num_experts), dim=1, dtype=torch.float)
        #routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be selected
        #expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            #idx, top_x = torch.where(expert_mask[expert_idx])
            indices = all_idx[:, expert_idx]
            scores = routing_weights[:, expert_idx][indices]
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[indices] #hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * scores[:, None] #routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states[indices] += current_hidden_states #final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        
        return final_hidden_states
