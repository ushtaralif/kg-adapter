import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class KGAdapter(nn.Module):
    def __init__(self, local_model_path, kg_encoder, kg_projector,
                 kg_scale_factor=0.1, max_length=512, device="cuda"):
        super().__init__()
        self.kg_encoder   = kg_encoder
        self.kg_projector = kg_projector
        self.kg_scale     = kg_scale_factor
        self.max_length   = max_length
        self._kg_bias     = None

        print(f"[KGAdapter] Loading LLM from {local_model_path}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            # local_files_only = True,
            local_files_only=False,
            dtype            = torch.bfloat16,
            device_map       = device,
        )
        for param in self.llm.parameters():
            param.requires_grad = False
        print(f"[KGAdapter] LLM frozen. d_model={self.llm.config.hidden_size}")

        self.token_emb = self.llm.get_input_embeddings()
        self.d_model   = self.llm.config.hidden_size

        self.kg_to_hidden = nn.Linear(
            kg_projector.num_kg_tokens * self.d_model,
            self.d_model
        )
        nn.init.xavier_uniform_(self.kg_to_hidden.weight)
        nn.init.zeros_(self.kg_to_hidden.bias)

        self._hook_handle = self._register_hook()

    def _register_hook(self):

        first_layer = self.llm.model.layers[0]

        def hook_fn(module, input, output):
            if self._kg_bias is None:
                return output

            hidden = output[0] if isinstance(output, tuple) else output
            bias = self._kg_bias.to(hidden.dtype)
            hidden = hidden + bias
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        return first_layer.register_forward_hook(hook_fn)

    def forward(self, input_ids, attention_mask, subj_ids, pred_ids,
                obj_ids, triple_mask, has_kg, labels=None):
        B = input_ids.size(0)

        kg_emb    = self.kg_encoder(subj_ids, pred_ids, obj_ids, triple_mask)
        kg_tokens = self.kg_projector(kg_emb) * self.kg_scale
        kg_tokens = kg_tokens * has_kg.float().view(B, 1, 1)

        kg_flat = kg_tokens.reshape(B, -1)
        kg_bias = self.kg_to_hidden(kg_flat)
        kg_bias = kg_bias.unsqueeze(1)

        self._kg_bias = kg_bias

        if labels is not None:

            L_input = input_ids.size(1)
            L_label = labels.size(1)
            if L_label < L_input:
                pad    = torch.full((B, L_input - L_label), -100,
                                    dtype=labels.dtype, device=labels.device)
                labels = torch.cat([labels, pad], dim=1)
            elif L_label > L_input:
                labels = labels[:, :L_input]

        outputs = self.llm(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            labels         = labels,
            return_dict    = True,
        )

        self._kg_bias = None
        return outputs

    def trainable_parameters(self):
        return (list(self.kg_encoder.parameters()) +
                list(self.kg_projector.parameters()) +
                list(self.kg_to_hidden.parameters()))

    def trainable_parameters_named(self):
        for name, p in self.kg_encoder.named_parameters():
            yield f"encoder.{name}", p
        for name, p in self.kg_projector.named_parameters():
            yield f"projector.{name}", p
        for name, p in self.kg_to_hidden.named_parameters():
            yield f"kg_to_hidden.{name}", p

    def save_adapter(self, path: str):
        import os; os.makedirs(path, exist_ok=True)
        torch.save({
            "kg_encoder":   self.kg_encoder.state_dict(),
            "kg_projector": self.kg_projector.state_dict(),
            "kg_to_hidden": self.kg_to_hidden.state_dict(),
        }, f"{path}/adapter_weights.pt")
        print(f"[KGAdapter] Saved to {path}/adapter_weights.pt")

    # def load_adapter(self, path: str):
    #     ckpt = torch.load(f"{path}/adapter_weights.pt", map_location="cpu")
    #     self.kg_encoder.load_state_dict(ckpt["kg_encoder"])
    #     self.kg_projector.load_state_dict(ckpt["kg_projector"])
    #     if "kg_to_hidden" in ckpt:
    #         self.kg_to_hidden.load_state_dict(ckpt["kg_to_hidden"])
    #     print(f"[KGAdapter] Loaded from {path}")

    def load_adapter(self, path: str, name: str = "adapter_weights"):
        ckpt = torch.load(f"{path}/{name}.pt", map_location="cpu")
        self.kg_encoder.load_state_dict(ckpt["kg_encoder"])
        self.kg_projector.load_state_dict(ckpt["kg_projector"])
        if "kg_to_hidden" in ckpt:
            self.kg_to_hidden.load_state_dict(ckpt["kg_to_hidden"])
        print(f"[KGAdapter] Loaded from {path}/{name}.pt")