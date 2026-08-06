"""
model.py — the reconstructor architecture (image -> Mermaid code).

See IDEA.md for the full design rationale, size analysis, and the
frozen/trainable layer breakdown this file implements. Short version:

  Encoder: BEiT-Base, taken from TrOCR's pretrained checkpoint (Microsoft) --
           TrOCR's own decoder is discarded (it outputs plain English, not
           Mermaid), but its encoder is kept because it was pretrained
           specifically to read text out of images (hundreds of millions
           of synthetic printed text lines), which is far more relevant to
           reading Mermaid node/edge labels than generic ImageNet
           classification pretraining. ~86M params; bottom 4 of 12 blocks
           frozen, top 8 fine-tuned (see freeze_encoder_layers below).
  Decoder: a small transformer decoder (6 layers, d_model=384), trained
           from scratch with a custom byte-level BPE tokenizer over
           Mermaid syntax (see train_tokenizer.py) instead of reusing a
           full 50k-token multilingual vocabulary. ~19M params.

Total: ~105M params (~400MB fp32 / ~200MB fp16) -- larger than the
original ViT-Tiny version, a deliberate trade made because the OCR-pretrained
encoder is expected to read Mermaid labels more reliably, and accuracy is
now the stated priority over size.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import VisionEncoderDecoderModel


class MermaidReconstructor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        trocr_checkpoint: str = "microsoft/trocr-base-stage1",
        d_model: int = 384,
        decoder_layers: int = 6,
        nhead: int = 6,
        dim_feedforward: int = 1536,
        max_len: int = 640,
        freeze_bottom_n_blocks: int = 4,
        pretrained: bool = True,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len

        # --- Encoder: TrOCR's pretrained BEiT-Base, decoder half discarded ---
        if pretrained:
            trocr = VisionEncoderDecoderModel.from_pretrained(trocr_checkpoint)
            self.encoder = trocr.encoder
            del trocr.decoder  # we only wanted the encoder half
        else:
            # structure-only (random weights) -- useful for quick shape/param
            # sanity checks without downloading the real checkpoint
            from transformers import BeitConfig, BeitModel
            self.encoder = BeitModel(BeitConfig())

        encoder_dim = self.encoder.config.hidden_size
        self.enc_proj = nn.Identity() if encoder_dim == d_model else nn.Linear(encoder_dim, d_model)

        if freeze_bottom_n_blocks > 0:
            self.freeze_encoder_layers(freeze_bottom_n_blocks)

        # --- Decoder: small transformer, trained from scratch ---
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(max_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def freeze_encoder_layers(self, n: int) -> None:
        """Freezes patch embedding, position embeddings, and the bottom `n`
        transformer blocks of the encoder (see IDEA.md for why the bottom
        layers specifically). The remaining top blocks + final layernorm
        stay trainable, just at a lower learning rate (set up in
        train_reconstructor.py's optimizer param groups, not here)."""
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        for block in self.encoder.layers[:n]:
            for param in block.parameters():
                param.requires_grad = False

    def encoder_param_groups(self):
        """Unfrozen-but-pretrained encoder params, for the differential
        learning rate set up in train_reconstructor.py."""
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def scratch_param_groups(self):
        """Everything trained from scratch: projection + decoder + head."""
        params = list(self.enc_proj.parameters())
        params += list(self.token_embed.parameters())
        params += list(self.pos_embed.parameters())
        params += list(self.decoder.parameters())
        params += list(self.output_proj.parameters())
        return params

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) -> memory: (B, num_patches[+cls], d_model)"""
        feats = self.encoder(images).last_hidden_state  # (B, N, encoder_dim)
        return self.enc_proj(feats)

    def forward(self, images: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        """Teacher-forcing forward pass. decoder_input_ids: (B, T) already
        shifted-right (starts with <s>). Returns logits (B, T, vocab_size)."""
        memory = self.encode(images)
        B, T = decoder_input_ids.shape
        positions = torch.arange(T, device=decoder_input_ids.device).unsqueeze(0).expand(B, T)
        tgt = self.token_embed(decoder_input_ids) + self.pos_embed(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt.device)
        padding_mask = decoder_input_ids == self.pad_id

        out = self.decoder(tgt, memory, tgt_mask=causal_mask, tgt_key_padding_mask=padding_mask)
        return self.output_proj(out)

    @torch.no_grad()
    def generate(self, images: torch.Tensor, bos_id: int, eos_id: int,
                 max_new_tokens: int = 500) -> torch.Tensor:
        """Greedy decoding. Returns token id sequences (B, <=max_new_tokens+1)."""
        device = images.device
        memory = self.encode(images)
        B = images.size(0)
        tokens = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            T = tokens.size(1)
            positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            tgt = self.token_embed(tokens) + self.pos_embed(positions)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
            out = self.decoder(tgt, memory, tgt_mask=causal_mask)
            next_logits = self.output_proj(out[:, -1, :])
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            next_token = torch.where(finished.unsqueeze(1), torch.full_like(next_token, eos_id), next_token)
            tokens = torch.cat([tokens, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == eos_id)
            if finished.all():
                break
        return tokens

    def num_params(self) -> dict:
        enc_frozen = sum(p.numel() for p in self.encoder.parameters() if not p.requires_grad)
        enc_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        dec = sum(p.numel() for p in self.decoder.parameters())
        proj = sum(p.numel() for p in self.enc_proj.parameters())
        emb = sum(p.numel() for p in self.token_embed.parameters()) + sum(p.numel() for p in self.pos_embed.parameters())
        head = sum(p.numel() for p in self.output_proj.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {
            "encoder_frozen": enc_frozen,
            "encoder_trainable": enc_trainable,
            "projection": proj,
            "decoder": dec,
            "embeddings_and_head": emb + head,
            "total": total,
            "total_trainable": enc_trainable + proj + dec + emb + head,
        }


if __name__ == "__main__":
    # Structural sanity check -- confirms the encoder/decoder wire together
    # and prints parameter counts before committing to downloading the real
    # ~330MB checkpoint or a long training run.
    #   python model.py            -> quick check, random weights (no download)
    #   python model.py --download -> also verifies the real TrOCR checkpoint loads
    import sys

    use_pretrained = "--download" in sys.argv
    model = MermaidReconstructor(vocab_size=6000, pretrained=use_pretrained)
    counts = model.num_params()
    for k, v in counts.items():
        print(f"{k:>22}: {v/1e6:7.2f}M params")
    fp32_mb = counts["total"] * 4 / (1024 ** 2)
    fp16_mb = counts["total"] * 2 / (1024 ** 2)
    print(f"\nEstimated size: {fp32_mb:.1f} MB (fp32)  /  {fp16_mb:.1f} MB (fp16)")
    print(f"Trainable share: {counts['total_trainable']/counts['total']*100:.1f}%")
