import torch
import torch.nn as nn

class DustbinScorePredictor(nn.Module):
    def __init__(self, hidden_dim=256, nhead=4, num_layers=2):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feats1, feats2, mask1=None, mask2=None):
        feats1 = feats1.permute(0, 2, 1)
        feats2 = feats2.permute(0, 2, 1)
        
        B, N1, _ = feats1.shape
        B, N2, _ = feats2.shape
        
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, hidden_dim)

        input_seq = torch.cat([cls_tokens, feats1, cls_tokens, feats2], dim=1)
        
        src_key_padding_mask = None
        if mask1 is not None or mask2 is not None:
            if mask1 is None: mask1 = torch.zeros((B, N1), dtype=torch.bool, device=feats1.device)
            if mask2 is None: mask2 = torch.zeros((B, N2), dtype=torch.bool, device=feats2.device)
            
            cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=feats1.device)
            
            src_key_padding_mask = torch.cat([cls_mask, mask1, cls_mask, mask2], dim=1)

        out = self.transformer(input_seq, src_key_padding_mask=src_key_padding_mask)
        
        cls_out_1 = out[:, 0, :] 
        
        cls_out_2 = out[:, 1 + N1, :]
        
        combined_features = torch.cat([cls_out_1, cls_out_2], dim=1)
        
        score = self.prediction_head(combined_features) # (B, 1)
        
        return score.squeeze() # (B,)