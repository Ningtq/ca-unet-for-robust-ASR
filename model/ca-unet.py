import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels1,
        out_channels2,
        do_conv_adapter=False,
        downsample=False,
        upsample=False,
        pool_type="max",
    ):
        super().__init__()
        
        # Downsample setting
        self.downsample = downsample
        self.pool_type = pool_type
        if self.downsample:
            self.pool = (
                nn.MaxPool2d(kernel_size=2, stride=2)
                if pool_type == "max"
                else nn.AvgPool2d(kernel_size=2, stride=2)
            )
        
        # Upsample setting
        self.upsample = upsample
        if self.upsample:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels1, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels1)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels1, out_channels=out_channels2, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels2)
        
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        
        # Channel adapter for residual connection
        self.do_conv_adapter = do_conv_adapter
        if self.do_conv_adapter:
            self.conv_adapter = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels2, kernel_size=1
            )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [B, in_channels, N, L]
        Returns:
            out: [B, out_channels2, N, L]
        """
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.do_conv_adapter:
            identity = self.conv_adapter(x)
        
        out += identity
        out = self.activation(out)
        
        # Downsample
        if self.downsample:
            out = self.pool(out)
        
        # Upsample
        if self.upsample:
            out = self.up(out)
        
        return out


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, channels, dropout_rate, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads!"
        
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        # Projection layers
        self.query_proj = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.key_proj = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.value_proj = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.LayerNorm(channels)
        
        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(dropout_rate)
        
        self._init_weights()
    
    def _init_weights(self):
        for conv in [self.query_proj, self.key_proj, self.value_proj]:
            nn.init.xavier_uniform_(conv.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)
    
    def forward(self, query, key, value):
        """
        Args:
            query, key, value: [B, C, L, D]
        Returns:
            out: [B, C, L, D]
        """
        identity = query
        batch_size, channels, length, dim = query.shape
        
        # Reshape to multi-head format
        Q = self.query_proj(query).view(
            batch_size, self.num_heads, self.head_dim, length, dim
        )
        K = self.key_proj(key).view(
            batch_size, self.num_heads, self.head_dim, length, dim
        )
        V = self.value_proj(value).view(
            batch_size, self.num_heads, self.head_dim, length, dim
        )
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention_scores = self.softmax(attention_scores)
        attention_scores = self.attention_dropout(attention_scores)
        
        # Apply attention to values
        out = torch.matmul(attention_scores, V)
        out = out.reshape(batch_size, channels, length, dim)
        
        # Output projection and residual connection
        out = self.output_proj(out)
        out = out + identity
        
        # Layer normalization
        out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return out


class AttentionBlock(nn.Module):
    def __init__(self, channels, dropout_rate):
        super().__init__()
        
        # Projection layers
        self.query_proj = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.key_proj = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.value_proj = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.LayerNorm(channels)
        
        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(dropout_rate)
        
        self._init_weights()
    
    def _init_weights(self):
        for conv in [self.query_proj, self.key_proj, self.value_proj]:
            nn.init.xavier_uniform_(conv.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)
    
    def forward(self, query, key, value):
        """
        Args:
            query, key, value: [B, C, L, D]
        Returns:
            out: [B, C, L, D]
        """
        identity = query
        batch_size, channels, length, dim = query.shape
        
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(dim)
        attention_scores = self.softmax(attention_scores)
        attention_scores = self.attention_dropout(attention_scores)
        
        # Apply attention to values
        out = torch.matmul(attention_scores, V)
        out = self.output_proj(out)
        out = out + identity
        
        return out


class GatedFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.Sigmoid(),  # Gate values in [0, 1]
        )
    
    def forward(self, main_feat, secondary_feat):
        gate_values = self.gate(main_feat)
        fused_feat = main_feat + gate_values * secondary_feat
        return fused_feat


class FFModule(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder branch 1 (enhanced features)
        self.enc1_res1 = ConvResidualBlock(
            in_channels=1, out_channels1=8, out_channels2=16,
            do_conv_adapter=True, downsample=True
        )
        self.enc1_att1 = MultiHeadAttentionBlock(channels=16, dropout_rate=0.1)
        
        self.enc1_res2 = ConvResidualBlock(
            in_channels=16, out_channels1=24, out_channels2=32,
            do_conv_adapter=True, downsample=True
        )
        self.enc1_att2 = MultiHeadAttentionBlock(channels=32, dropout_rate=0.1)
        
        self.enc1_res3 = ConvResidualBlock(
            in_channels=32, out_channels1=48, out_channels2=64,
            do_conv_adapter=True, downsample=True
        )
        self.enc1_att3 = MultiHeadAttentionBlock(channels=64, dropout_rate=0.1)
        
        self.enc1_res4 = ConvResidualBlock(
            in_channels=64, out_channels1=96, out_channels2=128,
            do_conv_adapter=True, downsample=True
        )
        self.enc1_att4 = MultiHeadAttentionBlock(channels=128, dropout_rate=0.1)
        
        # Encoder branch 2 (noisy features)
        self.enc2_res1 = ConvResidualBlock(
            in_channels=1, out_channels1=8, out_channels2=16,
            do_conv_adapter=True, downsample=True
        )
        self.enc2_att1 = MultiHeadAttentionBlock(channels=16, dropout_rate=0.1)
        
        self.enc2_res2 = ConvResidualBlock(
            in_channels=16, out_channels1=24, out_channels2=32,
            do_conv_adapter=True, downsample=True
        )
        self.enc2_att2 = MultiHeadAttentionBlock(channels=32, dropout_rate=0.1)
        
        self.enc2_res3 = ConvResidualBlock(
            in_channels=32, out_channels1=48, out_channels2=64,
            do_conv_adapter=True, downsample=True
        )
        self.enc2_att3 = MultiHeadAttentionBlock(channels=64, dropout_rate=0.1)
        
        self.enc2_res4 = ConvResidualBlock(
            in_channels=64, out_channels1=96, out_channels2=128,
            do_conv_adapter=True, downsample=True
        )
        self.enc2_att4 = MultiHeadAttentionBlock(channels=128, dropout_rate=0.1)
        
        # Decoder branch 1
        self.dec1_res1 = ConvResidualBlock(
            in_channels=128, out_channels1=96, out_channels2=64,
            do_conv_adapter=True, upsample=True
        )
        self.dec1_res2 = ConvResidualBlock(
            in_channels=64, out_channels1=48, out_channels2=32,
            do_conv_adapter=True, upsample=True
        )
        self.dec1_res3 = ConvResidualBlock(
            in_channels=32, out_channels1=24, out_channels2=16,
            do_conv_adapter=True, upsample=True
        )
        self.dec1_res4 = ConvResidualBlock(
            in_channels=16, out_channels1=8, out_channels2=1,
            do_conv_adapter=True, upsample=True
        )
        
        # Decoder branch 2
        self.dec2_res1 = ConvResidualBlock(
            in_channels=128, out_channels1=96, out_channels2=64,
            do_conv_adapter=True, upsample=True
        )
        self.dec2_res2 = ConvResidualBlock(
            in_channels=64, out_channels1=48, out_channels2=32,
            do_conv_adapter=True, upsample=True
        )
        self.dec2_res3 = ConvResidualBlock(
            in_channels=32, out_channels1=24, out_channels2=16,
            do_conv_adapter=True, upsample=True
        )
        self.dec2_res4 = ConvResidualBlock(
            in_channels=16, out_channels1=8, out_channels2=1,
            do_conv_adapter=True, upsample=True
        )
        
        # Adaptive convolution layers
        self.adp_conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.adp_conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.adp_conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
        self.adp_conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
        self.adp_conv5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1)
        self.adp_conv6 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1)
        
        # Cross-attention layers in decoder
        self.dec_att1 = MultiHeadAttentionBlock(channels=64, dropout_rate=0.1)
        self.dec_att2 = MultiHeadAttentionBlock(channels=64, dropout_rate=0.1)
        self.dec_att3 = MultiHeadAttentionBlock(channels=32, dropout_rate=0.1)
        self.dec_att4 = MultiHeadAttentionBlock(channels=32, dropout_rate=0.1)
        self.dec_att5 = MultiHeadAttentionBlock(channels=16, dropout_rate=0.1)
        self.dec_att6 = MultiHeadAttentionBlock(channels=16, dropout_rate=0.1)
        
        # Self-attention layers
        self.self_att1 = AttentionBlock(channels=16, dropout_rate=0.1)
        self.self_att2 = AttentionBlock(channels=16, dropout_rate=0.1)
        self.self_att3 = AttentionBlock(channels=32, dropout_rate=0.1)
        self.self_att4 = AttentionBlock(channels=32, dropout_rate=0.1)
        self.self_att5 = AttentionBlock(channels=64, dropout_rate=0.1)
        self.self_att6 = AttentionBlock(channels=64, dropout_rate=0.1)
        self.self_att7 = AttentionBlock(channels=1, dropout_rate=0.1)
        self.self_att8 = AttentionBlock(channels=1, dropout_rate=0.1)
        
        # Fusion module
        self.gated_fusion = GatedFusion(in_channels=1, out_channels=1)
    
    @staticmethod
    def pad_len(feat_ref, feat_to_adjust, mode="constant", value=0.0):
        """
        Adjust feat_to_adjust's last dimension to match feat_ref's last dimension.
        
        Args:
            feat_ref: Reference tensor, shape [B, C, H, W1]
            feat_to_adjust: Tensor to adjust, shape [B, C, H, W2]
            mode: Padding mode ('constant', 'reflect', 'replicate', 'circular')
            value: Fill value for 'constant' padding
        
        Returns:
            Adjusted tensor with last dimension matching feat_ref
        """
        target_len = feat_ref.size(-1)
        current_len = feat_to_adjust.size(-1)
        
        if current_len < target_len:
            # Pad feat_to_adjust (right padding by default)
            diff = target_len - current_len
            padding = (0, diff)  # (left, right) for last dimension
            return F.pad(feat_to_adjust, padding, mode=mode, value=value)
        elif current_len > target_len:
            # Center-crop feat_to_adjust
            start = (current_len - target_len) // 2
            return feat_to_adjust[..., start: start + target_len]
        else:
            # No adjustment needed
            return feat_to_adjust
    
    def forward(self, enhance_feat, noisy_feat):
        """
        Args:
            enhance_feat: Enhanced features [B, N, L]
            noisy_feat: Noisy features [B, N, L]
        Returns:
            out: Fused features [B, N, L]
        """
        # Add channel dimension
        enhance_feat = enhance_feat.unsqueeze(dim=1)  # [B, 1, N, L]
        noisy_feat = noisy_feat.unsqueeze(dim=1)
        
        # Encoder branch 1 (enhanced features)
        enc1_out1 = self.enc1_res1(enhance_feat)  # [B, 16, N, L]
        enc1_out2 = self.enc1_res2(enc1_out1)      # [B, 32, N, L]
        enc1_out3 = self.enc1_res3(enc1_out2)      # [B, 64, N, L]
        enc1_out4 = self.enc1_res4(enc1_out3)      # [B, 128, N, L]
        
        # Encoder branch 2 (noisy features)
        enc2_out1 = self.enc2_res1(noisy_feat)     # [B, 16, N, L]
        enc2_out2 = self.enc2_res2(enc2_out1)      # [B, 32, N, L]
        enc2_out3 = self.enc2_res3(enc2_out2)      # [B, 64, N, L]
        enc2_out4 = self.enc2_res4(enc2_out3)      # [B, 128, N, L]
        
        # Cross-attention between branches at each level
        ca1_out = self.enc1_att1(query=enc1_out1, key=enc2_out1, value=enc2_out1)
        ca2_out = self.enc2_att1(query=enc2_out1, key=enc1_out1, value=enc1_out1)
        
        ca3_out = self.enc1_att2(query=enc1_out2, key=enc2_out2, value=enc2_out2)
        ca4_out = self.enc2_att2(query=enc2_out2, key=enc1_out2, value=enc1_out2)
        
        ca5_out = self.enc1_att3(query=enc1_out3, key=enc2_out3, value=enc2_out3)
        ca6_out = self.enc2_att3(query=enc2_out3, key=enc1_out3, value=enc1_out3)
        
        ca7_out = self.enc1_att4(query=enc1_out4, key=enc2_out4, value=enc2_out4)
        ca8_out = self.enc2_att4(query=enc2_out4, key=enc1_out4, value=enc1_out4)
        
        # Self-attention within each branch
        sa1_out = self.self_att1(query=ca1_out, key=ca1_out, value=ca1_out)
        sa2_out = self.self_att2(query=ca2_out, key=ca2_out, value=ca2_out)
        
        sa3_out = self.self_att3(query=ca3_out, key=ca3_out, value=ca3_out)
        sa4_out = self.self_att4(query=ca4_out, key=ca4_out, value=ca4_out)
        
        sa5_out = self.self_att5(query=ca5_out, key=ca5_out, value=ca5_out)
        sa6_out = self.self_att6(query=ca6_out, key=ca6_out, value=ca6_out)
        
        # Decoder branch 1
        dec1_out1 = self.dec1_res1(ca7_out)
        dec1_out1 = self.pad_len(sa5_out, dec1_out1)
        concat1 = torch.cat([sa5_out, dec1_out1], dim=1)
        adp1_out = self.adp_conv1(concat1)
        dec_att1_out = self.dec_att1(query=adp1_out, key=adp1_out, value=adp1_out)
        
        dec1_out2 = self.dec1_res2(dec_att1_out)
        dec1_out2 = self.pad_len(sa3_out, dec1_out2)
        concat2 = torch.cat([sa3_out, dec1_out2], dim=1)
        adp2_out = self.adp_conv3(concat2)
        dec_att2_out = self.dec_att3(query=adp2_out, key=adp2_out, value=adp2_out)
        
        dec1_out3 = self.dec1_res3(dec_att2_out)
        dec1_out3 = self.pad_len(sa1_out, dec1_out3)
        concat3 = torch.cat([sa1_out, dec1_out3], dim=1)
        adp3_out = self.adp_conv5(concat3)
        dec_att3_out = self.dec_att5(query=adp3_out, key=adp3_out, value=adp3_out)
        
        dec1_out4 = self.dec1_res4(dec_att3_out)
        dec1_out4 = self.pad_len(enhance_feat, dec1_out4)
        
        # Decoder branch 2
        dec2_out1 = self.dec2_res1(ca8_out)
        dec2_out1 = self.pad_len(sa6_out, dec2_out1)
        concat4 = torch.cat([sa6_out, dec2_out1], dim=1)
        adp4_out = self.adp_conv2(concat4)
        dec_att4_out = self.dec_att2(query=adp4_out, key=adp4_out, value=adp4_out)
        
        dec2_out2 = self.dec2_res2(dec_att4_out)
        dec2_out2 = self.pad_len(sa4_out, dec2_out2)
        concat5 = torch.cat([sa4_out, dec2_out2], dim=1)
        adp5_out = self.adp_conv4(concat5)
        dec_att5_out = self.dec_att4(query=adp5_out, key=adp5_out, value=adp5_out)
        
        dec2_out3 = self.dec2_res3(dec_att5_out)
        dec2_out3 = self.pad_len(sa2_out, dec2_out3)
        concat6 = torch.cat([sa2_out, dec2_out3], dim=1)
        adp6_out = self.adp_conv6(concat6)
        dec_att6_out = self.dec_att6(query=adp6_out, key=adp6_out, value=adp6_out)
        
        dec2_out4 = self.dec2_res4(dec_att6_out)
        dec2_out4 = self.pad_len(noisy_feat, dec2_out4)
        
        # Final self-attention and fusion
        sa7_out = self.self_att7(
            query=(dec1_out4 + enhance_feat),
            key=(dec1_out4 + enhance_feat),
            value=(dec1_out4 + enhance_feat)
        )
        sa8_out = self.self_att8(
            query=(dec2_out4 + noisy_feat),
            key=(dec2_out4 + noisy_feat),
            value=(dec2_out4 + noisy_feat)
        )
        
        out = self.gated_fusion(sa7_out, sa8_out).squeeze(dim=1)  # [B, N, L]
        
        return out