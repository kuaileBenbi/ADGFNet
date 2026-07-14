import torch
import torch.nn as nn
import torch.nn.functional as F


def resize_to(feature, reference):
    if feature.shape[-2:] == reference.shape[-2:]:
        return feature
    return F.interpolate(
        feature, size=reference.shape[-2:], mode="bilinear", align_corners=False
    )


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        activation=True,
    ):
        if padding is None:
            padding = kernel_size // 2
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = ConvBNAct(
            out_channels, out_channels, kernel_size=3, activation=False
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBNAct(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                activation=False,
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.relu(x + residual)


class ResidualStage(nn.Sequential):
    def __init__(self, in_channels, out_channels, num_blocks=1, stride=1):
        layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels))
        super().__init__(*layers)


class LightweightResidualEncoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        channels=(16, 32, 64, 128),
        blocks_per_stage=(1, 1, 1, 1),
    ):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, channels[0], kernel_size=3),
            ResidualBlock(channels[0], channels[0]),
        )
        self.stage1 = ResidualStage(
            channels[0], channels[0], num_blocks=blocks_per_stage[0], stride=1
        )
        self.stage2 = ResidualStage(
            channels[0], channels[1], num_blocks=blocks_per_stage[1], stride=2
        )
        self.stage3 = ResidualStage(
            channels[1], channels[2], num_blocks=blocks_per_stage[2], stride=2
        )
        self.stage4 = ResidualStage(
            channels[2], channels[3], num_blocks=blocks_per_stage[3], stride=2
        )

    def forward(self, x):
        c1 = self.stage1(self.stem(x))
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        return c1, c2, c3, c4


class LocalContrastBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre = ConvBNAct(channels, channels, kernel_size=1, padding=0)
        self.dw3 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.dw5 = nn.Conv2d(
            channels, channels, kernel_size=5, padding=2, groups=channels, bias=False
        )
        self.pw = ConvBNAct(channels * 2, channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.pre(x)
        local3 = self.dw3(x)
        local5 = self.dw5(x)
        return self.pw(torch.cat([local3 - x, local5 - x], dim=1))


class GradientPriorBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("weight_x", kernel_x.repeat(channels, 1, 1, 1))
        self.register_buffer("weight_y", kernel_y.repeat(channels, 1, 1, 1))

    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x, padding=1, groups=self.channels).float()
        grad_y = F.conv2d(x, self.weight_y, padding=1, groups=self.channels).float()
        return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6).to(x.dtype)


class LocalAnomalyBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local_branch = LocalContrastBranch(channels)
        self.grad_branch = nn.Sequential(
            GradientPriorBranch(channels),
            ConvBNAct(channels, channels, kernel_size=1, padding=0),
        )
        self.fuse = nn.Sequential(
            ConvBNAct(channels * 2, channels, kernel_size=1, padding=0),
            ResidualBlock(channels, channels),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.anomaly_proj = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        anomaly = self.fuse(
            torch.cat([self.local_branch(x), self.grad_branch(x)], dim=1)
        )
        out = x + anomaly * self.scale(anomaly)
        anomaly_map = torch.sigmoid(self.anomaly_proj(anomaly))
        return out, anomaly_map


class RefinementBlock(nn.Sequential):
    def __init__(self, channels):
        super().__init__(
            ResidualBlock(channels, channels), ResidualBlock(channels, channels)
        )


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, num_classes=1):
        super().__init__(
            ConvBNAct(in_channels, in_channels, kernel_size=3),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )


class AnomalyGatedFusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.high_proj = ConvBNAct(channels, channels, kernel_size=1, padding=0)
        self.low_proj = ConvBNAct(channels, channels, kernel_size=1, padding=0)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out = ConvBNAct(channels * 2, channels, kernel_size=3)

    def forward(self, high_feature, low_feature, anomaly_map=None):
        high_feature = self.high_proj(resize_to(high_feature, low_feature))
        low_feature = self.low_proj(low_feature)
        if anomaly_map is not None:
            spatial_attn = resize_to(anomaly_map, low_feature)
            channel_attn = self.channel_gate(
                torch.cat([low_feature, high_feature], dim=1)
            )
            gate = spatial_attn * channel_attn
        else:
            gate = low_feature.new_full(
                (1, low_feature.shape[1], 1, 1), 0.5
            )
        fused = torch.cat(
            [low_feature * gate, high_feature * (1.0 - gate)], dim=1
        )
        return self.out(fused)


class AG_FPN(nn.Module):
    def __init__(self, encoder_channels, neck_channels):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        self.laterals = nn.ModuleList(
            [
                ConvBNAct(c1, neck_channels, kernel_size=1, padding=0),
                ConvBNAct(c2, neck_channels, kernel_size=1, padding=0),
                ConvBNAct(c3, neck_channels, kernel_size=1, padding=0),
                ConvBNAct(c4, neck_channels, kernel_size=1, padding=0),
            ]
        )
        self.fuse3 = AnomalyGatedFusionBlock(neck_channels)
        self.fuse2 = AnomalyGatedFusionBlock(neck_channels)
        self.fuse1 = AnomalyGatedFusionBlock(neck_channels)
        self.aggregate = ConvBNAct(
            neck_channels * 4, neck_channels, kernel_size=1, padding=0
        )

    def forward(self, features, anomaly_maps=None):
        c1, c2, c3, c4 = features
        p1 = self.laterals[0](c1)
        p2 = self.laterals[1](c2)
        p3 = self.laterals[2](c3)
        p4 = self.laterals[3](c4)

        a1 = a2 = a3 = None
        if anomaly_maps is not None:
            a1, a2, a3 = anomaly_maps

        p3 = self.fuse3(p4, p3, anomaly_map=a3)
        p2 = self.fuse2(p3, p2, anomaly_map=a2)
        p1 = self.fuse1(p2, p1, anomaly_map=a1)

        merged = torch.cat(
            [p1, resize_to(p2, p1), resize_to(p3, p1), resize_to(p4, p1)], dim=1
        )
        return self.aggregate(merged)


class ADGFNet(nn.Module):
    outputs_logits = True

    def __init__(
        self,
        in_channels=1,
        num_classes=1,
        encoder_channels=(16, 32, 64, 128),
        neck_channels=32,
        blocks_per_stage=(1, 1, 1, 1),
        use_refine=True,
    ):
        super().__init__()
        self.use_refine = use_refine
        self.encoder = LightweightResidualEncoder(
            in_channels=in_channels,
            channels=encoder_channels,
            blocks_per_stage=blocks_per_stage,
        )
        self.lab1 = LocalAnomalyBlock(encoder_channels[0])
        self.lab2 = LocalAnomalyBlock(encoder_channels[1])
        self.lab3 = LocalAnomalyBlock(encoder_channels[2])
        self.neck = AG_FPN(encoder_channels, neck_channels)
        self.refine = (
            RefinementBlock(neck_channels) if use_refine else nn.Identity()
        )
        self.head = SegmentationHead(neck_channels, num_classes)

    def forward(self, x):
        c1, c2, c3, c4 = self.encoder(x)
        f1, a1 = self.lab1(c1)
        f2, a2 = self.lab2(c2)
        f3, a3 = self.lab3(c3)
        fused = self.neck((f1, f2, f3, c4), anomaly_maps=(a1, a2, a3))
        refined = self.refine(fused)
        logits = self.head(refined)
        logits = resize_to(logits, x)
        return logits

    def evaluate(self, x):
        return torch.sigmoid(self.forward(x))


class ADGFNetLite(ADGFNet):
    def __init__(self, **kwargs):
        super().__init__(
            encoder_channels=(8, 16, 32, 64),
            neck_channels=8,
            use_refine=False,
            **kwargs,
        )


if __name__ == '__main__':
    # 1. Test basic model
    model = ADGFNet()
    input = torch.randn(1, 1, 256, 256)
    output = model(input)
    print(output.shape)
    


