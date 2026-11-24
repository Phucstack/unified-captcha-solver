from __future__ import annotations

import torch
import torch.nn as nn


class CaptchaCRNN(nn.Module):
    """Upgraded CRNN (CNN + BiGRU + Linear) for variable-length CAPTCHA OCR via CTC.

    Replaces legacy TensorFlow 1.x graphs from cnn_captcha & captcha_trainer
    with a compact, ONNX-exportable PyTorch architecture.
    """

    def __init__(self, num_classes: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
            dropout=0.1,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)
        feat = feat.mean(dim=2).permute(2, 0, 1)  # (T=W', B, C)
        seq, _ = self.rnn(feat)
        return self.classifier(seq)


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int | tuple[int, int] = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class CaptchaResNetCTC(nn.Module):
    """Lightweight ResNet + BiGRU + CTC inspired by captcha_trainer/network/ResNet.py."""

    def __init__(self, num_classes: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            _ResBlock(32, 64, stride=(2, 2)),
            _ResBlock(64, 128, stride=(2, 2)),
            _ResBlock(128, 128, stride=(2, 1)),
        )
        self.rnn = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
            dropout=0.1,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        feat = feat.mean(dim=2).permute(2, 0, 1)
        seq, _ = self.rnn(feat)
        return self.classifier(seq)


class _DenseLayer(nn.Module):
    def __init__(self, in_ch: int, growth_rate: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, growth_rate * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(growth_rate * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate * 2, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.net(x)], dim=1)


class _DenseBlock(nn.Module):
    def __init__(self, in_ch: int, num_layers: int, growth_rate: int = 16) -> None:
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(num_layers):
            layers.append(_DenseLayer(ch, growth_rate=growth_rate))
            ch += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_ch = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CaptchaDenseNetCTC(nn.Module):
    """DenseNet + BiGRU + CTC inspired by captcha_trainer/network/DenseNet.py."""

    def __init__(self, num_classes: int, hidden_size: int = 128) -> None:
        super().__init__()
        stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        b1 = _DenseBlock(32, num_layers=3, growth_rate=16)  # -> 80
        t1 = nn.Sequential(
            nn.BatchNorm2d(b1.out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(b1.out_ch, 64, kernel_size=1, bias=False),
            nn.AvgPool2d(2, 2),
        )
        b2 = _DenseBlock(64, num_layers=3, growth_rate=16)  # -> 112
        t2 = nn.Sequential(
            nn.BatchNorm2d(b2.out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(b2.out_ch, 128, kernel_size=1, bias=False),
            nn.AvgPool2d((2, 1), (2, 1)),
        )
        self.features = nn.Sequential(stem, b1, t1, b2, t2)
        self.rnn = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
            dropout=0.1,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = feat.mean(dim=2).permute(2, 0, 1)
        seq, _ = self.rnn(feat)
        return self.classifier(seq)


class _InvertedResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int | tuple[int, int], expand: int = 2) -> None:
        super().__init__()
        mid = in_ch * expand
        self.use_res = stride == 1 and in_ch == out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, stride=stride, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return x + out if self.use_res else out


class CaptchaMobileNetCTC(nn.Module):
    """Lightweight MobileNetV2-style depthwise separable CNN + BiGRU + CTC."""

    def __init__(self, num_classes: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            _InvertedResidual(32, 64, stride=2, expand=2),
            _InvertedResidual(64, 64, stride=1, expand=2),
            _InvertedResidual(64, 128, stride=(2, 1), expand=2),
            _InvertedResidual(128, 128, stride=1, expand=2),
        )
        self.rnn = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
            dropout=0.1,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = feat.mean(dim=2).permute(2, 0, 1)
        seq, _ = self.rnn(feat)
        return self.classifier(seq)


def build_model(arch: str, num_classes: int) -> nn.Module:
    arch_norm = arch.lower().strip()
    if arch_norm == "resnet":
        return CaptchaResNetCTC(num_classes=num_classes)
    if arch_norm == "densenet":
        return CaptchaDenseNetCTC(num_classes=num_classes)
    if arch_norm == "mobilenet":
        return CaptchaMobileNetCTC(num_classes=num_classes)
    if arch_norm == "crnn":
        return CaptchaCRNN(num_classes=num_classes)
    raise ValueError(
        f"Unsupported architecture '{arch}'. Choose 'crnn', 'resnet', 'densenet', or 'mobilenet'."
    )
