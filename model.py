"""model.py — архитектура модели сегментации облаков."""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn


def build_smp_unet(encoder: str = "resnet18", pretrained: bool = True,
                   in_channels: int = 3, classes: int = 1) -> nn.Module:
    import segmentation_models_pytorch as smp
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=classes,
    )
    return model


class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) * 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = nn.functional.pad(x, (dx // 2, dx - dx // 2, dy // 2, dy - dy // 2))
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class SmallUNet(nn.Module):
    """Компактная U-Net: 4 уровня вниз/вверх, base каналов 32 (~7.7M параметров)."""

    def __init__(self, in_channels: int = 3, base: int = 32, classes: int = 1):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.d3 = Down(base * 4, base * 8)
        self.d4 = Down(base * 8, base * 16)
        self.u1 = Up(base * 16, base * 8)
        self.u2 = Up(base * 8, base * 4)
        self.u3 = Up(base * 4, base * 2)
        self.u4 = Up(base * 2, base)
        self.out = nn.Conv2d(base, classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.out(x)


def build_model(arch: str = "unet", encoder: str = "resnet18", pretrained: bool = True,
                in_channels: int = 3, classes: int = 1) -> tuple[nn.Module, str]:
    """Возвращает (model, arch_фактически_использованная)."""
    if arch == "unet":
        try:
            return build_smp_unet(encoder, pretrained, in_channels, classes), "unet"
        except Exception as e:
            print(f"[model] smp недоступен ({type(e).__name__}: {e}); "
                  f"использую собственную SmallUNet.")
            return SmallUNet(in_channels, base=32, classes=classes), "small_unet"
    if arch == "small_unet":
        return SmallUNet(in_channels, base=32, classes=classes), "small_unet"
    raise ValueError(f"Неизвестная архитектура: {arch}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Самопроверка архитектур модели")
    parser.add_argument("--arch", default="unet", choices=["unet", "small_unet"])
    parser.add_argument("--encoder", default="resnet18")
    parser.add_argument("--size", type=int, default=384)
    args = parser.parse_args()

    torch.manual_seed(0)
    model, used = build_model(args.arch, args.encoder, pretrained=False)
    x = torch.randn(2, 3, args.size, args.size)
    with torch.no_grad():
        y = model(x)
    print(f"Архитектура: {used} | параметров: {count_params(model):,}")
    print(f"Вход: {tuple(x.shape)} -> Выход: {tuple(y.shape)}")
    assert y.shape == (2, 1, args.size, args.size), "Неверная форма выхода!"
    print("Самопроверка пройдена: форма выхода совпадает с маской.")
