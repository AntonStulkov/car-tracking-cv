"""Деблюр кадров (DeblurGAN, PyTorch).

DeblurGAN не ставится через pip, поэтому ожидается экспортированный в TorchScript
генератор (torch.jit.save). Без весов используется классическая резкость (unsharp mask).
"""
import cv2
import numpy as np
import torch


class Deblurrer:
    def __init__(self, weights: str | None = None, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        if weights:
            self.model = torch.jit.load(weights, map_location=self.device).eval()

    @torch.inference_mode()
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if self.model is None:
            return self._unsharp(frame)

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Сеть ожидает размеры, кратные 32
        ph, pw = (-h) % 32, (-w) % 32
        rgb = cv2.copyMakeBorder(rgb, 0, ph, 0, pw, cv2.BORDER_REFLECT)

        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(127.5).sub(1.0)  # [-1, 1]
        y = self.model(x.unsqueeze(0).to(self.device))[0]
        out = ((y.clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        out = out[:h, :w]
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _unsharp(frame: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(frame, (0, 0), 3)
        return cv2.addWeighted(frame, 1.5, blur, -0.5, 0)
