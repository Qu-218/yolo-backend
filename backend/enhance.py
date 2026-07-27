import cv2
import numpy as np
import torch
import time
import logging

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"低光照增强模块 当前使用设备: {device}")


class SCIEnhance:
    def __init__(self):
        self.device = device

    def enhance(self, img_bgr, intensity=0.7):
        """
        多阶段低光照图像增强管线
        :param img_bgr: OpenCV BGR 图像
        :param intensity: 增强强度 0.0~1.0
        :return: enhanced_bgr, enhance_cost(秒)
        """
        start = time.time()
        intensity = max(0.0, min(1.0, float(intensity)))

        if intensity < 0.01:
            return img_bgr, round(time.time() - start, 3)

        # 阶段1: CLAHE 自适应直方图均衡化，提升局部对比度
        enhanced = self._clahe(img_bgr)

        # 阶段2: Gamma 校正，压制过亮区域
        enhanced = self._gamma_correct(enhanced, gamma=0.75)

        # 阶段3: LAB 色彩空间 L 通道增强，保色增亮
        enhanced = self._lab_enhance(enhanced)

        # 阶段4: 双边滤波降噪，保留边缘
        enhanced = self._denoise(enhanced)

        # 阶段5: 锐化，恢复细节
        enhanced = self._sharpen(enhanced)

        # 按 intensity 混合原图与增强图
        result = cv2.addWeighted(img_bgr, 1.0 - intensity, enhanced, intensity, 0)

        enhance_cost = round(time.time() - start, 3)
        return result, enhance_cost

    def _clahe(self, img):
        """自适应直方图均衡化"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _gamma_correct(self, img, gamma=0.75):
        """Gamma 校正，gamma<1 提亮暗部"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)

    def _lab_enhance(self, img):
        """LAB 空间 L 通道增强"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # 对 L 通道做 CLAHE 已在阶段1完成，此处额外做对比度拉伸
        l_float = l.astype(np.float32)
        lo, hi = np.percentile(l_float, 1), np.percentile(l_float, 99)
        if hi - lo > 1:
            l_float = (l_float - lo) / (hi - lo) * 255.0
        l = np.clip(l_float, 0, 255).astype(np.uint8)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _denoise(self, img):
        """双边滤波降噪"""
        return cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)

    def _sharpen(self, img):
        """USM 锐化"""
        blur = cv2.GaussianBlur(img, (0, 0), 3)
        return cv2.addWeighted(img, 1.5, blur, -0.5, 0)


# 全局单例
enhancer = SCIEnhance()


def low_light_enhance_image(img_bgr, intensity=0.7):
    """对外接口，供 main.py 调用"""
    return enhancer.enhance(img_bgr, intensity)
