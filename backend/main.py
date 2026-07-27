"""
项目：暖记PLM
功能：基于YOLOv8的智能目标检测与计数后端API
技术栈：Python3.11 + FastAPI + YOLOv8 + OpenCV
"""
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import base64
import cv2
import numpy as np
from utils import get_object_count
import uvicorn
import time
# ==========新增导入低光照增强模块==========
from enhance import low_light_enhance_image

# ====================== 应用初始化与全局配置 ======================
app = FastAPI(title="暖记PLM 目标检测接口")


# 统一错误响应格式
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.status_code, "msg": exc.detail, "data": None}
    )

# 跨域中间件：限制为小程序域名（部署时替换为实际域名）
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "https://your-domain.com").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局常量配置
MODEL_DIR = os.getenv("MODEL_DIR", "./weights")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 上传图片最大限制：10MB

# 支持的模型尺寸及对应文件名
MODEL_SIZES = {
    "n": "yolov8n.pt",
    "s": "yolov8s.pt",
    "m": "yolov8m.pt",
    "l": "yolov8l.pt",
    "x": "yolov8x.pt",
}
DEFAULT_MODEL_SIZE = os.getenv("DEFAULT_MODEL_SIZE", "n")

# 模型缓存：避免重复加载不同尺寸的模型
model_cache: dict[str, YOLO] = {}


def get_model(size: str = DEFAULT_MODEL_SIZE) -> YOLO:
    """获取指定尺寸的YOLO模型，首次加载时自动缓存"""
    if size not in MODEL_SIZES:
        size = DEFAULT_MODEL_SIZE
    if size not in model_cache:
        weight_file = MODEL_SIZES[size]
        weight_path = os.path.join(MODEL_DIR, weight_file)
        if not os.path.exists(weight_path):
            weight_path = os.path.join(MODEL_DIR, MODEL_SIZES[DEFAULT_MODEL_SIZE])
            logger.warning(f"模型 {weight_file} 不存在，回退到默认模型 {MODEL_SIZES[DEFAULT_MODEL_SIZE]}")
        model_cache[size] = YOLO(weight_path)
        logger.info(f"YOLOv8{size} 模型加载完成")
    return model_cache[size]


# 启动时预加载默认模型
get_model(DEFAULT_MODEL_SIZE)
logger.info("暖记PLM后端服务就绪")


# ====================== 核心检测接口 ======================
@app.post("/api/detect")
async def detect_objects(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.5, ge=0.0, le=1.0),
    filter_classes: str | None = None,
    use_enhance: bool = False,
    enhance_intensity: float = Query(0.7, ge=0.0, le=1.0),
    model_size: str = Query("n", pattern="^[nsmxl]$")
):
    """
    目标检测接口
    :param file: 用户上传图片
    :param conf_threshold: 置信度阈值，过滤低置信度预测框
    :param filter_classes: 需要保留的类别id，逗号分隔字符串，例如"0"行人，"2"车辆；为空则检测全部类别
    :param use_enhance: 是否开启低光照图像增强（先增强，后检测）
    :param enhance_intensity: 增强强度 0.0~1.0
    :param model_size: 模型尺寸 n/s/m/l/x，越大精度越高但速度越慢
    :return: 识别总数、分类统计列表、带标注图片base64
    """
    start_time = time.time()

    # 校验文件类型，仅允许图片
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传有效的图片文件")

    # 读取图片二进制
    contents = await file.read()

    # 文件大小校验，防止超大图片占用内存
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="图片不能超过10MB，请压缩后重新上传")

    try:
        # 二进制数据解码为OpenCV图像
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="图片解析失败，请更换图片重试")

        # ==========【新增低光照增强逻辑】==========
        enhance_cost = 0.0
        enhanced_img = None
        if use_enhance:
            enhanced_img, enhance_cost = low_light_enhance_image(img, enhance_intensity)
            img = enhanced_img
        # ==========================================

        # 图片长边缩放，控制推理尺寸，提升速度
        h, w = img.shape[:2]
        max_size = 640
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))

        # 1. 先全局推理（不在这里限制classes）
        yolo = get_model(model_size)
        results = yolo.predict(img, conf=conf_threshold)
        result = results[0]

        # 2. 解析筛选类别列表
        target_class_ids = None
        if filter_classes and filter_classes.strip() != "":
            target_class_ids = [int(c) for c in filter_classes.split(",")]

        # 3. 手动过滤检测框【核心修复代码】
        if target_class_ids is not None and result.boxes is not None:
            # 筛选符合类别id的框
            mask = np.isin(result.boxes.cls.cpu().numpy(), target_class_ids)
            # 保留满足条件的框
            result.boxes = result.boxes[mask]

        # 绘制带检测框的结果图（此时只剩下目标类别）
        annotated_img = result.plot()

        # 将图片编码为base64，直接返回前端，无需持久化文件
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # 编码增强后的图片(供前端做原图对比)
        enhanced_img_b64 = ""
        if use_enhance and enhanced_img is not None:
            h2, w2 = enhanced_img.shape[:2]
            max_size2 = 640
            if max(h2, w2) > max_size2:
                scale2 = max_size2 / max(h2, w2)
                enhanced_img = cv2.resize(enhanced_img, (int(w2 * scale2), int(h2 * scale2)))
            _, buf2 = cv2.imencode('.jpg', enhanced_img)
            enhanced_img_b64 = base64.b64encode(buf2).decode('utf-8')

        # 统计各类目标数量
        count_dict = get_object_count(result, conf_threshold)
        count_list = [{"name": k, "num": v} for k, v in count_dict.items()]
        total_num = sum(count_dict.values())

        # 计算接口耗时并打印运行日志
        cost_time = round(time.time() - start_time, 3)
        logger.info(f"检测完成 模型:YOLOv8{model_size} 置信阈值:{conf_threshold} 筛选类别:{filter_classes} 增强:{use_enhance} 增强强度:{enhance_intensity} 增强耗时:{enhance_cost}s 目标数:{total_num} 总耗时:{cost_time}s")

        return {
            "code": 200,
            "msg": "检测成功",
            "cost_time": cost_time,
            "enhance_cost": enhance_cost,
            "model_size": model_size,
            "use_enhance": use_enhance,
            "enhance_intensity": enhance_intensity,
            "data": {
                "total": total_num,
                "count_list": count_list,
                "image_base64": img_base64,
                "enhanced_image_base64": enhanced_img_b64
            }
        }

    except Exception as e:
        logger.error(f"检测异常: {str(e)}")
        err_msg = str(e)
        if "cv2.imdecode" in err_msg or "NoneType" in err_msg:
            detail = "图片解析失败，请更换清晰有效的图片"
        else:
            detail = "检测服务异常，请稍后重试"
        raise HTTPException(status_code=500, detail=f"检测失败：{detail}")


# ====================== 低光照增强预览接口 ======================
@app.post("/api/enhance")
async def enhance_preview(
    file: UploadFile = File(...),
    intensity: float = Query(0.7, ge=0.0, le=1.0)
):
    """
    低光照增强预览：仅增强，不做检测，返回增强后的 base64 图片
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传有效的图片文件")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="图片不能超过10MB")

    try:
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="图片解析失败")

        enhanced, enhance_cost = low_light_enhance_image(img, intensity)

        _, buffer = cv2.imencode('.jpg', enhanced)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "code": 200,
            "msg": "增强成功",
            "enhance_cost": enhance_cost,
            "data": {
                "image_base64": img_base64
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"增强异常: {str(e)}")
        raise HTTPException(status_code=500, detail="图像增强失败，请稍后重试")


# ====================== 云托管健康检查接口 ======================
@app.get("/health")
@app.get("/")
def health_check():
    """
    服务健康检测（微信云托管专用）
    云托管平台会定期 ping 此接口来判断容器健康状态
    """
    return {"code": 200, "msg": "暖记PLM 后端服务运行正常", "status": "ok"}


# ====================== 模型信息接口 ======================
@app.get("/api/models")
def list_models():
    """返回可用模型列表及当前默认模型"""
    available = []
    for size, filename in MODEL_SIZES.items():
        weight_path = os.path.join(MODEL_DIR, filename)
        available.append({
            "size": size,
            "name": f"YOLOv8{size}",
            "filename": filename,
            "available": os.path.exists(weight_path)
        })
    return {
        "code": 200,
        "default_model": DEFAULT_MODEL_SIZE,
        "cached_models": list(model_cache.keys()),
        "models": available
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "80"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)