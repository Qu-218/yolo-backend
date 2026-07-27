"""
暖记PLM 工具函数
功能：YOLO检测结果目标计数
"""

def get_object_count(result, conf_threshold=0.5):
    """
    对YOLO推理结果进行目标统计
    :param result: YOLO单张图片推理结果对象
    :param conf_threshold: 置信度阈值
    :return: dict{类别名称:数量}
    """
    count_dict = {}
    names = result.names
    if result.boxes is not None:
        cls_ids = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        # 遍历所有检测框，过滤低置信目标
        for cid, conf in zip(cls_ids, confs):
            if conf >= conf_threshold:
                name = names[int(cid)]
                count_dict[name] = count_dict.get(name, 0) + 1
    return count_dict