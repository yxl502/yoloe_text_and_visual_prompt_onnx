"""
YOLOE Visual Prompt 纯 ONNX 推理脚本

功能：
  1. 在参考图上用 bbox 作为视觉提示（Visual Prompt）
  2. 通过 VPE 提取 ONNX 模型提取视觉提示嵌入（VPE）
  3. 将 VPE 传入检测 ONNX 模型，在目标图上完成检测
  4. 支持同图推理和跨图推理
  5. 支持多个 visual prompt（多个 bbox + 不同类别）

不依赖 torch / ultralytics，仅使用 onnxruntime + numpy + opencv。

依赖：
  pip install onnxruntime numpy opencv-python

使用方式：
  # 同图推理（参考图 = 目标图）
  python yoloe_visual_prompt_onnx_infer.py \
      --vpe_onnx_path yoloe_v8l_vpe.onnx \
      --det_onnx_path yoloe_v8l_vp_det.onnx \
      --ref_img_path ultralytics/assets/bus.jpg \
      --target_img_path ultralytics/assets/bus.jpg \
      --bboxes "221.52,405.8,344.98,857.54" \
      --cls 0 \
      --conf_thres 0.15 --iou_thres 0.7

  # 跨图推理（参考图 != 目标图）
  python yoloe_visual_prompt_onnx_infer.py \
      --vpe_onnx_path yoloe_v8l_vpe.onnx \
      --det_onnx_path yoloe_v8l_vp_det.onnx \
      --ref_img_path ultralytics/assets/bus.jpg \
      --target_img_path ultralytics/assets/zidane.jpg \
      --bboxes "221.52,405.8,344.98,857.54" \
      --cls 0 \
      --conf_thres 0.15 --iou_thres 0.7

  # 多个视觉提示
  python yoloe_visual_prompt_onnx_infer.py \
      --vpe_onnx_path yoloe_v8l_vpe.onnx \
      --det_onnx_path yoloe_v8l_vp_det.onnx \
      --ref_img_path ultralytics/assets/bus.jpg \
      --target_img_path ultralytics/assets/bus.jpg \
      --bboxes "221.52,405.8,344.98,857.54" "120,425,160,445" \
      --cls 0 1 \
      --conf_thres 0.15 --iou_thres 0.7




      python yoloe_visual_prompt_onnx_infer.py \
      --vpe_onnx_path yoloe_v8l_vpe.onnx \
      --det_onnx_path yoloe_v8l_vp_det.onnx \
      --ref_img_path ultralytics/assets/DJI_20260327171056_0090_V.jpg \
      --target_img_path ultralytics/assets/DJI_20260327171059_0091_V.jpg \
      --bboxes "1403, 2063, 1779, 2299" \
      --cls 0 \
      --conf_thres 0.15 --iou_thres 0.7


"""

import argparse

import cv2
import numpy as np
import onnxruntime as ort


class YoloeVisualPromptOnnx:
    """YOLOE Visual Prompt 纯 ONNX 推理器。

    流程：
      1. 用 VPE 提取 ONNX 模型从参考图 + bbox 提取 VPE（视觉提示嵌入）
      2. 将 VPE 传入检测 ONNX 模型，在目标图上检测

    不依赖 PyTorch / ultralytics，visual prompt mask 的构建使用纯 numpy 实现，
    复刻 ultralytics.data.augment.LoadVisualPrompt 的逻辑。
    """

    def __init__(
        self,
        vpe_onnx_path,
        det_onnx_path,
        confidence_thres=0.15,
        iou_thres=0.7,
        input_height=640,
        input_width=640,
    ):
        self.input_height = input_height
        self.input_width = input_width
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        # SAVPE 使用 stride 8 分辨率的 visual prompt mask
        self.mask_stride = 8
        self.mask_h = input_height // self.mask_stride   # 80
        self.mask_w = input_width // self.mask_stride     # 80

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # --- 加载 VPE 提取 ONNX 模型 ---
        self.vpe_session = ort.InferenceSession(vpe_onnx_path, providers=providers)
        self.vpe_input_names = [inp.name for inp in self.vpe_session.get_inputs()]
        # 从 vpe_mask 输入维度推断模型固定的类别数
        self.vpe_nc = None
        for inp in self.vpe_session.get_inputs():
            if inp.name in ("vpe_mask", "vpe", "mask") and len(inp.shape) >= 2:
                dim1 = inp.shape[1]
                if isinstance(dim1, int):
                    self.vpe_nc = dim1
                    break
        print(f"[INFO] VPE ONNX 输入: {self.vpe_input_names}  nc={self.vpe_nc}")

        # --- 加载检测 ONNX 模型 ---
        self.det_session = ort.InferenceSession(det_onnx_path, providers=providers)
        self.det_input_names = [inp.name for inp in self.det_session.get_inputs()]
        # 从 vpe 输入或输出维度推断检测模型的类别数
        self.det_nc = None
        for inp in self.det_session.get_inputs():
            if inp.name in ("vpe", "tpe", "cls_pe") and len(inp.shape) >= 2:
                dim1 = inp.shape[1]
                if isinstance(dim1, int):
                    self.det_nc = dim1
                    break
        if self.det_nc is None:
            outputs = self.det_session.get_outputs()
            out_shape = outputs[0].shape
            if len(out_shape) >= 2 and isinstance(out_shape[1], int):
                self.det_nc = out_shape[1] - 4
        print(f"[INFO] DET ONNX 输入: {self.det_input_names}  nc={self.det_nc}")

    # ===================== Visual Mask 构建（纯 numpy）=====================

    def build_visual_mask(self, ref_img, bboxes, cls_labels):
        """从参考图和 bbox 构建 visual prompt mask（纯 numpy 实现）。

        复刻 ultralytics LoadVisualPrompt 的逻辑：
          1. 计算 LetterBox 缩放参数（与 preprocess 一致）
          2. bbox 缩放到 LetterBox 坐标，再缩放到 mask 分辨率（stride 8 → 80x80）
          3. 生成二值 mask（复刻 make_mask）
          4. 按类别合并（同一类别的多个 bbox 取并集）

        Args:
            ref_img: 参考图（BGR, HWC, uint8）
            bboxes: bbox 列表，每个为 [x1, y1, x2, y2]（原图坐标）
            cls_labels: 类别索引列表

        Returns:
            vpe_mask: [1, Q, mask_h, mask_w] 的二值 mask（float32）
                     Q = 实际类别数（未补零）
        """
        # 1. 计算 LetterBox 参数（与 preprocess 使用相同公式）
        shape = ref_img.shape[:2]  # (H, W)
        r = min(self.input_height / shape[0], self.input_width / shape[1])
        pad_w = (self.input_width - int(round(shape[1] * r))) / 2
        pad_h = (self.input_height - int(round(shape[0] * r))) / 2

        # 2. 缩放 bbox: 原图坐标 → LetterBox 坐标 → mask 坐标
        mask_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            # LetterBox 缩放: 乘 r 加 padding
            x1 = x1 * r + pad_w
            y1 = y1 * r + pad_h
            x2 = x2 * r + pad_w
            y2 = y2 * r + pad_h
            # 缩放到 mask 分辨率: 除以 stride 8
            # (LetterBox 坐标 / 640 * 80 = LetterBox 坐标 / 8)
            x1 /= self.mask_stride
            y1 /= self.mask_stride
            x2 /= self.mask_stride
            y2 /= self.mask_stride
            mask_bboxes.append([x1, y1, x2, y2])

        mask_bboxes = np.array(mask_bboxes, dtype=np.float32)  # (n, 4)

        # 3. 生成二值 mask（复刻 LoadVisualPrompt.make_mask）
        #    对每个 bbox，在 mask 空间内标记属于该 bbox 的像素
        n = mask_bboxes.shape[0]
        bx1 = mask_bboxes[:, 0][:, None, None]  # (n, 1, 1)
        by1 = mask_bboxes[:, 1][:, None, None]
        bx2 = mask_bboxes[:, 2][:, None, None]
        by2 = mask_bboxes[:, 3][:, None, None]
        cols = np.arange(self.mask_w)[None, None, :]  # (1, 1, W)
        rows = np.arange(self.mask_h)[None, :, None]  # (1, H, 1)
        masks = ((cols >= bx1) & (cols < bx2) & (rows >= by1) & (rows < by2))  # (n, H, W)
        masks = masks.astype(np.float32)

        # 4. 按类别合并（复刻 LoadVisualPrompt 的合并逻辑）
        #    同一类别的多个 bbox mask 取并集
        cls_unique = sorted(set(cls_labels))
        cls_to_idx = {c: i for i, c in enumerate(cls_unique)}
        Q = len(cls_unique)
        visuals = np.zeros((Q, self.mask_h, self.mask_w), dtype=np.float32)
        for i, cls in enumerate(cls_labels):
            idx = cls_to_idx[cls]
            visuals[idx] = np.logical_or(visuals[idx], masks[i]).astype(np.float32)

        # 5. 添加 batch 维度
        vpe_mask = visuals[None]  # (1, Q, H, W)
        return vpe_mask

    # ===================== 图像预处理 =====================

    def preprocess(self, img):
        """LetterBox 预处理，与 yoloe_text_prompt_onnx_infer.py 一致。

        Returns:
            img_process: [1, 3, H, W] float32, 0-1, RGB, CHW
            ratio: (r, r) 缩放比例
            (pad_w, pad_h): padding 偏移
        """
        shape = img.shape[:2]
        new_shape = (self.input_height, self.input_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        img = (
            np.ascontiguousarray(np.einsum("HWC->CHW", img)[::-1], dtype=np.float32)
            / 255.0
        )
        img_process = img[None] if len(img.shape) == 3 else img
        return img_process, ratio, (pad_w, pad_h)

    # ===================== VPE 提取（ONNX）=====================

    def extract_vpe(self, ref_img, bboxes, cls_labels):
        """用 VPE 提取 ONNX 模型从参考图 + bbox 提取 VPE。

        Args:
            ref_img: 参考图（BGR, HWC, uint8）
            bboxes: bbox 列表
            cls_labels: 类别索引列表

        Returns:
            vpe: numpy array [1, nc, 512]，视觉提示嵌入（已补零到 vpe_nc）
        """
        # 1. 预处理参考图（LetterBox + 归一化）
        ref_input, _, _ = self.preprocess(ref_img)

        # 2. 构建 visual prompt mask（纯 numpy）
        vpe_mask = self.build_visual_mask(ref_img, bboxes, cls_labels)
        Q = vpe_mask.shape[1]  # 实际类别数

        # 3. 补零到 VPE ONNX 模型固定的类别数
        #    VPE 提取模型的 Q 维度在导出时固定为 num_classes
        #    推理时实际类别数不足的部分用零 mask 补齐
        if self.vpe_nc is not None and Q < self.vpe_nc:
            pad_count = self.vpe_nc - Q
            pad_mask = np.zeros(
                (1, pad_count, self.mask_h, self.mask_w), dtype=np.float32
            )
            vpe_mask = np.concatenate([vpe_mask, pad_mask], axis=1)
            print(f"[INFO] Visual mask 补零: {Q} → {self.vpe_nc} 类别")

        # 4. 运行 VPE 提取 ONNX 模型
        feed_dict = {}
        for name in self.vpe_input_names:
            if name in ("x", "images", "input"):
                feed_dict[name] = ref_input
            elif name in ("vpe_mask", "vpe", "mask"):
                feed_dict[name] = vpe_mask.astype(np.float32)
            else:
                # 未知输入名，按名称猜测
                if "image" in name.lower() or name.lower() == "x":
                    feed_dict[name] = ref_input
                else:
                    feed_dict[name] = vpe_mask.astype(np.float32)

        vpe = self.vpe_session.run(None, feed_dict)[0]  # [1, vpe_nc, 512]

        # 5. 设置类别信息
        cls_unique = sorted(set(cls_labels))
        self.class_names = [f"object{int(c)}" for c in cls_unique]
        self.num_classes = Q  # 实际 prompt 类别数
        print(f"[INFO] VPE shape: {vpe.shape}  classes: {self.class_names}")
        return vpe

    # ===================== 后处理 =====================

    def postprocess(self, image_ori, x, pad_w, pad_h, ratio):
        """NMS + 坐标还原，与 yoloe_text_prompt_onnx_infer.py 一致。"""
        x = np.einsum("bcn->bnc", x)
        x = x[np.amax(x[..., 4:], axis=-1) > self.confidence_thres]
        if len(x) == 0:
            return x
        x = np.c_[
            x[..., :4], np.amax(x[..., 4:], axis=-1), np.argmax(x[..., 4:], axis=-1)
        ]
        nms_idx = cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], self.confidence_thres, self.iou_thres)
        # NMSBoxes 在无结果时返回空 tuple，需保护
        if isinstance(nms_idx, tuple) and len(nms_idx) == 0:
            return np.empty((0, 6))
        x = x[nms_idx]
        if len(x) > 0:
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)

            x[..., [0, 2]] = x[:, [0, 2]].clip(0, image_ori.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, image_ori.shape[0])
        return x

    # ===================== 完整推理流程 =====================

    def process(self, ref_img, target_img, bboxes, cls_labels):
        """完整 visual prompt 推理流程。

        Args:
            ref_img: 参考图（BGR, HWC, uint8）
            target_img: 目标图（BGR, HWC, uint8）
            bboxes: bbox 列表，每个为 [x1, y1, x2, y2]
            cls_labels: 类别索引列表

        Returns:
            det_bbx: 检测结果数组 [N, 6]（x1, y1, x2, y2, score, cls_idx）
        """
        # 步骤1：从参考图提取 VPE（纯 ONNX）
        vpe = self.extract_vpe(ref_img, bboxes, cls_labels)

        # 补齐 VPE 到检测模型固定的类别数
        if self.det_nc is not None and vpe.shape[1] < self.det_nc:
            pad_count = self.det_nc - vpe.shape[1]
            vpe_pad = np.zeros((1, pad_count, vpe.shape[2]), dtype=np.float32)
            vpe = np.concatenate([vpe, vpe_pad], axis=1)
            print(f"[INFO] VPE 补零: {self.num_classes} → {self.det_nc} 类别")
        elif self.det_nc is not None and vpe.shape[1] > self.det_nc:
            print(f"[WARNING] VPE 类别数 {vpe.shape[1]} > DET nc {self.det_nc}，截断")
            vpe = vpe[:, :self.det_nc, :]

        # 步骤2：用 VPE 作为提示嵌入，在目标图上检测
        img_process, ratio, (pad_w, pad_h) = self.preprocess(target_img)

        feed_dict = {}
        for name in self.det_input_names:
            if name in ("x", "images", "input"):
                feed_dict[name] = img_process
            elif name in ("tpe", "vpe", "cls_pe", "text_pe"):
                feed_dict[name] = vpe.astype(np.float32)
            else:
                # 未知输入，尝试匹配
                if "x" in name.lower() or "image" in name.lower():
                    feed_dict[name] = img_process
                elif "pe" in name.lower() or "prompt" in name.lower():
                    feed_dict[name] = vpe.astype(np.float32)

        print(f"[INFO] DET ONNX 输入: {list(feed_dict.keys())}")
        res = self.det_session.run(None, feed_dict)[0]

        # 过滤补零类别：补零 VPE 是全零向量，einsum 输出为 0，sigmoid(0)=0.5，
        # 远超置信度阈值，产生大量误检。只保留真实类别（前 self.num_classes 个）
        if self.det_nc is not None and self.num_classes < self.det_nc:
            # res shape: [1, 4+nc, anchors] → 只保留前 4+num_classes 列
            res = res[:, :4 + self.num_classes, :]
            print(f"[INFO] 过滤补零类别: 保留前 {self.num_classes} 个类别")

        # 步骤3：后处理
        det_bbx = self.postprocess(target_img, res, pad_w, pad_h, ratio)
        return det_bbx

    # ===================== 可视化 =====================

    def draw_detections(self, img, box, class_names=None):
        """绘制检测结果。"""
        if class_names is None:
            class_names = self.class_names

        colors = [
            (255, 0, 255), (0, 255, 255), (255, 165, 0), (128, 0, 128),
            (0, 128, 128), (255, 192, 203), (255, 0, 0), (0, 255, 0),
            (0, 0, 255), (255, 255, 0),
        ]

        for i in range(box.shape[0]):
            x1, y1, x2, y2, score, idx = box[i]
            color = colors[int(idx) % len(colors)]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            label = f"{class_names[int(idx)]}: {score:.3f}"
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            label_x = int(x1)
            label_y = int(y1 - 10) if y1 - 10 > label_height else int(y1 + 10)

            cv2.rectangle(
                img,
                (label_x, label_y - label_height),
                (label_x + label_width, label_y + label_height),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                img, label, (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )
        return img

    def draw_visual_prompt(self, img, bboxes, cls_labels):
        """在参考图上绘制 visual prompt bbox。"""
        colors = [
            (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
        ]
        for bbox, cls_idx in zip(bboxes, cls_labels):
            x1, y1, x2, y2 = [int(v) for v in bbox]
            color = colors[int(cls_idx) % len(colors)]
            cv2.rectangle(img, (x1, y1), (int(x2), int(y2)), color, 2)
            cv2.putText(
                img, f"prompt:{int(cls_idx)}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
        return img


def parse_bbox_args(bbox_str_list):
    """解析命令行 bbox 参数。

    每个元素是逗号分隔的 "x1,y1,x2,y2"。
    返回 [[x1,y1,x2,y2], ...] 和对应的 cls 列表。
    """
    bboxes = []
    for s in bbox_str_list:
        coords = [float(v.strip()) for v in s.split(",")]
        assert len(coords) == 4, f"bbox 需要 4 个坐标值: {s}"
        bboxes.append(coords)
    return bboxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOE Visual Prompt 纯 ONNX 推理")
    parser.add_argument(
        "--vpe_onnx_path", type=str, required=True,
        help="VPE 提取 ONNX 模型路径（如 yoloe_v8l_vpe.onnx）",
    )
    parser.add_argument(
        "--det_onnx_path", type=str, required=True,
        help="检测 ONNX 模型路径（非 fused，支持 vpe 输入）",
    )
    parser.add_argument(
        "--ref_img_path", type=str, required=True,
        help="参考图路径（在其上画 bbox 作为 visual prompt）",
    )
    parser.add_argument(
        "--target_img_path", type=str, default=None,
        help="目标图路径（检测目标）。如不指定，默认与参考图相同",
    )
    parser.add_argument(
        "--bboxes", type=str, nargs="+", required=True,
        help='Visual prompt bbox，格式: "x1,y1,x2,y2"（可多个，空格分隔）',
    )
    parser.add_argument(
        "--cls", type=int, nargs="+", required=True,
        help="每个 bbox 对应的类别索引（空格分隔）",
    )
    parser.add_argument(
        "--conf_thres", type=float, default=0.15,
        help="置信度阈值（visual prompt 建议设低一些）",
    )
    parser.add_argument(
        "--iou_thres", type=float, default=0.7,
        help="NMS IoU 阈值",
    )
    parser.add_argument(
        "--save_path", type=str, default="result_vp.jpg",
        help="结果保存路径",
    )
    parser.add_argument(
        "--save_ref_path", type=str, default="ref_prompt.jpg",
        help="参考图（带 visual prompt 标注）保存路径",
    )

    args = parser.parse_args()

    # 解析 bbox 和 cls
    bboxes = parse_bbox_args(args.bboxes)
    cls_labels = args.cls
    assert len(bboxes) == len(cls_labels), "bbox 数量与 cls 数量不匹配"

    # 目标图默认与参考图相同
    target_img_path = args.target_img_path or args.ref_img_path

    # 初始化推理器（纯 ONNX，不依赖 torch / ultralytics）
    yoloe = YoloeVisualPromptOnnx(
        args.vpe_onnx_path, args.det_onnx_path,
        args.conf_thres, args.iou_thres,
    )

    # 读取图像
    ref_img = cv2.imread(args.ref_img_path)
    target_img = cv2.imread(target_img_path)
    assert ref_img is not None, f"无法读取参考图: {args.ref_img_path}"
    assert target_img is not None, f"无法读取目标图: {target_img_path}"

    print(f"[INFO] 参考图: {args.ref_img_path}  shape: {ref_img.shape}")
    print(f"[INFO] 目标图: {target_img_path}  shape: {target_img.shape}")
    print(f"[INFO] Visual prompt bbox: {bboxes}")
    print(f"[INFO] 类别索引: {cls_labels}")

    # 推理
    det_bbx = yoloe.process(ref_img, target_img, bboxes, cls_labels)
    print(f"[INFO] 检测结果: {det_bbx.shape[0] if len(det_bbx) > 0 else 0} 个目标")
    if len(det_bbx) > 0:
        for i, (x1, y1, x2, y2, score, idx) in enumerate(det_bbx):
            print(f"  [{i}] {yoloe.class_names[int(idx)]}: {score:.3f}  "
                  f"bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

    # 可视化并保存
    img_viz = yoloe.draw_detections(target_img.copy(), det_bbx)
    cv2.imwrite(args.save_path, img_viz)
    print(f"[INFO] 检测结果已保存: {args.save_path}")

    # 保存参考图（带 visual prompt 标注）
    ref_viz = yoloe.draw_visual_prompt(ref_img.copy(), bboxes, cls_labels)
    cv2.imwrite(args.save_ref_path, ref_viz)
    print(f"[INFO] 参考图标注已保存: {args.save_ref_path}")


"""
使用示例：

# 1. 首先导出两个 ONNX 模型（VPE 提取 + 检测）
#    参见 export_vp_onnx.py
python export_vp_onnx.py \
    --yoloe_model_path pretrain/yoloe-v8l-seg.pt \
    --export_type all \
    --num_classes 5

# 2. 同图推理
python yoloe_visual_prompt_onnx_infer.py \
    --vpe_onnx_path yoloe_v8l_vpe.onnx \
    --det_onnx_path yoloe_v8l_vp_det.onnx \
    --ref_img_path ultralytics/assets/bus.jpg \
    --target_img_path ultralytics/assets/bus.jpg \
    --bboxes "221.52,405.8,344.98,857.54" \
    --cls 0 \
    --conf_thres 0.15 --iou_thres 0.7

# 3. 跨图推理（在 bus.jpg 上画 person 的 bbox，在 zidane.jpg 上检测 person）
python yoloe_visual_prompt_onnx_infer.py \
    --vpe_onnx_path yoloe_v8l_vpe.onnx \
    --det_onnx_path yoloe_v8l_vp_det.onnx \
    --ref_img_path ultralytics/assets/bus.jpg \
    --target_img_path ultralytics/assets/zidane.jpg \
    --bboxes "221.52,405.8,344.98,857.54" \
    --cls 0 \
    --conf_thres 0.15 --iou_thres 0.7

# 4. 多个视觉提示
python yoloe_visual_prompt_onnx_infer.py \
    --vpe_onnx_path yoloe_v8l_vpe.onnx \
    --det_onnx_path yoloe_v8l_vp_det.onnx \
    --ref_img_path ultralytics/assets/bus.jpg \
    --target_img_path ultralytics/assets/bus.jpg \
    --bboxes "221.52,405.8,344.98,857.54" "120,425,160,445" \
    --cls 0 1 \
    --conf_thres 0.15 --iou_thres 0.7
"""
