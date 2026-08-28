用户输入
  ├── 参考图 + bbox/mask (visual prompt)
  └── 目标图

步骤一：提取 VPE
  参考图 ──→ LetterBox(640×640) ──→ LoadVisualPrompt(1/8下采样)
     │                                    │
     │                              visual mask [1, Q, 80, 80]
     │                                    │
     └────────────────────────────────────┤
                    YOLOE Backbone 前向 → P3/P4/P5 特征
                                            │
                                    SAVPE(x, visual_mask)
                                            │
                                      VPE [1, Q, embed]
                                            │
                              model.set_classes(["object0"...], VPE)
                                            │
步骤二：目标图推理                          ↓
  目标图 ──→ LetterBox(640×640) ──→ YOLOE Backbone
                                            │
                                    get_cls_pe(tpe=None, vpe=VPE)
                                            │
                                    cls_pe [1, Q, embed]
                                            │
                              YOLOEDetect.forward(x, cls_pe)
                                            │
                                    cv2: bbox 回归
                                    cv4: BNContrastiveHead(cls_feat, cls_pe) → 分类得分
                                            │
                                    NMS → 最终检测结果



VPE 生成过程：bbox/mask → LetterBox 缩放 → LoadVisualPrompt 下采样 1/8 生成二值 mask [Q, H/8, W/8] → 通过 SAVPE 编码为嵌入 [Q, embed]
SAVPE 的核心：利用 visual mask 做空间注意力，将参考图中目标区域的特征加权聚合成类别级别的嵌入向量
VPE 与 TPE 的统一：两者都是 [B, N, embed] 形式的类别嵌入，通过 get_cls_pe 拼接后传入 detect head 的对比学习分类头 cv4（BNContrastiveHead）
ONNX 模型输入：文本模式为 {"x": image, "tpe": text_embedding}；Visual prompt 模式的 ONNX 推理脚本本项目未提供，需自行实现（将 VPE 替换 tpe 输入）
跨图 vs 同图：同图推理时 VPE 直接参与当前图检测；跨图推理时先用参考图提取 VPE，set_classes 注入后对目标图推理
