# Remove Hand

最终方案用于移除机器人采集视频中的绿色手套，并尽量保护操作物体和机械结构不被修改。当前正式入口只保留在 `Remove_Hand` 顶层。

## 运行环境

当前版本仅支持 H20 机器环境，依赖 H20 上已有的 CUDA/PyTorch、模型目录和 ProPainter 目录。暂时没有适配 PPU；不要在 PPU 上直接使用本版本。

## 最终方案

流程：

```text
读取 base_0_rgb.mp4
→ 每帧 GroundingDINO 检测手套/手部/前臂候选框
→ SAM 生成像素级候选 mask
→ 通用手套身份约束：边缘进入、轨迹连续、候选连通域一致
→ 保守时序稳定：不确定区域保留原视频
→ ProPainter 仅在最终 mask 内修复
→ mask 外逐像素恢复原视频
→ H.264 / yuv420p / faststart 输出
→ 写出 metadata.json、status.json、timing.json
```

核心准则：宁可少去一点手，也不能模糊或删除操作物体。

## 文件

```text
Remove_Hand/
├── remove_hand.py                  # 最终单视频处理入口
├── batch_remove_hand.py             # 多任务自动 GPU 并行批处理入口
├── test_remove_hand.py              # 每个任务只取一条视频的测试入口
├── model_ckpt/                       # GroundingDINO / SAM / LaMa 模型
├── third_party/                      # ProPainter
├── _grounded_sam.py                 # DINO/SAM 与视频编码工具
├── _propainter_utils.py             # ProPainter 调用与状态写入工具
├── _mask_utils.py                   # mask 连通域、保护区和 writer 工具
├── requirements.txt
└── README.md
```

## 单条视频

```bash
cd /mnt/data/ts/data_process_wujie_arx-ac1

source /mnt/data/ts/data_process_wujie_arx-ac1/.venv/bin/activate

python Remove_Hand/remove_hand.py \
  --input /mnt/data/ts/yuyan/arx2/anniu_0714_50/0000/camera/base_0_rgb.mp4 \
  --models-root /mnt/data/ts/data_process_wujie_arx-ac1/Remove_Hand/model_ckpt \
  --propainter-root /mnt/data/ts/data_process_wujie_arx-ac1/Remove_Hand/third_party/ProPainter \
  --output-dir /mnt/data/ts/remove_hand/manual_anniu_final
```

输出：

```text
<output-dir>/base_0_rgb_remove_hand.mp4
<output-dir>/metadata.json
<output-dir>/status.json
<output-dir>/timing.json
```


## 测试入口

正式批量写回原数据前，建议先用测试入口。它会对每个任务只取一条 `base_0_rgb.mp4`，并把所有结果保存到 `/mnt/data/ts/remove_hand-test/<timestamp>_<task_name>/`，不会写入原始数据目录。

```bash
cd /mnt/data/ts/data_process_wujie_arx-ac1

source /mnt/data/ts/data_process_wujie_arx-ac1/.venv/bin/activate

python Remove_Hand/test_remove_hand.py \
  --task-list /mnt/data/ts/yuyan/arx2/anniu_0714_50 \
  --task-list /mnt/data/ts/yuyan/arx2/bottle_0714_50 \
  --task-list /mnt/data/ts/yuyan/arx2/cha_hua_0714_50 \
  --task-list /mnt/data/ts/yuyan/arx2/cup_0714_50 \
  --task-list /mnt/data/ts/yuyan/arx2/open_guizi_0714_50 \
  --task-list /mnt/data/ts/yuyan/arx2/put_pen_0714_10 \
  --task-list /mnt/data/ts/yuyan/arx2/towel_0714_50 \
  --workers-per-gpu 1
```

测试输出示例：

```text
/mnt/data/ts/remove_hand-test/20260715_153000_anniu_0714_50/base_0_rgb_remove_hand.mp4
/mnt/data/ts/remove_hand-test/20260715_153000_anniu_0714_50/metadata.json
/mnt/data/ts/remove_hand-test/20260715_153000_anniu_0714_50/status.json
/mnt/data/ts/remove_hand-test/20260715_153000_anniu_0714_50/timing.json
/mnt/data/ts/remove_hand-test/20260715_153000_anniu_0714_50/run.log
```

可以指定某个 episode：

```bash
cd /mnt/data/ts/data_process_wujie_arx-ac1

source /mnt/data/ts/data_process_wujie_arx-ac1/.venv/bin/activate

python Remove_Hand/test_remove_hand.py \
  --task-list /mnt/data/ts/yuyan/arx2/anniu_0714_50 \
  --episode 0000
```

## 任务列表批处理

直接把多个任务目录通过重复的 `--task-list` 传入，不需要额外创建列表文件：

```bash
cd /mnt/data/ts/data_process_wujie_arx-ac1
source /mnt/data/ts/data_process_wujie_arx-ac1/.venv/bin/activate

python Remove_Hand/batch_remove_hand.py \
  --task-list /mnt/data/ts/yuyan/arx2/anniu_0714_50 \
  --task-list /mnt/data/ts/yuyan/arx2/cha_hua_0714_50 \
  --dry-run
```

正式处理会自动检测 `nvidia-smi` 中的全部 GPU，并按任务视频自动分发：

```bash
cd /mnt/data/ts/data_process_wujie_arx-ac1
source /mnt/data/ts/data_process_wujie_arx-ac1/.venv/bin/activate

python Remove_Hand/batch_remove_hand.py \
  --task-list /mnt/data/ts/yuyan/arx2/anniu_0714_50 \
  --task-list /mnt/data/ts/yuyan/arx2/cha_hua_0714_50 \
  --workers-per-gpu 1
```

也可以手动指定 GPU，并预留每卡多个 worker：

```bash
cd /mnt/data/ts/data_process_wujie_arx-ac1

source /mnt/data/ts/data_process_wujie_arx-ac1/.venv/bin/activate

python Remove_Hand/batch_remove_hand.py \
  --task-list /mnt/data/ts/yuyan/arx2/shiguan/shiguan_green_dan_0709_100 \
  --gpus 0,1,2,3,4,5,6,7 \
  --workers-per-gpu 2


cd /mnt/data/ts/data_process_wujie_arx-ac1

source /mnt/data/ts/data_process_wujie_arx-ac1/.venv/bin/activate

python Remove_Hand/batch_remove_hand.py \
  --task-list /mnt/data/ts/yuyan/arx2/phone/packing_phone_0710_13 \
  --task-list /mnt/data/ts/yuyan/arx2/put_pen_0714_10 \
  --task-list /mnt/data/ts/yuyan/arx2/deng_0715_10 \
  --task-list /mnt/data/ts/yuyan/arx1/peel/peel_huanggua_0719_60 \
  --task-list /mnt/data/ts/yuyan/arx1/peel/peel_huanggua_0720_36 \
  --task-list /mnt/data/ts/yuyan/arx1/peel/peel_huanggua_0716_51 \
  --gpus 0,1,2,3,4,5,6,7 \
  --workers-per-gpu 2
```

输出和原始视频在同一目录，但不会覆盖原始视频：

```text
<task>/<episode>/camera/base_0_rgb.mp4                 # 原始视频，只读
<task>/<episode>/camera/base_0_rgb_remove_hand.mp4     # 最终结果
<task>/<episode>/camera/remove_hand/metadata.json      # 配置和 mask 统计
<task>/<episode>/camera/remove_hand/status.json        # 状态
<task>/<episode>/camera/remove_hand/timing.json        # 耗时
<task>/<episode>/camera/remove_hand/*.log              # 日志
```

默认模型和第三方代码目录为：

```text
/mnt/data/ts/data_process_wujie_arx-ac1/Remove_Hand/model_ckpt
/mnt/data/ts/data_process_wujie_arx-ac1/Remove_Hand/third_party/ProPainter
```


## JSON 含义

`metadata.json` 记录输入输出、帧数、FPS、mask 像素统计、模型调用次数和运行参数。

`status.json` 记录任务状态，成功时应为 complete/ok，失败时包含错误阶段和错误信息。

`timing.json` 记录端到端和各阶段耗时。重点看：

- `wall_s`：总耗时
- `frames`：总帧数
- `wall_s / frames`：平均单帧耗时
- `dino_s`、`sam_s`、`propainter_s`：主要瓶颈
- `mask_total_s`：检测、分割和 mask 后处理总耗时


## 关键参数调节顺序

下面这些参数都可以直接加到 `remove_hand.py`、`test_remove_hand.py` 和 `batch_remove_hand.py` 命令后面。建议先用 `test_remove_hand.py` 小范围测试效果，再用 `batch_remove_hand.py` 写回原数据目录。

| 优先级 | 参数 | 默认值 | 主要作用 | 调大/调小影响 |
|---:|---|---:|---|---|
| 1 | `--mask-dilation` | `5` | 控制最终 mask 膨胀范围，是“手套残留 vs 误伤物体”的核心参数 | 调大：手套边缘残留更少，但更容易糊到物体/夹爪；调小：更保护物体，但可能留绿边 |
| 2 | `--propainter-mask-dilation` | `1` | 控制 ProPainter 内部再膨胀多少 mask | 调大：补得更彻底，但边界更容易发糊；调小：更保守，保护边界 |
| 3 | `--mask-mode` | `strict_color` | 控制 mask 颜色筛选模式 | `strict_color` 更保守；`color_shadow` 会覆盖更多绿色阴影，但更容易误伤绿色物体 |
| 4 | `--box-threshold` | `0.22` | GroundingDINO 检测框置信度阈值 | 调低：减少漏检但误检更多；调高：更干净但可能漏手 |
| 5 | `--text-threshold` | `0.22` | GroundingDINO 文本匹配阈值 | 调低：更容易检出；调高：更严格、更保守 |
| 6 | `--roi-bottom-ratio` | `0.38` | 过滤画面上方检测框，只关注操作区域 | 调大：更忽略上方，减少误检；调小：检测范围更广 |
| 7 | `--box-expand-ratio` | `0.20` | SAM 输入框扩张比例 | 调大：更容易包住完整手套；调小：更贴近检测框，减少误伤 |
| 8 | `--min-box-margin` | `12` | SAM 输入框最小扩张像素 | 小手套框也能有上下文；过大可能扩大误伤范围 |
| 9 | `--min-area` | `100` | 过滤小连通域噪声 | 调大：减少小噪点；调小：保留更小的手套碎片 |
| 10 | `--neighbor-length` | `8` | ProPainter 邻近参考帧数量 | 调大：时序更稳但更慢更吃显存；调小：更快但可能闪烁 |
| 11 | `--subvideo-length` | `20` | ProPainter 分段长度 | 调大：长时序一致性更好但更吃显存；调小：更省显存但段间可能不连续 |
| 12 | `--ref-stride` | `5` | ProPainter 远参考帧间隔 | 调小：参考更密但更慢；调大：更快但一致性可能下降 |
| 13 | `--raft-iter` | `10` | ProPainter 光流迭代次数 | 调大：光流更准但更慢；调小：更快但质量可能下降 |
| 14 | `--dino-interval` | `1` | GroundingDINO 运行间隔 | `1` 表示每帧跑，质量最好；调大可以加速，但之前测试会明显降低 mask 稳定性 |

推荐保守基线：

```bash
--dino-interval 1 \
--mask-mode strict_color \
--box-threshold 0.22 \
--text-threshold 0.22 \
--mask-dilation 5 \
--propainter-mask-dilation 1 \
--neighbor-length 8 \
--ref-stride 5 \
--subvideo-length 20 \
--raft-iter 10
```

如果操作物体被误伤，优先用这一组：

```bash
--mask-mode strict_color \
--mask-dilation 3 \
--propainter-mask-dilation 0 \
--box-threshold 0.26 \
--text-threshold 0.26
```

如果手套边缘或绿色阴影残留明显，优先用这一组：

```bash
--mask-mode color_shadow \
--mask-dilation 6 \
--propainter-mask-dilation 2 \
--box-threshold 0.20 \
--text-threshold 0.20
```

示例：测试两条任务并使用更保护物体的参数：

```bash
cd /mnt/data/ts/data_process_wujie_arx-ac1
source /mnt/data/ts/data_process_wujie_arx-ac1/.venv/bin/activate

python Remove_Hand/test_remove_hand.py \
  --task-list /mnt/data/ts/yuyan/arx2/anniu_0714_50 \
  --task-list /mnt/data/ts/yuyan/arx2/cha_hua_0714_50 \
  --mask-mode strict_color \
  --mask-dilation 3 \
  --propainter-mask-dilation 0 \
  --box-threshold 0.26 \
  --text-threshold 0.26
```

## 推荐原则

先看完整视频效果，再用 `timing.json` 判断瓶颈。当前质量优先方案耗时主要来自 SAM、ProPainter 和 DINO；JSON 统计耗时可忽略。
