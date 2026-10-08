# TileLang FP32 attention

这个版本把输出累加留在芯片内部，并调整计算顺序，减少等待。
它用 Cube（矩阵乘法单元）算分数和输出贡献，用 Vector（向量计算单元）
做 FP32 softmax、累加和归一化。输入、两类单元之间传递的数据、最终输出均为 FP16。

它是独立实现，没有覆盖 `expert_v2` 或 Expert 对照版。
[中文优化历程](../../docs/optimization/tilelang-fp32.md)介绍为什么这样组织、
哪些改动有效，以及没有采用的尝试。

## 运行

准备已有的 CANN、PyTorch、torch_npu 和最新 `ascendc_pto` 主线的 TileLang-Ascend
环境，按[根目录说明](../../README.md#environment)配置 `env.local.sh`，然后：

```bash
source env.sh
npu-smi info
# 将 0 换成空闲设备。
python -m core.bench --backends expert_v3 --cases all --devices 0
```

框架会自动加入 Torch 参考实现并检查结果。本仓库不安装外部依赖。

## 支持范围

- 支持不带因果掩码的分组查询注意力（GQA）；FP16 输入和输出，每个 head 的维度为 128。
- Query 长度必须是 256 的整数倍，key 长度必须是 512 的整数倍；query head 数必须是 KV head 数的整数倍。
- 每组处理 256 个 query，每次访问 512 个 key，使用 24 组静态任务；不受 `FA_LOGICAL_BLOCKS` 影响。
- FP32 输出累加常驻片上向量存储区（UB），输入和输出各有两个 16 KiB 搬运槽。
- 使用 `T.reduce_max/sum`，不需要额外的融合softmax接口或框架补丁。
- 矩阵乘法与结果导出通过 UnitFlag 衔接；验证过的设备二进制没有整条 Vector 流水的 `BAR.V` 屏障。

正确性范围是仓库三个标准输入和验证过的有限值输入；Cube 通信仍是 FP16，
FP32 中间计算不代表支持任意数值范围。

## 修改源码

可编辑文件是 `impl_tl.py`。修改后在仓库根目录生成 `impl.py`：

```bash
python backends/expert_v3/preprocess.py \
  backends/expert_v3/impl_tl.py \
  -o backends/expert_v3/impl.py
```

生成工具已随本目录提供，不依赖原研究仓库或个人记忆目录。
