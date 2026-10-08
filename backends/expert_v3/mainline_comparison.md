# 主线 Expert v1/v2/v3 的完整三点对照

## Motivation

用户要求用 FA 原 harness 的全部三个标准输入，比较旧版与 PR #1863 中的 v2/v3。
先同步公开实现，再直接运行 `core.bench`，不另写计时器或临时挑点。

## Expectation

来源实现的历史成绩使用不同计时方式，不能填进当前对照表。本次在同一设备、同一批
标准输入和主线编译器下确认三版的正确性与设备用时，不预设每个输入都更快。
使用原工具的 20 次内部采样，不另套外层测量轮次。

## Change

对应关系：

| Backend | TileLang PR 入口 | 本次原报告中的名字 |
|---|---|---|
| `expert_v1` | `fa_opt/expert_v1/kernel.py` | `tl_expert_v1` |
| `expert_v2` | `fa_opt/expert_v2/kernel.py` | `tl_stream_v3` |
| `expert_v3` | `fa_opt/expert_v3/kernel.py` | `tl_attention_fp32_v2` |

v1 是主线旧版，v2 从 TileLang 写法出发组织流水，v3 从已验证的 CCE 调度出发，再用
TileLang 表达。版本命名在测量完成后统一，内核的 `impl_tl.py/impl.py` 字节不变，
仅注册名和导入路径改变。报告保留实际测量时的旧名，不改写历史原始数据。

三版来自 [FA PR #1863](https://github.com/tile-ai/tilelang-ascend/pull/1863) 的
`cbfd67e9b7ccd227730d20e1e6d6a2ae08cde156`。v1/v2 仅适配符号名和 Host 入口；
v3 默认生产路径与 PR 生成的 C++ 逐字一致。源身份见
[source_manifest.json](source_manifest.json)。`core/`、`problem/` 的计时、正确性
与标准输入未改变。

2026-10-08，Atlas A3（Ascend910_9392），CANN 9.1.0-beta.1、torch_npu 2.10.0，
编译器为当日最新 `ascendc_pto@4c7267018af02284ff28f1cd090a1492c7985bed`。
三版均使用 24 个逻辑 Cube 核；v3 保留来源的 O3 编译配置。
输入输出 FP16，非因果 GQA，12 个 query 头、1 个 K/V 头、维度 128，seed 0。
完整输入来自 [cases.py](../../problem/cases.py) 和
[canonical.py](../../problem/canonical.py)。

在当前新命名下复现：

```bash
source env.sh
npu-smi info
python -m core.bench --backends expert_v1,expert_v2,expert_v3 --cases all --devices 14
```

14 是本次设备，不是固定要求。CPU 阶段编译 9 个内核，随后在同一设备上预热 3 次，
统计原 torch_npu profiler 的 20 次设备内核 Duration 均值，不计编译、分配或 Host
布局准备。正确性参考为 `npu_fusion_attention`，`atol=rtol=1e-2`。
此表保留该编译器身份；日后主线更新后要重新验证，不能直接换标签。

## Result

三点、三版共 9 项正确性检查全部 PASS，最大绝对误差不超过 0.00012。
v2 相对 v1 为 1.605–1.623x，v3 相对 v1 为 1.917–1.920x，
v3 相对 v2 为 1.183–1.194x。保留三条可选择路线，不以 v3 覆盖 v1/v2；
收益仅限这些输入和环境，不代表所有输入上最快。

| Batch | 序列长度 | expert_v1，ms | expert_v2，ms | expert_v3，ms |
|---:|---:|---:|---:|---:|
| 2 | 131072 | 1792.043 | 1116.421 | 934.714 |
| 2 | 65536 | 447.936 | 276.739 | 233.308 |
| 1 | 32768 | 56.062 | 34.541 | 29.194 |

[mainline_comparison.csv](mainline_comparison.csv) 保留原报告的全部 12 行，含参考、
设备用时与逐项正确性。不混用首页历史中位数或来源优化历程的数字。
23 项 Host 回归通过；更名后的入口另外用原矩阵中的 32K 输入验证。
