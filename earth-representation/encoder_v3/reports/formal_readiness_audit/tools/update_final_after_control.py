from pathlib import Path

path = Path(__file__).resolve().parents[1] / "11_final_report.md"
text = path.read_text(encoding="utf-8")


def replace(old, new):
    global text
    assert old in text, old[:80]
    text = text.replace(old, new)


replace(
    "> 本报告不签发训练授权，不启动正式训练。仅在 D10 单独授权下执行了受控真实更新验证；该验证发生授权超用并失败，详见 `evidence/controlled_update/incident.json`。未使用已停止的 186 次预检预算，不训练任务头。",
    "> 本报告不签发训练授权，不启动正式训练。仅在 D10 单独授权下执行了受控真实更新验证；先前超用经用户 addendum 接受，修正版 2 次唯一更新 + 1 次恢复通过旧 FP32 预检容差（非正式新架构容差）。未使用已停止的 186 次预检预算，不训练任务头。",
)
replace(
    "| V3-AUDIT-016 | P0 | 无真实更新/恢复等价证据；D10 授权超用且未通过 | `BLOCKED` |",
    "| V3-AUDIT-016 | P0 | 受控恢复仅在旧 FP32 预检容差内，不是正式新架构容差 | `PARTIAL_PREFLIGHT_TOLERANCE` |",
)
replace(
    "| V3-AUDIT-023 | P0 | D10 受控更新超用授权且未建立精确恢复：4 次唯一更新/6 步、首次精确比较失败、第二次因 optimizer 状态设备错误中止 | 未通过；restore 设备缺陷已修复，需重新授权修正版 |",
    "| V3-AUDIT-023 | P0 | 先前超用/失败后，修正版 2 次唯一更新 + 1 次恢复通过旧 FP32 预检容差（参数 max 7.45e-9、优化器相对 L2 1.47e-9、loss 差 2.47e-8）；非正式容差 | 旧容差内通过；正式验收仍未批准 |",
)
replace(
    "- **D10 受控更新（用户已授权）**：两次尝试实际产生 4 次唯一更新/6 个 optimizer step，超出 2 次唯一更新授权；首次 exact resume 失败（参数/优化器不相等，loss 差 5.03e-8），第二次在 optimizer 状态 CPU/CUDA 比较处中止。参数已真实改变，但 pilot 预算仍 214/214、186 未用；`execution_equivalence` 仍 BLOCKED。详见 `evidence/controlled_update/incident.json`。",
    "- **D10 受控更新（用户已授权，超用经 addendum 接受）**：三次尝试累计 6 次唯一更新/9 个 optimizer step；前两次失败（首次精确恢复不等、第二次 optimizer 状态设备错误）。修正版严格 2 次唯一更新 + 1 次恢复，参数最大差 7.45e-9、优化器状态相对 L2 1.47e-9、loss 差 2.47e-8，落在旧 FP32 预检容差内，但不是正式新架构容差；`execution_equivalence` 仍未正式批准。pilot 预算仍 214/214、186 未用。详见 `evidence/controlled_update/report.json` 与 `incident.json`。",
)
replace(
    "- **恢复**：合成检查点保存 Encoder/Decoder/AdamW/Scheduler/Teacher/Predictor/RNG/游标/身份；D10 已生成受控真实 checkpoint，但精确恢复失败且发生授权超用；`restore_checkpoint` 的 optimizer 状态设备缺陷已修复，修正版需重新授权。",
    "- **恢复**：合成检查点保存 Encoder/Decoder/AdamW/Scheduler/Teacher/Predictor/RNG/游标/身份；D10 修正版已在旧 FP32 预检容差内完成受控真实 checkpoint/恢复；`restore_checkpoint` 的 optimizer 状态设备缺陷已修复；正式新架构容差仍需批准。",
)
replace(
    "| execution_equivalence | NOT/BLOCKED（D10 超用且精确恢复失败） | 未批准 |",
    "| execution_equivalence | NOT/仅旧 FP32 预检容差（D10 修正版） | 未批准 |",
)
replace(
    "6. **D10 受控真实更新已执行但授权超用且精确恢复失败**；4 次唯一更新/6 步、首次参数/优化器不相等、第二次中止；`execution_equivalence=BLOCKED`（V3-AUDIT-016/023）。",
    "6. **D10 修正版通过旧 FP32 预检容差，但正式新架构容差/批准缺失**；先前超用已记录并经用户 addendum 接受；`execution_equivalence` 仍不能正式通过（V3-AUDIT-016/023）。",
)
replace(
    "5. **D10 修正版**：先处理授权超用与证据不一致，再申请一次全新的 2 次唯一更新 + 1 次恢复（≤20 分钟、≤10 GiB、独立账本，不使用已停止的 186）；当前 D10 结果不得复用为通过证据。",
    "5. **D10 修正版**：已完成 2 次唯一更新 + 1 次恢复并通过旧 FP32 预检容差；下一步是批准/冻结新架构正式容差并复核，不再重复真实更新。",
)
replace(
    "| `08_execution_audit.json` | D10 受控更新超用/失败、预算不变、修正版请求 |",
    "| `08_execution_audit.json` | D10 修正版旧容差通过、正式新架构容差未批、预算不变 |",
)
replace(
    "| `evidence/controlled_update/incident.json` | D10 授权超用、失败原因、restore 设备修复 |",
    "| `evidence/controlled_update/incident.json` | D10 超用/失败/修正版通过旧容差、restore 设备修复 |",
)
replace(
    "**结论重申**：本轮完成了当前版本的工程回归、结构/数据/任务核查、四个 A 类修复、真实无更新诊断和逐项准入矩阵；D4-D6 已按用户批准正式冻结，但 D10 受控更新发生授权超用并失败，必须重新授权修正版。正式训练仍被数据清单（含 EnMAP 独立准入）、B2=0、正式面板/日程、Runner 两路线/JEPA、控制阈值、真实更新验收和正式授权硬性阻塞。**不得以本报告作为训练授权。**",
    "**结论重申**：本轮完成了当前版本的工程回归、结构/数据/任务核查、五个已修复/冻结项、真实无更新诊断和逐项准入矩阵；D4-D6 已按用户批准正式冻结，D10 修正版在旧 FP32 预检容差内完成受控更新/恢复，但正式新架构容差未批准，先前超用仍记录在案。正式训练仍被数据清单（含 EnMAP 独立准入）、B2=0、正式面板/日程、Runner 两路线/JEPA、控制阈值、正式执行等价和正式授权硬性阻塞。**不得以本报告作为训练授权。**",
)

path.write_text(text, encoding="utf-8")
print("updated", len(text))
