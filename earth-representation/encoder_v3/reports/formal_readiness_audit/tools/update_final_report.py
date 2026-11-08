"""Update the final report after the 2026-11-08 approvals and controlled check."""
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "11_final_report.md"
text = path.read_text(encoding="utf-8")


def replace(old, new):
    global text
    assert old in text, old[:80]
    text = text.replace(old, new)


replace(
    "> 本报告不签发训练授权，不启动正式训练，不执行真实优化器更新，不使用已停止的 186 次预检预算，不训练任务头。历史 214 次预检更新保持原状。",
    "> 本报告不签发训练授权，不启动正式训练。仅在 D10 单独授权下执行了受控真实更新验证；该验证发生授权超用并失败，详见 `evidence/controlled_update/incident.json`。未使用已停止的 186 次预检预算，不训练任务头。",
)
replace("`be8d93…68eac`", "`ee3137…d5184`")
text = text.replace("2026-11-07\n", "2026-11-08\n", 1)
text = text.replace("222 项", "223 项").replace("222 通过", "223 通过").replace("222 项 GPU", "223 项 GPU")
text = text.replace("7 个 CUDA 跳过", "8 个 CUDA 跳过").replace("7 个 CUDA 用例", "8 个 CUDA 用例").replace("7 项随后", "8 项随后")
text = text.replace("37.5 s", "78.1 s")

replace(
    "| V3-AUDIT-015 | P1 | SRF/Decoder/Transformer/执行默认值未正式冻结 | 待决策 |",
    "| V3-AUDIT-015 | P1 | SRF/Decoder/Transformer/执行默认值未正式冻结 | 已批准冻结（D4-D6） |",
)
replace(
    "| V3-AUDIT-022 | P3 | 文件名发现依赖 `rg`，初始 PATH 缺失 | 环境说明，已记录 |",
    "| V3-AUDIT-022 | P3 | 文件名发现依赖 `rg`，初始 PATH 缺失 | 环境说明，已记录 |\n"
    "| V3-AUDIT-023 | P0 | D10 受控更新超用授权且未建立精确恢复：4 次唯一更新/6 步、首次精确比较失败、第二次因 optimizer 状态设备错误中止 | 未通过；restore 设备缺陷已修复，需重新授权修正版 |",
)

replace(
    "- **恢复**：合成检查点保存 Encoder/Decoder/AdamW/Scheduler/Teacher/Predictor/RNG/游标/身份；真实 checkpoint、真实更新后恢复未验证。",
    "- **恢复**：合成检查点保存 Encoder/Decoder/AdamW/Scheduler/Teacher/Predictor/RNG/游标/身份；D10 已生成受控真实 checkpoint，但精确恢复失败且发生授权超用；`restore_checkpoint` 的 optimizer 状态设备缺陷已修复，修正版需重新授权。",
)

replace(
    "**计数校正**：旧 admission 报告写 1/6/7；逐项实际为 **1 SATISFIED / 5 PARTIAL_PREFLIGHT / 8 NOT_SATISFIED**。按新结构化门禁，14/14 均未获正式批准。",
    "**计数校正**：旧 admission 报告写 1/6/7；逐项实际为 **1 SATISFIED / 5 PARTIAL_PREFLIGHT / 8 NOT_SATISFIED**。D4-D6 后，`optimizer_choice`、`merge_configuration`、`hypernetwork_hidden_structure`、`mask_rectangle_distribution` 已有结构化用户批准；其余 10 项未批准。",
)
replace(
    "| optimizer_choice | SATISFIED（AdamW，用户已选） | 未批准 |\n"
    "| merge_configuration | PARTIAL（单轮 B 已确认；曲线/ P 关系未正式） | 未批准 |\n"
    "| hypernetwork_hidden_structure | PARTIAL（仅预检） | 未批准 |",
    "| optimizer_choice | SATISFIED（AdamW，用户已选） | 已批准（D4-D6） |\n"
    "| merge_configuration | PARTIAL（单轮 B、曲线/P 关系已冻结；正式数据仍缺） | 已批准（D4-D6） |\n"
    "| hypernetwork_hidden_structure | PARTIAL（仅预检） | 已批准（D4-D6） |",
)
replace(
    "| mask_rectangle_distribution | PARTIAL（仅预检；单位语义待定） | 未批准 |",
    "| mask_rectangle_distribution | PARTIAL（仅预检；单位语义待定） | 已批准（D4-D6） |",
)

replace(
    "6. **无真实更新/恢复等价证据**；`execution_equivalence=BLOCKED`（V3-AUDIT-016）。",
    "6. **D10 受控真实更新已执行但授权超用且精确恢复失败**；4 次唯一更新/6 步、首次参数/优化器不相等、第二次中止；`execution_equivalence=BLOCKED`（V3-AUDIT-016/023）。",
)
replace(
    "因此不能判为 `ENGINEERING_READY_PENDING_APPROVAL`：即使研究者批准所有数据/科学项，仍缺受控真实更新验收和自动两路线/JEPA 编排；也不能生成 `FORMAL_TRAINING_AUTHORIZED`。",
    "因此不能判为 `ENGINEERING_READY_PENDING_APPROVAL`：即使研究者批准所有数据/科学项，仍缺正确的受控真实更新验收、正式数据/B2/面板和自动两路线/JEPA 编排；也不能生成 `FORMAL_TRAINING_AUTHORIZED`。",
)
replace(
    "5. **D10**：单独授权最小真实更新/恢复验证（2 次匹配更新 + 1 次恢复；≤20 分钟、≤10 GiB、独立账本，不使用已停止的 186）；通过后 `execution_equivalence` 才可申请通过。",
    "5. **D10 修正版**：先处理授权超用与证据不一致，再申请一次全新的 2 次唯一更新 + 1 次恢复（≤20 分钟、≤10 GiB、独立账本，不使用已停止的 186）；当前 D10 结果不得复用为通过证据。",
)

replace(
    "| `06_numerical_audit.json` | 223 项回归、数值检查、容差范围、真实无更新 |",
    "| `06_numerical_audit.json` | 223 项回归、数值检查、容差范围、真实无更新 + 受控更新事件 |",
)
replace(
    "| `08_execution_audit.json` | 0 真实更新、预算、受控验证请求 |",
    "| `08_execution_audit.json` | D10 受控更新超用/失败、预算不变、修正版请求 |",
)
replace(
    "| `evidence/full_suite_gpu.json` | 223 项 GPU 全量通过、CUDA 可用 |\n"
    "| `evidence/cpu_verify.json` | 223 项 CPU 运行（215 通过、8 个 CUDA 跳过） |",
    "| `evidence/full_suite_gpu.json` | 223 项 GPU 全量通过、CUDA 可用 |\n"
    "| `evidence/cpu_verify.json` | 223 项 CPU 运行（215 通过、8 个 CUDA 跳过） |\n"
    "| `evidence/controlled_update/incident.json` | D10 授权超用、失败原因、restore 设备修复 |\n"
    "| `evidence/controlled_update/report.json` | 首次受控更新报告（状态 FAILED） |\n"
    "| `approvals/D4_D6_formal_freeze.json` | D4-D6 用户批准与正式冻结记录 |\n"
    "| `03a_enmap_admission.json` | EnMAP 独立准入审查（未准入） |",
)

replace(
    "**结论重申**：本轮完成了当前版本的工程回归、结构/数据/任务核查、四个 A 类修复、真实无更新诊断和逐项准入矩阵；但正式训练仍被数据清单、B2=0、正式面板/日程、Runner、控制阈值、真实更新验收、JEPA 和授权硬性阻塞。**不得以本报告作为训练授权。**",
    "**结论重申**：本轮完成了当前版本的工程回归、结构/数据/任务核查、四个 A 类修复、真实无更新诊断和逐项准入矩阵；D4-D6 已按用户批准正式冻结，但 D10 受控更新发生授权超用并失败，必须重新授权修正版。正式训练仍被数据清单（含 EnMAP 独立准入）、B2=0、正式面板/日程、Runner 两路线/JEPA、控制阈值、真实更新验收和正式授权硬性阻塞。**不得以本报告作为训练授权。**",
)

path.write_text(text, encoding="utf-8")
print("updated", len(text))



