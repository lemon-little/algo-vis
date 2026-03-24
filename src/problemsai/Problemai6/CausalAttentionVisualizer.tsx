import { ConfigurableVisualizer } from "@/components/visualizers/ConfigurableVisualizer";
import { CoreIdeaBox } from "@/components/visualizers/CoreIdeaBox";
import { ProblemInput } from "@/types/visualization";
import { generateCausalAttentionSteps } from "./algorithm";
import { getAiProblemCoreIdea } from "@/config/aiProblemCoreIdeas";

interface CausalAttentionInput extends ProblemInput {
  Q: string;
  K: string;
  V: string;
  dk: number;
  tokens: string;
}

const DEFAULT_Q = "[[1,0],[0,1],[1,1],[0,0]]";
const DEFAULT_K = "[[1,0],[0,1],[1,1],[0,0]]";
const DEFAULT_V = "[[1,0,0],[0,1,0],[1,1,0],[0,0,1]]";
const DEFAULT_DK = 2;
const DEFAULT_TOKENS = "The,cat,sat,down";

function parseMatrix(raw: string | number[][]): number[][] {
  if (Array.isArray(raw)) return raw as number[][];
  try {
    const parsed = JSON.parse(raw);
    if (
      Array.isArray(parsed) &&
      parsed.every(
        (row: unknown) =>
          Array.isArray(row) && (row as unknown[]).every((v) => typeof v === "number")
      )
    ) {
      return parsed as number[][];
    }
  } catch {
    // fall through
  }
  return JSON.parse(DEFAULT_Q) as number[][];
}

function parseTokens(raw: string, seqLen: number): string[] {
  const parts = raw.split(",").map((t) => t.trim()).filter(Boolean);
  if (parts.length === seqLen) return parts;
  return Array.from({ length: seqLen }, (_, i) => `t${i}`);
}

function fmt(v?: number | null, digits = 3): string {
  if (v === undefined || v === null || isNaN(v as number)) return "--";
  return (v as number).toFixed(digits);
}

function getHeatColor(weight: number): string {
  const w = Math.max(0, Math.min(weight, 1));
  const hue = 220 - w * 180;
  const sat = 70;
  const light = 88 - w * 30;
  return `hsl(${hue}, ${sat}%, ${light}%)`;
}

function getTextColor(weight: number): string {
  return weight > 0.5 ? "#fff" : "#1e293b";
}

interface MatrixGridProps {
  label: string;
  matrix: number[][];
  highlightRow?: number;
  badge?: string;
  rowLabels?: string[];
}

function MatrixGrid({ label, matrix, highlightRow, badge, rowLabels }: MatrixGridProps) {
  if (!matrix || matrix.length === 0) return null;
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-semibold text-gray-800">{label}</h4>
        {badge && (
          <span className="text-xs px-2 py-0.5 bg-blue-50 text-blue-700 rounded-full">
            {badge}
          </span>
        )}
      </div>
      <div className="overflow-x-auto">
        <table className="text-xs border-collapse">
          <tbody>
            {matrix.map((row, i) => (
              <tr key={i}>
                <td className="pr-2 text-gray-400 font-mono text-right w-10 truncate">
                  {rowLabels ? rowLabels[i] : `r${i}`}
                </td>
                {row.map((val, j) => {
                  const isHl = highlightRow === i;
                  return (
                    <td
                      key={j}
                      className={`w-14 h-9 text-center font-mono border border-gray-100 rounded transition-all ${
                        isHl ? "ring-2 ring-blue-400 font-semibold" : ""
                      }`}
                      style={{ backgroundColor: isHl ? "#eff6ff" : "#f8fafc" }}
                    >
                      {fmt(val, 2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

interface CausalMaskGridProps {
  label: string;
  mask: boolean[][];
  tokenLabels: string[];
  subtitle?: string;
}

function CausalMaskGrid({ label, mask, tokenLabels, subtitle }: CausalMaskGridProps) {
  if (!mask || mask.length === 0) return null;
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
      <div className="mb-3">
        <h4 className="text-sm font-semibold text-gray-800">{label}</h4>
        {subtitle && <p className="text-xs text-gray-500 mt-0.5">{subtitle}</p>}
      </div>
      <div className="overflow-x-auto">
        <table className="text-xs border-collapse">
          <thead>
            <tr>
              <td className="w-10" />
              {tokenLabels.map((tok, j) => (
                <td key={j} className="w-14 text-center text-gray-400 font-mono pb-1">
                  {tok}
                </td>
              ))}
            </tr>
          </thead>
          <tbody>
            {mask.map((row, i) => (
              <tr key={i}>
                <td className="pr-2 text-gray-400 font-mono text-right w-10 truncate">
                  {tokenLabels[i]}
                </td>
                {row.map((masked, j) => (
                  <td
                    key={j}
                    className="w-14 h-9 text-center font-mono border border-white/60 rounded transition-all"
                    style={{
                      backgroundColor: masked ? "#fef2f2" : "#f0fdf4",
                      color: masked ? "#dc2626" : "#16a34a",
                    }}
                    title={masked ? `${tokenLabels[i]} 不能看到 ${tokenLabels[j]}（未来）` : `${tokenLabels[i]} 可以看到 ${tokenLabels[j]}`}
                  >
                    {masked ? "✗" : "✓"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-2 flex items-center gap-4 text-[10px] text-gray-500">
        <span className="flex items-center gap-1">
          <span className="inline-block w-4 h-4 rounded bg-green-50 text-green-600 text-center">✓</span>
          允许（当前/过去位置）
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-4 h-4 rounded bg-red-50 text-red-600 text-center">✗</span>
          屏蔽（未来位置）
        </span>
      </div>
    </div>
  );
}

interface MaskedScoresGridProps {
  label: string;
  scaledScores: number[][];
  maskedDisplay: (number | null)[][];
  causalMask: boolean[][];
  tokenLabels: string[];
  subtitle?: string;
  highlightRow?: number;
}

function MaskedScoresGrid({
  label,
  scaledScores,
  maskedDisplay,
  causalMask,
  tokenLabels,
  subtitle,
  highlightRow,
}: MaskedScoresGridProps) {
  if (!maskedDisplay || maskedDisplay.length === 0) return null;
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
      <div className="mb-3">
        <h4 className="text-sm font-semibold text-gray-800">{label}</h4>
        {subtitle && <p className="text-xs text-gray-500 mt-0.5">{subtitle}</p>}
      </div>
      <div className="overflow-x-auto">
        <table className="text-xs border-collapse">
          <thead>
            <tr>
              <td className="w-10" />
              {tokenLabels.map((tok, j) => (
                <td key={j} className="w-14 text-center text-gray-400 font-mono pb-1">
                  {tok}
                </td>
              ))}
            </tr>
          </thead>
          <tbody>
            {maskedDisplay.map((row, i) => (
              <tr key={i}>
                <td className="pr-2 text-gray-400 font-mono text-right w-10 truncate">
                  {tokenLabels[i]}
                </td>
                {row.map((val, j) => {
                  const isMasked = causalMask[i][j];
                  const isHl = highlightRow === i;
                  const rawVal = scaledScores[i]?.[j];
                  return (
                    <td
                      key={j}
                      className={`w-14 h-9 text-center font-mono border rounded transition-all ${
                        isHl && !isMasked ? "ring-2 ring-blue-400" : ""
                      }`}
                      style={{
                        backgroundColor: isMasked ? "#fef2f2" : "#f8fafc",
                        color: isMasked ? "#dc2626" : "#1e293b",
                        borderColor: isMasked ? "#fecaca" : "#f1f5f9",
                      }}
                      title={isMasked ? `-∞（已掩码）` : `${fmt(rawVal, 4)}`}
                    >
                      {isMasked ? "-∞" : fmt(val, 2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

interface AttentionHeatmapProps {
  label: string;
  weights: number[][];
  causalMask: boolean[][];
  tokenLabels: string[];
  subtitle?: string;
}

function AttentionHeatmap({ label, weights, causalMask, tokenLabels, subtitle }: AttentionHeatmapProps) {
  if (!weights || weights.length === 0) return null;
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
      <div className="mb-3">
        <h4 className="text-sm font-semibold text-gray-800">{label}</h4>
        {subtitle && <p className="text-xs text-gray-500 mt-0.5">{subtitle}</p>}
      </div>
      <div className="overflow-x-auto">
        <table className="text-xs border-collapse">
          <thead>
            <tr>
              <td className="w-10" />
              {tokenLabels.map((tok, j) => (
                <td key={j} className="w-14 text-center text-gray-400 font-mono pb-1">
                  {tok}
                </td>
              ))}
            </tr>
          </thead>
          <tbody>
            {weights.map((row, i) => (
              <tr key={i}>
                <td className="pr-2 text-gray-400 font-mono text-right w-10 truncate">
                  {tokenLabels[i]}
                </td>
                {row.map((val, j) => {
                  const isMasked = causalMask[i][j];
                  const bg = isMasked ? "#fef2f2" : getHeatColor(val);
                  const tc = isMasked ? "#dc2626" : getTextColor(val);
                  return (
                    <td
                      key={j}
                      className="w-14 h-10 text-center font-mono border border-white/60 rounded transition-all"
                      style={{ backgroundColor: bg, color: tc }}
                      title={isMasked ? "已掩码（权重=0）" : `${tokenLabels[i]} → ${tokenLabels[j]}: ${fmt(val)}`}
                    >
                      {isMasked ? "0" : fmt(val, 2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-2 flex items-center gap-3 text-[10px] text-gray-400">
        <span className="inline-block w-4 h-2 rounded" style={{ background: getHeatColor(0) }} />
        <span>低注意力</span>
        <span className="inline-block w-4 h-2 rounded" style={{ background: getHeatColor(0.5) }} />
        <span className="inline-block w-4 h-2 rounded" style={{ background: getHeatColor(1) }} />
        <span>高注意力</span>
        <span className="inline-block w-4 h-2 rounded bg-red-100" />
        <span className="text-red-400">已掩码</span>
      </div>
    </div>
  );
}

function PhaseTag({ phase }: { phase: string }) {
  const map: Record<string, { label: string; color: string }> = {
    init: { label: "初始化", color: "bg-gray-100 text-gray-700" },
    mask: { label: "构建掩码", color: "bg-orange-100 text-orange-700" },
    "dot-product": { label: "点积 QKᵀ", color: "bg-blue-100 text-blue-700" },
    scale: { label: "缩放 /√d_k", color: "bg-violet-100 text-violet-700" },
    "apply-mask": { label: "应用掩码", color: "bg-red-100 text-red-700" },
    softmax: { label: "Softmax", color: "bg-amber-100 text-amber-700" },
    output: { label: "加权求和", color: "bg-emerald-100 text-emerald-700" },
    complete: { label: "计算完成", color: "bg-green-100 text-green-700" },
  };
  const info = map[phase] ?? { label: phase, color: "bg-gray-100 text-gray-700" };
  return (
    <span className={`text-xs px-3 py-1 rounded-full font-semibold ${info.color}`}>
      {info.label}
    </span>
  );
}

function CausalAttentionVisualizer() {
  const coreIdea = getAiProblemCoreIdea(10006);

  return (
    <ConfigurableVisualizer<CausalAttentionInput, Record<string, never>>
      config={{
        defaultInput: {
          Q: DEFAULT_Q,
          K: DEFAULT_K,
          V: DEFAULT_V,
          dk: DEFAULT_DK,
          tokens: DEFAULT_TOKENS,
        },
        algorithm: (input) => {
          const Q = parseMatrix(input.Q as string | number[][]);
          const K = parseMatrix(input.K as string | number[][]);
          const V = parseMatrix(input.V as string | number[][]);
          const dk =
            typeof input.dk === "number"
              ? input.dk
              : parseFloat(String(input.dk)) || DEFAULT_DK;
          const tokenStr = typeof input.tokens === "string" ? input.tokens : DEFAULT_TOKENS;
          const tokens = parseTokens(tokenStr, Q.length);
          return generateCausalAttentionSteps(Q, K, V, dk, tokens);
        },
        inputTypes: [
          { type: "string", key: "Q", label: "Q 矩阵（JSON）" },
          { type: "string", key: "K", label: "K 矩阵（JSON）" },
          { type: "string", key: "V", label: "V 矩阵（JSON）" },
          { type: "number", key: "dk", label: "d_k（Key 维度）", min: 1 },
          { type: "string", key: "tokens", label: "Token 标签（逗号分隔）" },
        ],
        inputFields: [
          { type: "string", key: "Q", label: "Q 矩阵（JSON）", placeholder: DEFAULT_Q },
          { type: "string", key: "K", label: "K 矩阵（JSON）", placeholder: DEFAULT_K },
          { type: "string", key: "V", label: "V 矩阵（JSON）", placeholder: DEFAULT_V },
          { type: "number", key: "dk", label: "d_k（Key 维度）", placeholder: String(DEFAULT_DK) },
          { type: "string", key: "tokens", label: "Token 标签（逗号分隔）", placeholder: DEFAULT_TOKENS },
        ],
        testCases: [
          {
            label: "示例（4 Token）",
            value: { Q: DEFAULT_Q, K: DEFAULT_K, V: DEFAULT_V, dk: DEFAULT_DK, tokens: DEFAULT_TOKENS },
          },
          {
            label: "3 Token / d_k=2",
            value: {
              Q: "[[1,2],[3,4],[0,1]]",
              K: "[[1,1],[2,2],[0,1]]",
              V: "[[1,0,0],[0,1,0],[0,0,1]]",
              dk: 2,
              tokens: "I,love,you",
            },
          },
          {
            label: "5 Token / d_k=4",
            value: {
              Q: "[[1,0,1,0],[0,1,0,1],[1,1,0,0],[0,0,1,1],[1,0,0,1]]",
              K: "[[1,0,1,0],[0,1,0,1],[1,1,0,0],[0,0,1,1],[1,0,0,1]]",
              V: "[[1,0],[0,1],[1,1],[0,0],[1,1]]",
              dk: 4,
              tokens: "A,B,C,D,E",
            },
          },
        ],
        render: ({ visualization, variables }) => {
          const currentInput = visualization.input as CausalAttentionInput;
          const Q = parseMatrix(currentInput.Q as string | number[][]);
          const K = parseMatrix(currentInput.K as string | number[][]);
          const V = parseMatrix(currentInput.V as string | number[][]);
          const tokenStr = typeof currentInput.tokens === "string" ? currentInput.tokens : DEFAULT_TOKENS;
          const tokenLabels = parseTokens(tokenStr, Q.length);

          const phase = (variables?.phase as string) ?? "init";
          const currentQueryIdx = variables?.currentQueryIdx as number | undefined;
          const causalMask = (variables?.causalMask as boolean[][] | undefined) ?? [];
          const scores = (variables?.scores as number[][] | undefined) ?? [];
          const scaledScores = (variables?.scaledScores as number[][] | undefined) ?? [];
          const maskedScoresDisplay = (variables?.maskedScoresDisplay as (number | null)[][] | undefined) ?? [];
          const attentionWeights = (variables?.attentionWeights as number[][] | undefined) ?? [];
          const output = (variables?.output as number[][] | undefined) ?? [];
          const scale = variables?.scale as number | undefined;

          const showMask = ["mask", "dot-product", "scale", "apply-mask", "softmax", "output", "complete"].includes(phase);
          const showScores = ["dot-product", "scale", "apply-mask", "softmax", "output", "complete"].includes(phase);
          const showMaskedScores = ["apply-mask", "softmax", "output", "complete"].includes(phase);
          const showWeights = ["softmax", "output", "complete"].includes(phase);
          const showOutput = ["output", "complete"].includes(phase);

          const phaseSteps = [
            { id: "init", label: "① 输入 Q/K/V" },
            { id: "mask", label: "② 构建掩码" },
            { id: "dot-product", label: "③ QKᵀ 点积" },
            { id: "scale", label: "④ /√d_k 缩放" },
            { id: "apply-mask", label: "⑤ 应用掩码" },
            { id: "softmax", label: "⑥ Softmax" },
            { id: "output", label: "⑦ × V 加权" },
            { id: "complete", label: "⑧ 输出" },
          ];

          return (
            <div className="space-y-4">
              {coreIdea && <CoreIdeaBox {...coreIdea} />}

              {/* 标题与公式 */}
              <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <div>
                    <h3 className="text-base font-semibold text-gray-900">
                      因果注意力（掩码注意力）
                    </h3>
                    <p className="text-xs text-gray-500 mt-0.5 font-mono">
                      Attention(Q,K,V) = softmax( mask( QKᵀ/√d_k ) ) · V
                    </p>
                  </div>
                  <PhaseTag phase={phase} />
                </div>
                {scale !== undefined && (
                  <p className="text-xs text-gray-400 mt-2">
                    缩放因子：1/√{currentInput.dk} = {scale.toFixed(4)}
                  </p>
                )}
              </div>

              {/* 因果掩码说明 */}
              {showMask && causalMask.length > 0 && (
                <CausalMaskGrid
                  label="因果掩码矩阵（Causal Mask）"
                  mask={causalMask}
                  tokenLabels={tokenLabels}
                  subtitle="✓ = 允许注意（当前/过去），✗ = 屏蔽（未来位置设为 -∞）"
                />
              )}

              {/* Q K V 矩阵 */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <MatrixGrid
                  label="Q（Query）"
                  matrix={Q}
                  highlightRow={phase === "dot-product" ? currentQueryIdx : undefined}
                  badge={`${Q.length}×${Q[0]?.length}`}
                  rowLabels={tokenLabels}
                />
                <MatrixGrid
                  label="K（Key）"
                  matrix={K}
                  badge={`${K.length}×${K[0]?.length}`}
                  rowLabels={tokenLabels}
                />
                <MatrixGrid
                  label="V（Value）"
                  matrix={V}
                  badge={`${V.length}×${V[0]?.length}`}
                  rowLabels={tokenLabels}
                />
              </div>

              {/* 原始分数矩阵 */}
              {showScores && !showMaskedScores && scaledScores.length > 0 && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <MaskedScoresGrid
                    label={phase === "scale" ? "缩放后分数（QKᵀ/√d_k）" : "原始点积分数（QKᵀ）"}
                    scaledScores={scaledScores.length > 0 ? scaledScores : scores}
                    maskedDisplay={scaledScores.length > 0
                      ? scaledScores.map((row) => row.map((v) => v))
                      : scores.map((row) => row.map((v) => v))}
                    causalMask={causalMask.length > 0 ? causalMask : Array.from({ length: Q.length }, (_, i) => Array.from({ length: Q.length }, (__, j) => j > i))}
                    tokenLabels={tokenLabels}
                    subtitle={phase === "scale" ? `已除以 √d_k（= ${scale?.toFixed(4) ?? "?"}），掩码尚未应用` : "未应用缩放和掩码"}
                    highlightRow={phase === "dot-product" ? currentQueryIdx : undefined}
                  />
                  <div className="bg-gray-50 rounded-lg border border-dashed border-gray-200 p-4 flex items-center justify-center">
                    <p className="text-xs text-gray-400">等待应用掩码…</p>
                  </div>
                </div>
              )}

              {/* 掩码后分数与注意力权重 */}
              {showMaskedScores && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <MaskedScoresGrid
                    label="掩码后分数（-∞ 表示已屏蔽）"
                    scaledScores={scaledScores}
                    maskedDisplay={maskedScoresDisplay}
                    causalMask={causalMask}
                    tokenLabels={tokenLabels}
                    subtitle="上三角已置为 -∞，Softmax 后对应权重为 0"
                  />
                  {showWeights ? (
                    <AttentionHeatmap
                      label="注意力权重（因果 Softmax）"
                      weights={attentionWeights}
                      causalMask={causalMask}
                      tokenLabels={tokenLabels}
                      subtitle="每行只在允许位置上分配权重，上三角权重恒为 0"
                    />
                  ) : (
                    <div className="bg-gray-50 rounded-lg border border-dashed border-gray-200 p-4 flex items-center justify-center">
                      <p className="text-xs text-gray-400">等待 Softmax 步骤…</p>
                    </div>
                  )}
                </div>
              )}

              {/* 输出矩阵 */}
              {showOutput && (
                <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <h4 className="text-sm font-semibold text-gray-800">
                        输出矩阵（因果注意力权重 · V）
                      </h4>
                      <p className="text-xs text-gray-500 mt-0.5">
                        每个位置的输出仅融合了当前及之前 Token 的信息
                      </p>
                    </div>
                    <span className="text-xs px-2 py-0.5 bg-emerald-50 text-emerald-700 rounded-full">
                      {output.length}×{output[0]?.length}
                    </span>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="text-xs border-collapse">
                      <tbody>
                        {output.map((row, i) => (
                          <tr key={i}>
                            <td className="pr-2 text-gray-400 font-mono text-right w-10 truncate">
                              {tokenLabels[i]}
                            </td>
                            {row.map((val, j) => (
                              <td
                                key={j}
                                className="w-16 h-10 text-center font-mono border border-gray-100 rounded bg-emerald-50 text-emerald-800"
                              >
                                {fmt(val, 3)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* 计算流程进度条 */}
              <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
                <h4 className="text-sm font-semibold text-gray-800 mb-3">计算流程</h4>
                <div className="flex items-center flex-wrap gap-2 text-xs">
                  {phaseSteps.map((step, idx, arr) => {
                    const currentIdx = arr.findIndex((s) => s.id === phase);
                    const isDone = currentIdx >= idx;
                    return (
                      <div key={step.id} className="flex items-center gap-2">
                        <span
                          className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
                            step.id === phase
                              ? "bg-blue-600 text-white shadow-sm"
                              : isDone
                              ? "bg-blue-100 text-blue-700"
                              : "bg-gray-100 text-gray-400"
                          }`}
                        >
                          {step.label}
                        </span>
                        {idx < arr.length - 1 && (
                          <span className="text-gray-300">→</span>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          );
        },
      }}
    />
  );
}

export default CausalAttentionVisualizer;
