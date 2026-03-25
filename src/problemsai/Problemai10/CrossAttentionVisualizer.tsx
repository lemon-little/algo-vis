import { ConfigurableVisualizer } from "@/components/visualizers/ConfigurableVisualizer";
import { CoreIdeaBox } from "@/components/visualizers/CoreIdeaBox";
import { ProblemInput } from "@/types/visualization";
import { generateCrossAttentionSteps } from "./algorithm";
import { getAiProblemCoreIdea } from "@/config/aiProblemCoreIdeas";
import { InlineMath } from "react-katex";
import "katex/dist/katex.min.css";

interface CrossAttentionInput extends ProblemInput {
  Q: string;
  K: string;
  V: string;
  dk: number;
}

const DEFAULT_Q = "[[1,0,1],[0,1,1]]";
const DEFAULT_K = "[[1,0,0],[0,1,0],[0,0,1],[1,1,0]]";
const DEFAULT_V = "[[1,0],[0,1],[1,1],[0,0]]";
const DEFAULT_DK = 3;

function parseMatrix(raw: string | number[][], fallback: string): number[][] {
  if (Array.isArray(raw)) return raw as number[][];
  try {
    const parsed = JSON.parse(raw);
    if (
      Array.isArray(parsed) &&
      parsed.every(
        (row: unknown) =>
          Array.isArray(row) && row.every((v) => typeof v === "number")
      )
    ) {
      return parsed as number[][];
    }
  } catch {
    // fall through
  }
  return JSON.parse(fallback) as number[][];
}

function fmt(v?: number, digits = 3): string {
  if (v === undefined || isNaN(v)) return "--";
  return v.toFixed(digits);
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
  colorFn?: (row: number, col: number, val: number) => string;
  badge?: string;
  rowPrefix?: string;
}

function MatrixGrid({
  label,
  matrix,
  highlightRow,
  colorFn,
  badge,
  rowPrefix = "r",
}: MatrixGridProps) {
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
                <td className="pr-2 text-gray-400 font-mono">
                  {rowPrefix}
                  {i}
                </td>
                {row.map((val, j) => {
                  const bg = colorFn ? colorFn(i, j, val) : undefined;
                  const isHl = highlightRow === i;
                  return (
                    <td
                      key={j}
                      className={`w-14 h-9 text-center font-mono border border-gray-100 rounded transition-all ${
                        isHl ? "ring-2 ring-blue-400 font-semibold" : ""
                      }`}
                      style={{
                        backgroundColor:
                          bg ?? (isHl ? "#eff6ff" : "#f8fafc"),
                      }}
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

interface HeatmapGridProps {
  label: string;
  matrix: number[][];
  subtitle?: string;
  rowLabel?: string;
  colLabel?: string;
}

function HeatmapGrid({
  label,
  matrix,
  subtitle,
  rowLabel = "tgt",
  colLabel = "src",
}: HeatmapGridProps) {
  if (!matrix || matrix.length === 0) return null;
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
              <td />
              {matrix[0]?.map((_, j) => (
                <td
                  key={j}
                  className="text-center text-gray-400 font-mono pb-1"
                >
                  {colLabel}
                  {j}
                </td>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={i}>
                <td className="pr-2 text-gray-400 font-mono text-right w-8">
                  {rowLabel}
                  {i}
                </td>
                {row.map((val, j) => {
                  const bg = getHeatColor(val);
                  const tc = getTextColor(val);
                  return (
                    <td
                      key={j}
                      className="w-14 h-10 text-center font-mono border border-white/60 rounded transition-all"
                      style={{ backgroundColor: bg, color: tc }}
                      title={`${rowLabel}${i} → ${colLabel}${j}: ${fmt(val)}`}
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
      <div className="mt-2 flex items-center gap-2 text-[10px] text-gray-400">
        <span
          className="inline-block w-4 h-2 rounded"
          style={{ background: getHeatColor(0) }}
        />
        <span>低</span>
        <span
          className="inline-block w-4 h-2 rounded"
          style={{ background: getHeatColor(0.5) }}
        />
        <span
          className="inline-block w-4 h-2 rounded"
          style={{ background: getHeatColor(1) }}
        />
        <span>高</span>
      </div>
    </div>
  );
}

function PhaseTag({ phase }: { phase: string }) {
  const map: Record<string, { label: string; color: string }> = {
    init: { label: "初始化", color: "bg-gray-100 text-gray-700" },
    "dot-product": {
      label: "点积 QKᵀ",
      color: "bg-blue-100 text-blue-700",
    },
    scale: { label: "缩放 /√d_k", color: "bg-violet-100 text-violet-700" },
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

function CrossAttentionVisualizer() {
  const coreIdea = getAiProblemCoreIdea(10010);

  return (
    <ConfigurableVisualizer<CrossAttentionInput, Record<string, never>>
      config={{
        defaultInput: {
          Q: DEFAULT_Q,
          K: DEFAULT_K,
          V: DEFAULT_V,
          dk: DEFAULT_DK,
        },
        algorithm: (input) => {
          const Q = parseMatrix(input.Q as string | number[][], DEFAULT_Q);
          const K = parseMatrix(input.K as string | number[][], DEFAULT_K);
          const V = parseMatrix(input.V as string | number[][], DEFAULT_V);
          const dk =
            typeof input.dk === "number"
              ? input.dk
              : parseFloat(String(input.dk)) || DEFAULT_DK;
          return generateCrossAttentionSteps(Q, K, V, dk);
        },
        inputTypes: [
          { type: "string", key: "Q", label: "Q 矩阵 Decoder（JSON）" },
          { type: "string", key: "K", label: "K 矩阵 Encoder（JSON）" },
          { type: "string", key: "V", label: "V 矩阵 Encoder（JSON）" },
          { type: "number", key: "dk", label: "d_k（Key 维度）", min: 1 },
        ],
        inputFields: [
          {
            type: "string",
            key: "Q",
            label: "Q 矩阵 Decoder（JSON）",
            placeholder: DEFAULT_Q,
          },
          {
            type: "string",
            key: "K",
            label: "K 矩阵 Encoder（JSON）",
            placeholder: DEFAULT_K,
          },
          {
            type: "string",
            key: "V",
            label: "V 矩阵 Encoder（JSON）",
            placeholder: DEFAULT_V,
          },
          {
            type: "number",
            key: "dk",
            label: "d_k（Key 维度）",
            placeholder: String(DEFAULT_DK),
          },
        ],
        testCases: [
          {
            label: "示例（tgt=2, src=4）",
            value: {
              Q: DEFAULT_Q,
              K: DEFAULT_K,
              V: DEFAULT_V,
              dk: DEFAULT_DK,
            },
          },
          {
            label: "翻译场景（tgt=3, src=3）",
            value: {
              Q: "[[1,1],[0,1],[1,0]]",
              K: "[[1,0],[0,1],[1,1]]",
              V: "[[1,0,1],[0,1,0],[1,1,1]]",
              dk: 2,
            },
          },
          {
            label: "长源序列（tgt=2, src=5）",
            value: {
              Q: "[[1,0,1,0],[0,1,0,1]]",
              K: "[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,1,0,0]]",
              V: "[[1,0],[0,1],[1,1],[0,0],[1,0]]",
              dk: 4,
            },
          },
        ],
        render: ({ visualization, variables }) => {
          const currentInput =
            visualization.input as CrossAttentionInput;
          const Q = parseMatrix(
            currentInput.Q as string | number[][],
            DEFAULT_Q
          );
          const K = parseMatrix(
            currentInput.K as string | number[][],
            DEFAULT_K
          );
          const V = parseMatrix(
            currentInput.V as string | number[][],
            DEFAULT_V
          );

          const phase = (variables?.phase as string) ?? "init";
          const currentQueryIdx = variables?.currentQueryIdx as
            | number
            | undefined;
          const scores = (variables?.scores as number[][] | undefined) ?? [];
          const scaledScores =
            (variables?.scaledScores as number[][] | undefined) ?? [];
          const attentionWeights =
            (variables?.attentionWeights as number[][] | undefined) ?? [];
          const output =
            (variables?.output as number[][] | undefined) ?? [];
          const scale = variables?.scale as number | undefined;

          const showScores = [
            "dot-product",
            "scale",
            "softmax",
            "output",
            "complete",
          ].includes(phase);
          const showScaled = [
            "scale",
            "softmax",
            "output",
            "complete",
          ].includes(phase);
          const showWeights = ["softmax", "output", "complete"].includes(phase);
          const showOutput = ["output", "complete"].includes(phase);

          return (
            <div className="space-y-4">
              {coreIdea && <CoreIdeaBox {...coreIdea} />}

              {/* 标题与公式 */}
              <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <div>
                    <h3 className="text-base font-semibold text-gray-900">
                      交叉注意力（Cross-Attention）
                    </h3>
                    <p className="text-xs text-gray-500 mt-1 font-mono">
                      <InlineMath math="\text{CrossAttention}(Q,K,V)=\text{softmax}\!\left(\dfrac{QK^\top}{\sqrt{d_k}}\right)V" />
                    </p>
                  </div>
                  <PhaseTag phase={phase} />
                </div>
                {scale !== undefined && (
                  <p className="text-xs text-gray-400 mt-2">
                    缩放因子：
                    <InlineMath math={`\\frac{1}{\\sqrt{d_k}} = \\frac{1}{\\sqrt{${currentInput.dk}}} \\approx ${scale.toFixed(4)}`} />
                  </p>
                )}
              </div>

              {/* 架构说明：Q 来自 Decoder，K/V 来自 Encoder */}
              <div className="bg-amber-50 rounded-lg border border-amber-200 p-4">
                <h4 className="text-xs font-semibold text-amber-800 mb-2">
                  交叉注意力 vs 自注意力
                </h4>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs">
                  <div className="bg-white rounded border border-amber-100 p-2 text-center">
                    <div className="font-semibold text-blue-700">Q（Query）</div>
                    <div className="text-gray-500">来自 Decoder</div>
                    <div className="text-gray-400 font-mono mt-0.5">
                      {Q.length}×{Q[0]?.length} &nbsp;→ 目标序列
                    </div>
                  </div>
                  <div className="bg-white rounded border border-amber-100 p-2 text-center">
                    <div className="font-semibold text-violet-700">K（Key）</div>
                    <div className="text-gray-500">来自 Encoder</div>
                    <div className="text-gray-400 font-mono mt-0.5">
                      {K.length}×{K[0]?.length} &nbsp;→ 源序列
                    </div>
                  </div>
                  <div className="bg-white rounded border border-amber-100 p-2 text-center">
                    <div className="font-semibold text-emerald-700">
                      V（Value）
                    </div>
                    <div className="text-gray-500">来自 Encoder</div>
                    <div className="text-gray-400 font-mono mt-0.5">
                      {V.length}×{V[0]?.length} &nbsp;→ 源序列
                    </div>
                  </div>
                </div>
              </div>

              {/* Q / K / V 矩阵 */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <MatrixGrid
                  label="Q（Decoder Query）"
                  matrix={Q}
                  highlightRow={
                    phase === "dot-product" ? currentQueryIdx : undefined
                  }
                  badge={`${Q.length}×${Q[0]?.length}`}
                  rowPrefix="tgt"
                />
                <MatrixGrid
                  label="K（Encoder Key）"
                  matrix={K}
                  badge={`${K.length}×${K[0]?.length}`}
                  rowPrefix="src"
                />
                <MatrixGrid
                  label="V（Encoder Value）"
                  matrix={V}
                  badge={`${V.length}×${V[0]?.length}`}
                  rowPrefix="src"
                />
              </div>

              {/* 分数矩阵与注意力权重热力图 */}
              {showScores && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <HeatmapGrid
                    label="注意力分数（QKᵀ）"
                    matrix={showScaled ? scaledScores : scores}
                    subtitle={
                      showScaled
                        ? `已除以 √d_k（≈ ${scale?.toFixed(4) ?? "?"}）`
                        : "原始点积分数（Decoder × Encoder）"
                    }
                    rowLabel="tgt"
                    colLabel="src"
                  />
                  {showWeights ? (
                    <HeatmapGrid
                      label="注意力权重（Softmax）"
                      matrix={attentionWeights}
                      subtitle="每行之和为 1，表示 Decoder 对 Encoder 各位置的关注度"
                      rowLabel="tgt"
                      colLabel="src"
                    />
                  ) : (
                    <div className="bg-gray-50 rounded-lg border border-dashed border-gray-200 p-4 flex items-center justify-center">
                      <p className="text-xs text-gray-400">
                        等待 Softmax 步骤…
                      </p>
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
                        输出矩阵（attention weights · V）
                      </h4>
                      <p className="text-xs text-gray-500 mt-0.5">
                        Decoder 每个位置融合了 Encoder 的全局上下文，
                        <InlineMath math="\text{shape} = (\text{tgtLen},\, d_v)" />
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
                            <td className="pr-2 text-gray-400 font-mono text-right w-10">
                              tgt{i}
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

              {/* 计算流程 */}
              <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
                <h4 className="text-sm font-semibold text-gray-800 mb-3">
                  计算流程
                </h4>
                <div className="flex items-center flex-wrap gap-2 text-xs">
                  {[
                    { id: "init", label: "① 输入 Q/K/V" },
                    { id: "dot-product", label: "② QKᵀ 点积" },
                    { id: "scale", label: "③ /√d_k 缩放" },
                    { id: "softmax", label: "④ Softmax" },
                    { id: "output", label: "⑤ × V 加权" },
                    { id: "complete", label: "⑥ 输出" },
                  ].map((step, idx, arr) => {
                    const isDone =
                      arr.findIndex((s) => s.id === phase) >= idx;
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

export default CrossAttentionVisualizer;
