import { InlineMath, BlockMath } from "react-katex";
import "katex/dist/katex.min.css";
import { ConfigurableVisualizer } from "@/components/visualizers/ConfigurableVisualizer";
import { CoreIdeaBox } from "@/components/visualizers/CoreIdeaBox";
import { ProblemInput } from "@/types/visualization";
import { generateFFNSteps } from "./algorithm";
import { getAiProblemCoreIdea } from "@/config/aiProblemCoreIdeas";

interface FFNInput extends ProblemInput {
  x: string;
  W1: string;
  b1: string;
  W2: string;
  b2: string;
}

const DEFAULT_X = "[1.0, 0.5, -0.3]";
const DEFAULT_W1 =
  "[[0.5,-0.3,0.8],[0.2,0.7,-0.4],[0.9,-0.1,0.3],[-0.6,0.4,0.7],[0.1,-0.8,0.5],[0.3,0.6,-0.2]]";
const DEFAULT_B1 = "[0.1, -0.1, 0.2, 0.0, -0.2, 0.1]";
const DEFAULT_W2 =
  "[[0.4,0.3,-0.5,0.7,-0.1,0.2],[-0.3,0.8,0.1,-0.4,0.6,-0.7],[0.5,-0.2,0.9,0.3,-0.8,0.4]]";
const DEFAULT_B2 = "[0.0, 0.1, -0.1]";

function parseVec(raw: string | number[]): number[] {
  if (Array.isArray(raw)) return raw as number[];
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed) && parsed.every((v: unknown) => typeof v === "number")) {
      return parsed as number[];
    }
  } catch { /* fall through */ }
  return JSON.parse(DEFAULT_X) as number[];
}

function parseMatrix(raw: string | number[][], fallback: string): number[][] {
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
  } catch { /* fall through */ }
  return JSON.parse(fallback) as number[][];
}

function fmt(v?: number | null, digits = 3): string {
  if (v === undefined || v === null || isNaN(v as number)) return "--";
  const num = v as number;
  return (num >= 0 ? "" : "") + num.toFixed(digits);
}

function vecColor(val: number, maxAbs: number): string {
  if (val === 0) return "hsl(0,0%,94%)";
  const t = Math.min(Math.abs(val) / (maxAbs || 1), 1);
  return val > 0
    ? `hsl(220,70%,${90 - t * 35}%)`
    : `hsl(0,60%,${90 - t * 25}%)`;
}

function vecTextColor(val: number, maxAbs: number): string {
  const t = Math.min(Math.abs(val) / (maxAbs || 1), 1);
  return t > 0.6 ? "#fff" : "#1e293b";
}

// ── 小型向量显示 ───────────────────────────────────────────────────────
interface VecChipsProps {
  values: number[];
  highlightIdx?: number;
  maxShow?: number;
  chipSize?: "sm" | "md";
}
function VecChips({ values, highlightIdx, maxShow = 8, chipSize = "sm" }: VecChipsProps) {
  const maxAbs = Math.max(...values.map(Math.abs), 0.001);
  const display = values.slice(0, maxShow);
  const extra = values.length - maxShow;
  const pad = chipSize === "md" ? "px-2.5 py-2" : "px-2 py-1.5";
  const textSz = chipSize === "md" ? "text-xs" : "text-[11px]";
  return (
    <div className="flex flex-wrap gap-1">
      {display.map((v, i) => {
        const isHl = highlightIdx === i;
        const bg = isHl ? "#2563eb" : vecColor(v, maxAbs);
        const tc = isHl ? "#fff" : vecTextColor(v, maxAbs);
        return (
          <div
            key={i}
            className={`rounded font-mono transition-all ${pad} ${textSz} flex flex-col items-center ${
              isHl ? "ring-2 ring-blue-300 scale-110 shadow" : ""
            }`}
            style={{ backgroundColor: bg, color: tc, minWidth: chipSize === "md" ? 48 : 40 }}
          >
            <span className="text-[8px] opacity-50">[{i}]</span>
            <span>{fmt(v, 2)}</span>
          </div>
        );
      })}
      {extra > 0 && (
        <div className="rounded bg-gray-100 text-gray-400 text-[10px] flex items-center px-2">
          +{extra}
        </div>
      )}
    </div>
  );
}

// ── 向量卡片（含标签 + 维度 badge）─────────────────────────────────────
interface VecCardProps {
  label: string | React.ReactNode;
  values: number[];
  badge?: string;
  badgeColor?: string;
  subtitle?: string;
  highlightIdx?: number;
  borderColor?: string;
}
function VecCard({ label, values, badge, badgeColor = "bg-blue-50 text-blue-700", subtitle, highlightIdx, borderColor }: VecCardProps) {
  if (!values || values.length === 0) return null;
  return (
    <div
      className="bg-white rounded-lg border p-4 shadow-sm"
      style={{ borderColor: borderColor ?? "#e5e7eb" }}
    >
      <div className="flex items-center justify-between mb-2">
        <div>
          <div className="text-sm font-semibold text-gray-800">{label}</div>
          {subtitle && <p className="text-xs text-gray-400 mt-0.5">{subtitle}</p>}
        </div>
        {badge && (
          <span className={`text-xs px-2 py-0.5 rounded-full ${badgeColor}`}>{badge}</span>
        )}
      </div>
      <VecChips values={values} highlightIdx={highlightIdx} />
    </div>
  );
}

// ── 步骤说明面板：核心新增 ──────────────────────────────────────────────
interface StepExplainerProps {
  phase: string;
  currentNeuron?: number;
  x: number[];
  W1: number[][];
  b1: number[];
  W2: number[][];
  b2: number[];
  linear1: number[];
  z1: number[];
  h: number[];
  linear2: number[];
  output: number[];
  dModel: number;
  dFF: number;
  dOut: number;
}

function DotProductBreakdown({
  rowVec,
  colVec,
  result,
  rowLabel,
  colLabel,
}: {
  rowVec: number[];
  colVec: number[];
  result: number;
  rowLabel: string;
  colLabel: string;
}) {
  const maxTerms = 4;
  const showAll = rowVec.length <= maxTerms;
  const terms = showAll ? rowVec : rowVec.slice(0, maxTerms);
  return (
    <div className="bg-gray-50 rounded-lg p-3 text-xs font-mono space-y-1.5">
      <div className="text-gray-500 mb-1">
        <InlineMath math={`${rowLabel} \\cdot ${colLabel}`} />
        {" "}= {terms.map((w, i) => (
          <span key={i}>
            {i > 0 && <span className="text-gray-400"> + </span>}
            <span
              className={`px-1 rounded ${w >= 0 ? "bg-blue-50 text-blue-700" : "bg-rose-50 text-rose-700"}`}
            >
              {fmt(w, 2)}
            </span>
            <span className="text-gray-400">×</span>
            <span
              className={`px-1 rounded ${(colVec[i] ?? 0) >= 0 ? "bg-blue-50 text-blue-700" : "bg-rose-50 text-rose-700"}`}
            >
              {fmt(colVec[i] ?? 0, 2)}
            </span>
          </span>
        ))}
        {!showAll && <span className="text-gray-400"> + …</span>}
      </div>
      <div className="flex items-center gap-2 pt-1 border-t border-gray-200">
        <span className="text-gray-400">= </span>
        <span className="text-base font-bold text-blue-700">{fmt(result, 4)}</span>
      </div>
    </div>
  );
}

function StepExplainer({
  phase, currentNeuron,
  x, W1, b1, W2, b2,
  linear1, z1, h, linear2, output,
  dModel, dFF, dOut,
}: StepExplainerProps) {
  // Phase → border color
  const phaseColor: Record<string, string> = {
    init: "#e5e7eb",
    linear1: "#93c5fd",
    bias1: "#c4b5fd",
    relu: "#fcd34d",
    linear2: "#67e8f9",
    bias2: "#5eead4",
    complete: "#6ee7b7",
  };
  const border = phaseColor[phase] ?? "#e5e7eb";

  // ── init ─────────────────────────────────────────────────────────────
  if (phase === "init") {
    return (
      <div className="bg-white rounded-lg border-2 p-4 shadow-sm space-y-3" style={{ borderColor: border }}>
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold px-2 py-0.5 bg-gray-100 text-gray-700 rounded-full">初始化</span>
          <span className="text-sm font-semibold text-gray-800">接收 Attention 层输出 x，准备前向传播</span>
        </div>
        <div className="grid grid-cols-3 gap-3 text-xs text-center">
          <div className="bg-blue-50 rounded-lg p-3">
            <div className="text-blue-500 font-mono mb-1"><InlineMath math={"\\mathbf{x} \\in \\mathbb{R}^{d_{model}}"} /></div>
            <div className="text-gray-600">输入向量</div>
            <div className="text-blue-700 font-bold mt-1">d = {dModel}</div>
          </div>
          <div className="bg-violet-50 rounded-lg p-3">
            <div className="text-violet-500 font-mono mb-1"><InlineMath math={"\\xrightarrow{\\text{FFN}}"} /></div>
            <div className="text-gray-600">两层全连接 + ReLU</div>
            <div className="text-violet-700 font-bold mt-1">d_ff = {dFF}</div>
          </div>
          <div className="bg-emerald-50 rounded-lg p-3">
            <div className="text-emerald-500 font-mono mb-1"><InlineMath math={"\\mathbf{y} \\in \\mathbb{R}^{d_{model}}"} /></div>
            <div className="text-gray-600">输出向量</div>
            <div className="text-emerald-700 font-bold mt-1">d = {dOut}</div>
          </div>
        </div>
        <VecCard
          label={<><InlineMath math={"\\mathbf{x}"} />（输入）</>}
          values={x}
          badge={`d_model=${dModel}`}
        />
      </div>
    );
  }

  // ── linear1: W1[i]·x ──────────────────────────────────────────────────
  if (phase === "linear1" && currentNeuron !== undefined) {
    const i = currentNeuron;
    const w1row = W1[i] ?? [];
    const partialResult = linear1[i] ?? 0;
    return (
      <div className="bg-white rounded-lg border-2 p-4 shadow-sm space-y-3" style={{ borderColor: border }}>
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold px-2 py-0.5 bg-blue-100 text-blue-700 rounded-full">第一层线性变换</span>
          <span className="text-sm font-semibold text-gray-800">
            计算第 {i} 个中间层神经元的加权和
          </span>
        </div>
        <div className="text-xs text-gray-500">
          <InlineMath math={`\\text{linear}_1[${i}] = W_1[${i},:] \\cdot \\mathbf{x}`} />
        </div>
        {/* 三列：输入 → 操作 → 输出 */}
        <div className="grid grid-cols-[1fr_auto_1fr] gap-3 items-start">
          {/* 输入 */}
          <div className="space-y-2">
            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide">输入</div>
            <div className="bg-blue-50 rounded-lg p-2 space-y-1">
              <div className="text-[10px] text-blue-500 font-mono">
                <InlineMath math={"\\mathbf{x}"} /> (d_model={dModel})
              </div>
              <VecChips values={x} />
            </div>
            <div className="bg-blue-50 rounded-lg p-2 space-y-1">
              <div className="text-[10px] text-blue-500 font-mono">
                <InlineMath math={`W_1[${i},:]`} /> (第{i}行权重)
              </div>
              <VecChips values={w1row} highlightIdx={undefined} />
            </div>
          </div>

          {/* 箭头 */}
          <div className="flex items-center pt-8 text-blue-300 text-xl">→</div>

          {/* 输出 */}
          <div className="space-y-2">
            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
              本步计算结果
            </div>
            <DotProductBreakdown
              rowVec={w1row}
              colVec={x}
              result={partialResult}
              rowLabel={`W_1[${i},:]`}
              colLabel={"\\mathbf{x}"}
            />
            {linear1.length > 0 && (
              <div className="bg-blue-50 rounded-lg p-2 space-y-1">
                <div className="text-[10px] text-blue-500">已计算 {linear1.length}/{dFF} 个神经元</div>
                <VecChips values={[...linear1, ...Array(dFF - linear1.length).fill(NaN)]} highlightIdx={i} />
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // ── bias1: W1·x + b1 ─────────────────────────────────────────────────
  if (phase === "bias1") {
    return (
      <div className="bg-white rounded-lg border-2 p-4 shadow-sm space-y-3" style={{ borderColor: border }}>
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold px-2 py-0.5 bg-violet-100 text-violet-700 rounded-full">加偏置 b₁</span>
          <span className="text-sm font-semibold text-gray-800">
            每个神经元加上各自的偏置项，得到第一层完整输出
          </span>
        </div>
        <div className="text-xs text-gray-500">
          <InlineMath math={"\\mathbf{z}_1 = W_1 \\mathbf{x} + \\mathbf{b}_1"} />
        </div>
        <div className="grid grid-cols-[1fr_auto_1fr] gap-3 items-start">
          <div className="space-y-2">
            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide">输入</div>
            <VecCard
              label={<><InlineMath math={"W_1 \\mathbf{x}"} />（矩阵乘结果）</>}
              values={linear1}
              badge={`d_ff=${dFF}`}
              badgeColor="bg-blue-50 text-blue-700"
            />
            <VecCard
              label={<><InlineMath math={"\\mathbf{b}_1"} />（偏置向量）</>}
              values={b1}
              badgeColor="bg-violet-50 text-violet-700"
            />
          </div>
          <div className="flex items-center pt-16 text-violet-300 text-xl">→</div>
          <div className="space-y-2">
            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide">输出</div>
            <VecCard
              label={<><InlineMath math={"\\mathbf{z}_1"} />（第一层输出，含正负值）</>}
              values={z1}
              badge={`d_ff=${dFF}`}
              badgeColor="bg-violet-50 text-violet-700"
              borderColor="#c4b5fd"
            />
            <div className="text-xs text-gray-400 bg-violet-50 rounded p-2">
              正值将在 ReLU 后激活，负值将被截断为 0。
              激活数量预测：{z1.filter((v) => v > 0).length}/{dFF}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ── relu: h[i] = max(0, z1[i]) ──────────────────────────────────────
  if (phase === "relu" && currentNeuron !== undefined) {
    const i = currentNeuron;
    const z1Val = z1[i] ?? 0;
    const hVal = h[i] ?? 0;
    const isActive = hVal > 0;
    return (
      <div className="bg-white rounded-lg border-2 p-4 shadow-sm space-y-3" style={{ borderColor: border }}>
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold px-2 py-0.5 bg-amber-100 text-amber-700 rounded-full">ReLU 激活</span>
          <span className="text-sm font-semibold text-gray-800">
            对第 {i} 个神经元应用 max(0,·) 非线性激活
          </span>
        </div>
        <div className="text-xs text-gray-500">
          <InlineMath math={`h[${i}] = \\text{ReLU}(z_1[${i}]) = \\max(0,\\, z_1[${i}])`} />
        </div>
        <div className="grid grid-cols-[1fr_auto_1fr] gap-3 items-start">
          {/* 输入 */}
          <div className="space-y-2">
            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide">输入</div>
            <div className="bg-amber-50 rounded-lg p-3">
              <div className="text-[10px] text-amber-600 mb-1">
                <InlineMath math={`z_1[${i}]`} />（第一层输出第{i}位）
              </div>
              <span
                className={`text-lg font-bold font-mono ${z1Val >= 0 ? "text-blue-700" : "text-rose-600"}`}
              >
                {fmt(z1Val, 4)}
              </span>
              <div className="text-xs text-gray-400 mt-1">
                {z1Val >= 0 ? "正值 → 将被保留" : "负值 → 将被截断为 0"}
              </div>
            </div>
          </div>

          {/* 操作 */}
          <div className="flex flex-col items-center pt-6 gap-1">
            <span className={`text-xs px-2 py-1 rounded-full font-semibold ${isActive ? "bg-blue-100 text-blue-700" : "bg-gray-100 text-gray-500"}`}>
              max(0, ·)
            </span>
            <span className="text-amber-300 text-xl">→</span>
          </div>

          {/* 输出 */}
          <div className="space-y-2">
            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide">输出</div>
            <div
              className="rounded-lg p-3"
              style={{ backgroundColor: isActive ? "#eff6ff" : "#f1f5f9" }}
            >
              <div className="text-[10px] mb-1" style={{ color: isActive ? "#2563eb" : "#64748b" }}>
                <InlineMath math={`h[${i}]`} />
              </div>
              <span
                className="text-lg font-bold font-mono"
                style={{ color: isActive ? "#1d4ed8" : "#94a3b8" }}
              >
                {fmt(hVal, 4)}
              </span>
              <div className="text-xs mt-1" style={{ color: isActive ? "#3b82f6" : "#94a3b8" }}>
                {isActive ? `✓ 激活（保留 ${fmt(z1Val, 4)}）` : "✗ 抑制（截断为 0）"}
              </div>
            </div>

            {h.length > 0 && (
              <div className="bg-amber-50 rounded-lg p-2 space-y-1">
                <div className="text-[10px] text-amber-600">
                  已激活 {h.filter((v) => v > 0).length}/{h.length} 个神经元
                </div>
                <VecChips
                  values={[...h, ...Array(dFF - h.length).fill(NaN)]}
                  highlightIdx={i}
                />
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // ── linear2: W2[i]·h ─────────────────────────────────────────────────
  if (phase === "linear2" && currentNeuron !== undefined) {
    const i = currentNeuron;
    const w2row = W2[i] ?? [];
    const partialResult = linear2[i] ?? 0;
    return (
      <div className="bg-white rounded-lg border-2 p-4 shadow-sm space-y-3" style={{ borderColor: border }}>
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold px-2 py-0.5 bg-cyan-100 text-cyan-700 rounded-full">第二层线性变换</span>
          <span className="text-sm font-semibold text-gray-800">
            将激活后的 h 压缩回输出维度，计算第 {i} 个输出
          </span>
        </div>
        <div className="text-xs text-gray-500">
          <InlineMath math={`\\text{linear}_2[${i}] = W_2[${i},:] \\cdot \\mathbf{h}`} />
        </div>
        <div className="grid grid-cols-[1fr_auto_1fr] gap-3 items-start">
          <div className="space-y-2">
            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide">输入</div>
            <div className="bg-cyan-50 rounded-lg p-2 space-y-1">
              <div className="text-[10px] text-cyan-600">
                <InlineMath math={"\\mathbf{h}"} /> — ReLU 输出 (d_ff={dFF})
              </div>
              <VecChips values={h} />
            </div>
            <div className="bg-cyan-50 rounded-lg p-2 space-y-1">
              <div className="text-[10px] text-cyan-600">
                <InlineMath math={`W_2[${i},:]`} /> 第{i}行权重
              </div>
              <VecChips values={w2row} />
            </div>
          </div>
          <div className="flex items-center pt-8 text-cyan-300 text-xl">→</div>
          <div className="space-y-2">
            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide">本步计算结果</div>
            <DotProductBreakdown
              rowVec={w2row}
              colVec={h}
              result={partialResult}
              rowLabel={`W_2[${i},:]`}
              colLabel={"\\mathbf{h}"}
            />
            {linear2.length > 0 && (
              <div className="bg-cyan-50 rounded-lg p-2 space-y-1">
                <div className="text-[10px] text-cyan-600">已计算 {linear2.length}/{dOut} 个输出</div>
                <VecChips values={[...linear2, ...Array(dOut - linear2.length).fill(NaN)]} highlightIdx={i} />
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // ── bias2: W2·h + b2 ─────────────────────────────────────────────────
  if (phase === "bias2") {
    return (
      <div className="bg-white rounded-lg border-2 p-4 shadow-sm space-y-3" style={{ borderColor: border }}>
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold px-2 py-0.5 bg-teal-100 text-teal-700 rounded-full">加偏置 b₂</span>
          <span className="text-sm font-semibold text-gray-800">
            加上第二层偏置，得到 FFN 最终输出
          </span>
        </div>
        <div className="text-xs text-gray-500">
          <InlineMath math={"\\mathbf{y} = W_2 \\mathbf{h} + \\mathbf{b}_2"} />
        </div>
        <div className="grid grid-cols-[1fr_auto_1fr] gap-3 items-start">
          <div className="space-y-2">
            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide">输入</div>
            <VecCard
              label={<><InlineMath math={"W_2 \\mathbf{h}"} />（矩阵乘结果）</>}
              values={linear2}
              badge={`d_out=${dOut}`}
              badgeColor="bg-cyan-50 text-cyan-700"
            />
            <VecCard
              label={<><InlineMath math={"\\mathbf{b}_2"} />（偏置）</>}
              values={b2}
              badgeColor="bg-teal-50 text-teal-700"
            />
          </div>
          <div className="flex items-center pt-16 text-teal-300 text-xl">→</div>
          <div className="space-y-2">
            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide">输出</div>
            <VecCard
              label={<><InlineMath math={"\\mathbf{y}"} />（FFN 最终输出）</>}
              values={output}
              badge={`d_out=${dOut}`}
              badgeColor="bg-emerald-50 text-emerald-700"
              borderColor="#6ee7b7"
            />
            <div className="text-xs text-gray-400 bg-teal-50 rounded p-2">
              输出维度与输入一致（d_model={dOut}），后接残差连接 <InlineMath math={"\\mathbf{x} + \\text{FFN}(\\mathbf{x})"} />
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ── complete ─────────────────────────────────────────────────────────
  if (phase === "complete") {
    return (
      <div className="bg-white rounded-lg border-2 p-4 shadow-sm space-y-3" style={{ borderColor: border }}>
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold px-2 py-0.5 bg-emerald-100 text-emerald-700 rounded-full">✓ 计算完成</span>
          <span className="text-sm font-semibold text-gray-800">
            前向传播：<InlineMath math={"\\mathbf{x} \\to z_1 \\to \\mathbf{h} \\to \\mathbf{y}"} />
          </span>
        </div>
        <div className="grid grid-cols-4 gap-2 text-xs">
          {[
            { label: "输入 x", values: x, color: "#eff6ff", border: "#93c5fd", badge: `d=${dModel}` },
            { label: "z₁ = W₁x+b₁", values: z1, color: "#f5f3ff", border: "#c4b5fd", badge: `d=${dFF}` },
            { label: "h = ReLU(z₁)", values: h, color: "#fffbeb", border: "#fcd34d", badge: `激活${h.filter(v=>v>0).length}/${dFF}` },
            { label: "输出 y", values: output, color: "#ecfdf5", border: "#6ee7b7", badge: `d=${dOut}` },
          ].map(({ label, values, color, border: bc, badge }) => (
            <div key={label} className="rounded-lg p-2.5 space-y-1.5" style={{ backgroundColor: color, border: `1.5px solid ${bc}` }}>
              <div className="font-semibold text-gray-700 text-[11px]">{label}</div>
              <div className="text-[9px] text-gray-400">{badge}</div>
              <VecChips values={values} maxShow={4} />
            </div>
          ))}
        </div>
      </div>
    );
  }

  return null;
}

// ── Phase Tag ────────────────────────────────────────────────────────────
function PhaseTag({ phase }: { phase: string }) {
  const map: Record<string, { label: string; color: string }> = {
    init:     { label: "初始化",       color: "bg-gray-100 text-gray-700" },
    linear1:  { label: "W₁·x 矩阵乘", color: "bg-blue-100 text-blue-700" },
    bias1:    { label: "+b₁ 加偏置",   color: "bg-violet-100 text-violet-700" },
    relu:     { label: "ReLU 激活",    color: "bg-amber-100 text-amber-700" },
    linear2:  { label: "W₂·h 矩阵乘", color: "bg-cyan-100 text-cyan-700" },
    bias2:    { label: "+b₂ 加偏置",   color: "bg-teal-100 text-teal-700" },
    complete: { label: "计算完成",     color: "bg-emerald-100 text-emerald-700" },
  };
  const info = map[phase] ?? { label: phase, color: "bg-gray-100 text-gray-700" };
  return (
    <span className={`text-xs px-3 py-1 rounded-full font-semibold ${info.color}`}>
      {info.label}
    </span>
  );
}

// ── 主组件 ───────────────────────────────────────────────────────────────
function FFNVisualizer() {
  const coreIdea = getAiProblemCoreIdea(10007);

  return (
    <ConfigurableVisualizer<FFNInput, Record<string, never>>
      config={{
        defaultInput: {
          x: DEFAULT_X,
          W1: DEFAULT_W1,
          b1: DEFAULT_B1,
          W2: DEFAULT_W2,
          b2: DEFAULT_B2,
        },

        algorithm: (input) => {
          const x  = parseVec(input.x as string | number[]);
          const W1 = parseMatrix(input.W1 as string | number[][], DEFAULT_W1);
          const b1 = parseVec(input.b1 as string | number[]);
          const W2 = parseMatrix(input.W2 as string | number[][], DEFAULT_W2);
          const b2 = parseVec(input.b2 as string | number[]);
          return generateFFNSteps(x, W1, b1, W2, b2);
        },

        inputTypes: [
          { type: "string", key: "x",  label: "输入向量 x（JSON）" },
          { type: "string", key: "W1", label: "W1 矩阵（JSON）" },
          { type: "string", key: "b1", label: "b1 偏置（JSON）" },
          { type: "string", key: "W2", label: "W2 矩阵（JSON）" },
          { type: "string", key: "b2", label: "b2 偏置（JSON）" },
        ],

        inputFields: [
          { type: "string", key: "x",  label: "输入向量 x（JSON）", placeholder: DEFAULT_X },
          { type: "string", key: "W1", label: "W1 矩阵（JSON）",   placeholder: DEFAULT_W1 },
          { type: "string", key: "b1", label: "b1 偏置（JSON）",   placeholder: DEFAULT_B1 },
          { type: "string", key: "W2", label: "W2 矩阵（JSON）",   placeholder: DEFAULT_W2 },
          { type: "string", key: "b2", label: "b2 偏置（JSON）",   placeholder: DEFAULT_B2 },
        ],

        testCases: [
          {
            label: "d_model=3, d_ff=6（默认）",
            value: { x: DEFAULT_X, W1: DEFAULT_W1, b1: DEFAULT_B1, W2: DEFAULT_W2, b2: DEFAULT_B2 },
          },
          {
            label: "d_model=2, d_ff=4",
            value: {
              x:  "[0.8, -0.5]",
              W1: "[[0.6,-0.4],[0.3,0.7],[-0.5,0.8],[0.1,-0.9]]",
              b1: "[0.1,-0.1,0.2,-0.2]",
              W2: "[[0.4,0.3,-0.5,0.7],[-0.3,0.8,0.1,-0.4]]",
              b2: "[0.0, 0.1]",
            },
          },
          {
            label: "全正输入 d_model=4, d_ff=8",
            value: {
              x:  "[1.0, 0.8, 0.6, 0.4]",
              W1: "[[0.5,-0.3,0.8,0.2],[0.2,0.7,-0.4,0.1],[0.9,-0.1,0.3,-0.5],[-0.6,0.4,0.7,0.3],[0.1,-0.8,0.5,-0.2],[0.3,0.6,-0.2,0.8],[-0.4,0.2,0.9,-0.1],[0.7,-0.5,0.4,0.6]]",
              b1: "[0.1,-0.1,0.2,0.0,-0.2,0.1,0.3,-0.3]",
              W2: "[[0.4,0.3,-0.5,0.7,-0.1,0.2,0.6,-0.4],[-0.3,0.8,0.1,-0.4,0.6,-0.7,0.2,0.5],[0.5,-0.2,0.9,0.3,-0.8,0.4,-0.1,0.7],[0.1,0.4,-0.3,0.6,0.2,-0.5,0.8,-0.2]]",
              b2: "[0.0,0.1,-0.1,0.05]",
            },
          },
        ],

        render: ({ visualization, variables }) => {
          const currentInput = visualization.input as FFNInput;
          const W1 = parseMatrix(currentInput.W1 as string | number[][], DEFAULT_W1);
          const b1 = parseVec(currentInput.b1 as string | number[]);
          const W2 = parseMatrix(currentInput.W2 as string | number[][], DEFAULT_W2);
          const b2 = parseVec(currentInput.b2 as string | number[]);

          const phase         = (variables?.phase         as string)               ?? "init";
          const currentNeuron = variables?.currentNeuron  as number | undefined;
          const x             = (variables?.x             as number[] | undefined)  ?? [];
          const linear1       = (variables?.linear1       as number[] | undefined)  ?? [];
          const z1            = (variables?.z1            as number[] | undefined)  ?? [];
          const h             = (variables?.h             as number[] | undefined)  ?? [];
          const linear2       = (variables?.linear2       as number[] | undefined)  ?? [];
          const output        = (variables?.output        as number[] | undefined)  ?? [];
          const dModel        = (variables?.dModel        as number | undefined)    ?? 0;
          const dFF           = (variables?.dFF           as number | undefined)    ?? 0;
          const dOut          = (variables?.dOut          as number | undefined)    ?? 0;

          const phaseSteps = [
            { id: "init",    label: "① 输入 x"    },
            { id: "linear1", label: "② W₁·x"     },
            { id: "bias1",   label: "③ +b₁"      },
            { id: "relu",    label: "④ ReLU"      },
            { id: "linear2", label: "⑤ W₂·h"     },
            { id: "bias2",   label: "⑥ +b₂"      },
            { id: "complete",label: "⑦ 输出"      },
          ];

          return (
            <div className="space-y-4">
              {coreIdea && <CoreIdeaBox {...coreIdea} />}

              {/* 标题 + 公式 + 阶段标签 */}
              <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <div>
                    <h3 className="text-base font-semibold text-gray-900">
                      前馈神经网络（FFN / Feed-Forward Network）
                    </h3>
                    <div className="mt-1">
                      <BlockMath math={"\\text{FFN}(\\mathbf{x}) = W_2 \\cdot \\text{ReLU}(W_1 \\mathbf{x} + \\mathbf{b}_1) + \\mathbf{b}_2"} />
                    </div>
                  </div>
                  <PhaseTag phase={phase} />
                </div>
                {dModel > 0 && (
                  <p className="text-xs text-gray-400 mt-1">
                    <InlineMath math={`d_{model}=${dModel}`} />
                    &nbsp;→&nbsp;
                    <InlineMath math={`d_{ff}=${dFF}`} />
                    （中间层扩展 {Math.round(dFF / dModel)}×）
                    &nbsp;→&nbsp;
                    <InlineMath math={`d_{out}=${dOut}`} />
                  </p>
                )}
              </div>

              {/* ★ 当前步骤说明（核心）*/}
              <StepExplainer
                phase={phase}
                currentNeuron={currentNeuron}
                x={x}
                W1={W1}
                b1={b1}
                W2={W2}
                b2={b2}
                linear1={linear1}
                z1={z1}
                h={h}
                linear2={linear2}
                output={output}
                dModel={dModel}
                dFF={dFF}
                dOut={dOut}
              />

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

export default FFNVisualizer;
