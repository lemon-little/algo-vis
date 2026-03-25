import { InlineMath } from "react-katex";
import "katex/dist/katex.min.css";
import { ConfigurableVisualizer } from "@/components/visualizers/ConfigurableVisualizer";
import { CoreIdeaBox } from "@/components/visualizers/CoreIdeaBox";
import { ProblemInput } from "@/types/visualization";
import { generateResidualConnectionSteps } from "./algorithm";
import { getAiProblemCoreIdea } from "@/config/aiProblemCoreIdeas";

const PROBLEM_ID = 10009;

interface ResidualInput extends ProblemInput {
  inputVec: string | number[];
  sublayerVec: string | number[];
}

const DEFAULT_INPUT = "[1, 2, 3, 4]";
const DEFAULT_SUBLAYER = "[0.5, -0.5, 1.0, -1.0]";

function parseVec(raw: string | number[], fallback: number[]): number[] {
  if (Array.isArray(raw)) return raw as number[];
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed) && parsed.every((v: unknown) => typeof v === "number")) {
      return parsed as number[];
    }
  } catch {
    /* ignore */
  }
  return fallback;
}

function fmt(v: number | undefined, d = 3): string {
  if (v === undefined || Number.isNaN(v)) return "--";
  return v.toFixed(d);
}

function ValueCell({
  value,
  label,
  color = "gray",
  highlight = false,
}: {
  value: number;
  label: string;
  color?: "gray" | "blue" | "amber" | "purple" | "emerald" | "rose";
  highlight?: boolean;
}) {
  const colorMap: Record<string, string> = {
    gray: "bg-gray-50 border-gray-200 text-gray-800",
    blue: "bg-blue-50 border-blue-200 text-blue-800",
    amber: "bg-amber-50 border-amber-200 text-amber-800",
    purple: "bg-purple-50 border-purple-200 text-purple-800",
    emerald: "bg-emerald-50 border-emerald-200 text-emerald-800",
    rose: "bg-rose-50 border-rose-200 text-rose-800",
  };
  const base = colorMap[color] ?? colorMap.gray;
  const ring = highlight ? "ring-2 ring-offset-1 ring-blue-400 scale-105" : "";
  return (
    <div className="flex flex-col items-center">
      <div
        className={`w-14 h-12 flex items-center justify-center rounded border text-sm font-mono font-semibold transition-all ${base} ${ring}`}
      >
        {fmt(value, 3)}
      </div>
      <div className="text-xs text-gray-400 mt-1">{label}</div>
    </div>
  );
}

function VectorRow({
  values,
  label,
  color,
  highlightIdx,
}: {
  values: number[];
  label: string;
  color: "gray" | "blue" | "amber" | "purple" | "emerald" | "rose";
  highlightIdx?: number;
}) {
  return (
    <div>
      <div className="text-xs font-semibold text-gray-600 mb-2">{label}</div>
      <div className="flex gap-2 flex-wrap">
        {values.map((v, i) => (
          <ValueCell
            key={i}
            value={v}
            label={`[${i}]`}
            color={color}
            highlight={highlightIdx === i}
          />
        ))}
      </div>
    </div>
  );
}

const PHASES = ["init", "sublayer", "add", "addDone", "lnMean", "lnVar", "lnNorm", "done"];
const PHASE_LABELS: Record<string, string> = {
  init: "初始化",
  sublayer: "子层输出",
  add: "残差相加",
  addDone: "相加完成",
  lnMean: "计算均值",
  lnVar: "计算方差",
  lnNorm: "归一化",
  done: "完成",
};
const PHASE_COLORS: Record<string, string> = {
  init: "bg-gray-100 text-gray-600",
  sublayer: "bg-blue-100 text-blue-700",
  add: "bg-amber-100 text-amber-700",
  addDone: "bg-amber-100 text-amber-800",
  lnMean: "bg-purple-100 text-purple-700",
  lnVar: "bg-purple-100 text-purple-800",
  lnNorm: "bg-cyan-100 text-cyan-700",
  done: "bg-emerald-100 text-emerald-700",
};

function ResidualConnectionVisualizer() {
  const coreIdea = getAiProblemCoreIdea(PROBLEM_ID);
  const defaultInputVec = parseVec(DEFAULT_INPUT, [1, 2, 3, 4]);
  const defaultSublayerVec = parseVec(DEFAULT_SUBLAYER, [0.5, -0.5, 1.0, -1.0]);

  return (
    <ConfigurableVisualizer<ResidualInput, Record<string, never>>
      config={{
        defaultInput: {
          inputVec: DEFAULT_INPUT,
          sublayerVec: DEFAULT_SUBLAYER,
        },

        algorithm: (input) => {
          const x = parseVec(input.inputVec, defaultInputVec);
          const fx = parseVec(input.sublayerVec, defaultSublayerVec);
          const len = Math.min(x.length, fx.length);
          return generateResidualConnectionSteps(x.slice(0, len), fx.slice(0, len));
        },

        inputTypes: [
          { type: "string", key: "inputVec", label: "输入 x（JSON）" },
          { type: "string", key: "sublayerVec", label: "子层输出 F(x)（JSON）" },
        ],

        inputFields: [
          {
            type: "string",
            key: "inputVec",
            label: "输入 x（JSON）",
            placeholder: DEFAULT_INPUT,
          },
          {
            type: "string",
            key: "sublayerVec",
            label: "子层输出 F(x)（JSON）",
            placeholder: DEFAULT_SUBLAYER,
          },
        ],

        testCases: [
          {
            label: "默认示例",
            value: { inputVec: DEFAULT_INPUT, sublayerVec: DEFAULT_SUBLAYER },
          },
          {
            label: "小扰动 F(x)",
            value: { inputVec: "[1, 2, 3, 4]", sublayerVec: "[0.1, -0.1, 0.2, -0.2]" },
          },
          {
            label: "零子层输出",
            value: { inputVec: "[1, 2, 3, 4]", sublayerVec: "[0, 0, 0, 0]" },
          },
          {
            label: "大值输入",
            value: { inputVec: "[10, 20, 30, 40]", sublayerVec: "[5, -5, 10, -10]" },
          },
        ],

        render: ({ variables }) => {
          const phase = (variables?.phase as string) ?? "init";
          const input = (variables?.input as number[] | undefined) ?? defaultInputVec;
          const sublayerOutput = (variables?.sublayerOutput as number[] | undefined) ?? defaultSublayerVec;
          const added = variables?.added as number[] | undefined;
          const addIdx = variables?.addIdx as number | undefined;
          const mean = variables?.mean as number | undefined;
          const variance = variables?.variance as number | undefined;
          const normalized = variables?.normalized as number[] | undefined;

          const phaseIdx = PHASES.indexOf(phase);
          const showSublayer = phaseIdx >= PHASES.indexOf("sublayer");
          const showAdded = phaseIdx >= PHASES.indexOf("add");
          const showStats = phaseIdx >= PHASES.indexOf("lnMean");
          const showOutput = phaseIdx >= PHASES.indexOf("lnNorm");

          return (
            <div className="space-y-4">
              {/* Core idea */}
              {coreIdea && <CoreIdeaBox {...coreIdea} />}

              {/* Header */}
              <div className="bg-white rounded-lg border border-gray-200 p-5 shadow-sm">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">
                      Transformer 残差连接（Add &amp; Norm）
                    </h3>
                    <p className="text-sm text-gray-600 mt-1">
                      将子层输入直接加到子层输出，再送入层归一化，形成跳跃路径使梯度畅通反传。
                    </p>
                  </div>
                  <span
                    className={`text-xs px-2 py-1 rounded-full font-medium ml-4 shrink-0 ${PHASE_COLORS[phase] ?? "bg-gray-100 text-gray-600"}`}
                  >
                    {PHASE_LABELS[phase] ?? phase}
                  </span>
                </div>
              </div>

              {/* Architecture diagram */}
              <div className="bg-blue-50 rounded-lg border border-blue-200 p-4">
                <h4 className="text-sm font-semibold text-blue-800 mb-3">数据流示意</h4>
                <div className="flex items-center justify-center gap-3 text-xs text-center flex-wrap">
                  {/* x */}
                  <div className="flex flex-col items-center gap-1">
                    <div className="px-3 py-2 bg-white rounded border border-blue-200 font-mono text-blue-800 font-semibold">
                      x
                    </div>
                    <div className="text-blue-600">输入</div>
                  </div>
                  {/* arrow fork */}
                  <div className="text-blue-400 text-lg">→</div>
                  {/* sublayer */}
                  <div className="flex flex-col items-center gap-1">
                    <div className="px-3 py-2 bg-amber-50 rounded border border-amber-200 font-mono text-amber-800">
                      Sublayer(x)
                    </div>
                    <div className="text-amber-600">子层（注意力/FFN）</div>
                  </div>
                  {/* plus */}
                  <div className="flex flex-col items-center gap-1">
                    <div className="px-3 py-2 bg-rose-50 rounded-full border border-rose-200 font-bold text-rose-700 w-10 h-10 flex items-center justify-center text-lg">
                      +
                    </div>
                    <div className="text-rose-600">残差相加</div>
                  </div>
                  {/* arrow */}
                  <div className="text-blue-400 text-lg">→</div>
                  {/* LayerNorm */}
                  <div className="flex flex-col items-center gap-1">
                    <div className="px-3 py-2 bg-purple-50 rounded border border-purple-200 font-mono text-purple-800">
                      LayerNorm
                    </div>
                    <div className="text-purple-600">层归一化</div>
                  </div>
                  {/* arrow */}
                  <div className="text-blue-400 text-lg">→</div>
                  {/* output */}
                  <div className="flex flex-col items-center gap-1">
                    <div className="px-3 py-2 bg-emerald-50 rounded border border-emerald-200 font-mono text-emerald-800 font-semibold">
                      Output
                    </div>
                    <div className="text-emerald-600">输出</div>
                  </div>
                </div>
                {/* skip connection label */}
                <div className="text-center text-xs text-blue-500 mt-2">
                  ↑ 跳跃连接（x 绕过子层直接相加）
                </div>
              </div>

              {/* Input x */}
              <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
                <h4 className="text-sm font-semibold text-gray-800 mb-3">
                  输入 x（维度 = {input.length}）
                </h4>
                <VectorRow values={input} label="x" color="blue" />
              </div>

              {/* Sublayer output F(x) */}
              {showSublayer && (
                <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
                  <h4 className="text-sm font-semibold text-gray-800 mb-3">
                    子层输出 F(x)
                  </h4>
                  <VectorRow values={sublayerOutput} label="F(x)" color="amber" />
                </div>
              )}

              {/* Residual addition */}
              {showAdded && added && (
                <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
                  <h4 className="text-sm font-semibold text-gray-800 mb-3">
                    残差相加：x + F(x)&nbsp;
                    <span className="font-normal text-gray-500 text-xs">
                      = <InlineMath math={"x + \\text{Sublayer}(x)"} />
                    </span>
                  </h4>
                  <div className="space-y-2">
                    {/* Element-wise table */}
                    <div className="overflow-x-auto">
                      <table className="text-xs border-collapse w-full">
                        <thead>
                          <tr className="bg-gray-50">
                            <th className="border border-gray-200 px-2 py-1 text-left text-gray-500">idx</th>
                            <th className="border border-gray-200 px-2 py-1 text-center text-blue-700">x[i]</th>
                            <th className="border border-gray-200 px-2 py-1 text-center text-amber-700">F(x)[i]</th>
                            <th className="border border-gray-200 px-2 py-1 text-center text-rose-700">x[i]+F(x)[i]</th>
                          </tr>
                        </thead>
                        <tbody>
                          {input.map((xv, i) => {
                            const fxv = sublayerOutput[i];
                            const sumv = added[i];
                            const isActive = phase === "add" && addIdx === i;
                            const isDone = sumv !== undefined && (phase !== "add" || (addIdx ?? -1) >= i);
                            return (
                              <tr
                                key={i}
                                className={isActive ? "bg-amber-50" : isDone ? "bg-rose-50/40" : ""}
                              >
                                <td className="border border-gray-200 px-2 py-1 font-mono text-gray-500">{i}</td>
                                <td className="border border-gray-200 px-2 py-1 font-mono text-center text-blue-800">
                                  {fmt(xv, 3)}
                                </td>
                                <td className="border border-gray-200 px-2 py-1 font-mono text-center text-amber-800">
                                  {fmt(fxv, 3)}
                                </td>
                                <td className="border border-gray-200 px-2 py-1 font-mono text-center text-rose-800 font-semibold">
                                  {sumv !== undefined ? fmt(sumv, 3) : "…"}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}

              {/* LayerNorm stats */}
              {showStats && (
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
                    <h4 className="text-sm font-semibold text-gray-800 mb-3">LayerNorm 统计量</h4>
                    <div className="space-y-3">
                      <div className="flex items-center gap-3">
                        <span className="text-xs text-gray-500 w-16">均值 μ</span>
                        {mean !== undefined ? (
                          <div className="px-3 py-1.5 bg-purple-50 border border-purple-200 rounded text-sm font-mono font-semibold text-purple-800">
                            {fmt(mean)}
                          </div>
                        ) : (
                          <span className="text-gray-400 text-sm">等待计算…</span>
                        )}
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="text-xs text-gray-500 w-16">方差 σ²</span>
                        {variance !== undefined ? (
                          <div className="px-3 py-1.5 bg-amber-50 border border-amber-200 rounded text-sm font-mono font-semibold text-amber-800">
                            {fmt(variance)}
                          </div>
                        ) : (
                          <span className="text-gray-400 text-sm">等待计算…</span>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
                    <h4 className="text-sm font-semibold text-gray-800 mb-2">
                      LayerNorm 公式
                    </h4>
                    <div className="space-y-1 text-sm text-gray-700">
                      <div>
                        <InlineMath math={"\\mu = \\frac{1}{d}\\sum_{i} a_i"} />
                      </div>
                      <div>
                        <InlineMath math={"\\sigma^2 = \\frac{1}{d}\\sum_{i}(a_i - \\mu)^2"} />
                      </div>
                      <div>
                        <InlineMath math={"\\hat{a}_i = \\frac{a_i - \\mu}{\\sqrt{\\sigma^2 + \\varepsilon}}"} />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Normalized output */}
              {showOutput && normalized && (
                <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
                  <h4 className="text-sm font-semibold text-gray-800 mb-3">
                    最终输出&nbsp;
                    <span className="font-normal text-gray-500 text-xs">
                      = <InlineMath math={"\\text{LayerNorm}(x + \\text{Sublayer}(x))"} />
                    </span>
                  </h4>
                  <VectorRow values={normalized} label="output" color="emerald" />
                </div>
              )}

              {/* Pipeline progress */}
              <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
                <h4 className="text-sm font-semibold text-gray-800 mb-3">计算流程</h4>
                <div className="flex items-center gap-1 flex-wrap text-xs">
                  {PHASES.map((key, idx) => {
                    const currentIdx = PHASES.indexOf(phase);
                    const isDone = idx < currentIdx;
                    const isActive = key === phase;
                    return (
                      <div key={key} className="flex items-center">
                        <div
                          className={`px-2 py-1 rounded text-center ${
                            isActive
                              ? "bg-blue-500 text-white font-semibold"
                              : isDone
                              ? "bg-emerald-100 text-emerald-700"
                              : "bg-gray-100 text-gray-400"
                          }`}
                        >
                          {PHASE_LABELS[key]}
                        </div>
                        {idx < PHASES.length - 1 && (
                          <div
                            className={`w-3 h-0.5 mx-0.5 ${isDone || isActive ? "bg-blue-300" : "bg-gray-200"}`}
                          />
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Full formula */}
              <div className="bg-rose-50 rounded-lg border border-rose-200 p-4">
                <h4 className="text-sm font-semibold text-rose-800 mb-3">核心公式</h4>
                <div className="space-y-2 text-rose-900">
                  <div>
                    <InlineMath math={"\\text{Output} = \\text{LayerNorm}(x + \\text{Sublayer}(x))"} />
                  </div>
                  <div className="text-xs text-rose-600 mt-2">
                    其中 Sublayer 可以是多头注意力（MHA）或前馈网络（FFN）
                  </div>
                </div>
              </div>

              {/* Importance note */}
              <div className="bg-gray-50 rounded-lg border border-gray-200 p-4">
                <h4 className="text-sm font-semibold text-gray-700 mb-2">为什么残差连接至关重要？</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs text-gray-600">
                  <div className="bg-white rounded p-3 border border-gray-100">
                    <div className="font-semibold text-gray-800 mb-1">梯度高速公路</div>
                    <div>跳跃连接提供梯度直通路径，缓解深层网络的梯度消失问题。</div>
                  </div>
                  <div className="bg-white rounded p-3 border border-gray-100">
                    <div className="font-semibold text-gray-800 mb-1">学习残差映射</div>
                    <div>子层只需学习输入与目标的"差量" F(x)，而非完整变换。</div>
                  </div>
                  <div className="bg-white rounded p-3 border border-gray-100">
                    <div className="font-semibold text-gray-800 mb-1">信息保留</div>
                    <div>原始输入信号始终保留，子层退化为恒等映射时模型不失效。</div>
                  </div>
                </div>
              </div>
            </div>
          );
        },
      }}
    />
  );
}

export default ResidualConnectionVisualizer;
