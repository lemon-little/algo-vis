import { InlineMath } from "react-katex";
import "katex/dist/katex.min.css";
import { ConfigurableVisualizer } from "@/components/visualizers/ConfigurableVisualizer";
import { CoreIdeaBox } from "@/components/visualizers/CoreIdeaBox";
import { ProblemInput } from "@/types/visualization";
import { generateLayerNormSteps } from "./algorithm";
import { getAiProblemCoreIdea } from "@/config/aiProblemCoreIdeas";

interface LayerNormInput extends ProblemInput {
  inputVec: string | number[];
  gamma: number;
  beta: number;
  epsilon: number;
}

const defaultVec = "[2, 4, 6, 8]";

function parseVector(raw: string | number[]): number[] {
  if (Array.isArray(raw)) return raw as number[];
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed) && parsed.every((v: unknown) => typeof v === "number")) {
      return parsed as number[];
    }
  } catch {
    /* ignore */
  }
  return JSON.parse(defaultVec) as number[];
}

function formatNum(val: number | undefined, digits = 3): string {
  if (val === undefined || Number.isNaN(val)) return "--";
  return val.toFixed(digits);
}

function LayerNormVisualizer() {
  const coreIdea = getAiProblemCoreIdea(10008);

  return (
    <ConfigurableVisualizer<LayerNormInput, Record<string, never>>
      config={{
        defaultInput: {
          inputVec: defaultVec,
          gamma: 1,
          beta: 0,
          epsilon: 0.00001,
        },
        algorithm: (input) => {
          const vec = parseVector(input.inputVec);
          const gamma = input.gamma ?? 1;
          const beta = input.beta ?? 0;
          const epsilon = input.epsilon ?? 0.00001;
          return generateLayerNormSteps(vec, gamma, beta, epsilon);
        },
        inputTypes: [
          { type: "string", key: "inputVec", label: "输入向量（JSON）" },
          { type: "number", key: "gamma", label: "γ (缩放)" },
          { type: "number", key: "beta", label: "β (平移)" },
          { type: "number", key: "epsilon", label: "ε (稳定项)" },
        ],
        inputFields: [
          { type: "string", key: "inputVec", label: "输入向量（JSON）", placeholder: defaultVec },
          { type: "number", key: "gamma", label: "γ (缩放)", placeholder: "1" },
          { type: "number", key: "beta", label: "β (平移)", placeholder: "0" },
          { type: "number", key: "epsilon", label: "ε (稳定项)", placeholder: "0.00001" },
        ],
        testCases: [
          { label: "默认示例", value: { inputVec: defaultVec, gamma: 1, beta: 0, epsilon: 0.00001 } },
          { label: "缩放γ=2", value: { inputVec: defaultVec, gamma: 2, beta: 0, epsilon: 0.00001 } },
          { label: "平移β=1", value: { inputVec: defaultVec, gamma: 1, beta: 1, epsilon: 0.00001 } },
          { label: "大方差向量", value: { inputVec: "[1, 10, 100, 1000]", gamma: 1, beta: 0, epsilon: 0.00001 } },
        ],
        render: ({ variables }) => {
          const input = (variables?.input as number[] | undefined) || parseVector(defaultVec);
          const mean = variables?.mean as number | undefined;
          const variance = variables?.variance as number | undefined;
          const normalized = variables?.normalized as number[] | undefined;
          const output = variables?.output as number[] | undefined;
          const phase = (variables?.phase as string) || "init";
          const gamma = (variables?.gamma as number) ?? 1;
          const beta = (variables?.beta as number) ?? 0;

          const dim = input.length;

          // Color scale for values: map to hsl blue→red
          const allVals = [...input, ...(normalized ?? []), ...(output ?? [])];
          const minVal = allVals.length ? Math.min(...allVals) : 0;
          const maxVal = allVals.length ? Math.max(...allVals) : 1;
          const toColor = (v: number): string => {
            const ratio = maxVal === minVal ? 0.5 : (v - minVal) / (maxVal - minVal);
            const hue = Math.round((1 - ratio) * 240); // 240=blue, 0=red
            return `hsl(${hue}, 70%, 85%)`;
          };

          return (
            <div className="space-y-4">
              {coreIdea && <CoreIdeaBox {...coreIdea} />}

              {/* Header */}
              <div className="bg-white rounded-lg border border-gray-200 p-5 shadow-sm">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">层归一化（Layer Normalization）</h3>
                    <p className="text-sm text-gray-600 mt-1">
                      在单个样本的特征维度上计算均值和方差，归一化后通过可学习参数 γ/β 恢复表达能力。
                    </p>
                  </div>
                  <span
                    className={`text-xs px-2 py-1 rounded-full font-medium ml-4 shrink-0 ${
                      phase === "init"
                        ? "bg-gray-100 text-gray-600"
                        : phase === "mean"
                        ? "bg-blue-100 text-blue-700"
                        : phase === "variance"
                        ? "bg-amber-100 text-amber-700"
                        : phase === "normalize"
                        ? "bg-purple-100 text-purple-700"
                        : phase === "scale"
                        ? "bg-emerald-100 text-emerald-700"
                        : "bg-green-100 text-green-700"
                    }`}
                  >
                    {phase === "init"
                      ? "初始化"
                      : phase === "mean"
                      ? "计算均值"
                      : phase === "variance"
                      ? "计算方差"
                      : phase === "normalize"
                      ? "标准化"
                      : phase === "scale"
                      ? "缩放平移"
                      : "完成"}
                  </span>
                </div>
              </div>

              {/* LayerNorm vs BatchNorm contrast note */}
              <div className="bg-blue-50 rounded-lg border border-blue-200 p-4">
                <h4 className="text-sm font-semibold text-blue-800 mb-1">与批归一化的区别</h4>
                <div className="grid grid-cols-2 gap-3 text-xs text-blue-700 mt-2">
                  <div className="bg-white rounded p-2 border border-blue-100">
                    <div className="font-semibold mb-1">层归一化（LayerNorm）</div>
                    <div>沿特征维度计算（同一样本内）</div>
                    <div className="text-blue-500 mt-1">适合 NLP/Transformer</div>
                  </div>
                  <div className="bg-white rounded p-2 border border-blue-100">
                    <div className="font-semibold mb-1">批归一化（BatchNorm）</div>
                    <div>沿批次维度计算（同一特征）</div>
                    <div className="text-blue-500 mt-1">适合 CNN/视觉任务</div>
                  </div>
                </div>
              </div>

              {/* Input vector visualization */}
              <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
                <h4 className="text-sm font-semibold text-gray-800 mb-3">
                  输入向量 x（维度 = {dim}）
                </h4>
                <div className="flex gap-2 flex-wrap">
                  {input.map((v, i) => (
                    <div key={i} className="flex flex-col items-center">
                      <div
                        className="w-14 h-12 flex items-center justify-center rounded border text-sm font-mono font-semibold"
                        style={{ backgroundColor: toColor(v), borderColor: "#d1d5db" }}
                      >
                        {formatNum(v, 2)}
                      </div>
                      <div className="text-xs text-gray-400 mt-1">x[{i}]</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Statistics */}
              <div className="grid gap-4 md:grid-cols-2">
                <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
                  <h4 className="text-sm font-semibold text-gray-800 mb-3">统计量（沿特征维度）</h4>
                  <div className="space-y-3">
                    <div className="flex items-center gap-3">
                      <span className="text-xs text-gray-500 w-16">均值 μ</span>
                      {mean !== undefined ? (
                        <div className="px-3 py-1.5 bg-blue-50 border border-blue-200 rounded text-sm font-mono font-semibold text-blue-800">
                          {formatNum(mean)}
                        </div>
                      ) : (
                        <span className="text-gray-400 text-sm">等待计算...</span>
                      )}
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-xs text-gray-500 w-16">方差 σ²</span>
                      {variance !== undefined ? (
                        <div className="px-3 py-1.5 bg-amber-50 border border-amber-200 rounded text-sm font-mono font-semibold text-amber-800">
                          {formatNum(variance)}
                        </div>
                      ) : (
                        <span className="text-gray-400 text-sm">等待计算...</span>
                      )}
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
                  <h4 className="text-sm font-semibold text-gray-800 mb-3">可学习参数</h4>
                  <div className="space-y-3">
                    <div className="flex items-center gap-3">
                      <span className="text-xs text-gray-500 w-16">γ (缩放)</span>
                      <div className="px-3 py-1.5 bg-purple-50 border border-purple-200 rounded text-sm font-mono font-semibold text-purple-800">
                        {formatNum(gamma, 2)}
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-xs text-gray-500 w-16">β (平移)</span>
                      <div className="px-3 py-1.5 bg-pink-50 border border-pink-200 rounded text-sm font-mono font-semibold text-pink-800">
                        {formatNum(beta, 2)}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Normalized vector */}
              {normalized && normalized.length > 0 && (
                <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
                  <h4 className="text-sm font-semibold text-gray-800 mb-3">
                    标准化向量 x̂ &nbsp;
                    <span className="font-normal text-gray-500 text-xs">
                      = <InlineMath math={"\\hat{x}_i = \\dfrac{x_i - \\mu}{\\sqrt{\\sigma^2 + \\varepsilon}}"} />
                    </span>
                  </h4>
                  <div className="flex gap-2 flex-wrap">
                    {normalized.map((v, i) => (
                      <div key={i} className="flex flex-col items-center">
                        <div className="w-14 h-12 flex items-center justify-center rounded border text-sm font-mono font-semibold bg-purple-50 border-purple-200 text-purple-800">
                          {formatNum(v, 3)}
                        </div>
                        <div className="text-xs text-gray-400 mt-1">x̂[{i}]</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Output vector */}
              {output && output.length > 0 && (
                <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
                  <h4 className="text-sm font-semibold text-gray-800 mb-3">
                    输出向量 y &nbsp;
                    <span className="font-normal text-gray-500 text-xs">
                      = <InlineMath math={"y_i = \\gamma \\cdot \\hat{x}_i + \\beta"} />
                    </span>
                  </h4>
                  <div className="flex gap-2 flex-wrap">
                    {output.map((v, i) => (
                      <div key={i} className="flex flex-col items-center">
                        <div className="w-14 h-12 flex items-center justify-center rounded border text-sm font-mono font-semibold bg-emerald-50 border-emerald-200 text-emerald-800">
                          {formatNum(v, 3)}
                        </div>
                        <div className="text-xs text-gray-400 mt-1">y[{i}]</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Pipeline progress bar */}
              <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
                <h4 className="text-sm font-semibold text-gray-800 mb-3">计算流程</h4>
                <div className="flex items-center gap-1 text-xs">
                  {[
                    { key: "init", label: "初始化" },
                    { key: "mean", label: "计算均值" },
                    { key: "variance", label: "计算方差" },
                    { key: "normalize", label: "标准化" },
                    { key: "scale", label: "缩放平移" },
                    { key: "done", label: "完成" },
                  ].map((s, idx, arr) => {
                    const phases = ["init", "mean", "variance", "normalize", "scale", "done"];
                    const currentIdx = phases.indexOf(phase);
                    const stepIdx = phases.indexOf(s.key);
                    const isDone = stepIdx < currentIdx;
                    const isActive = s.key === phase;
                    return (
                      <div key={s.key} className="flex items-center">
                        <div
                          className={`px-2 py-1 rounded text-center ${
                            isActive
                              ? "bg-blue-500 text-white font-semibold"
                              : isDone
                              ? "bg-emerald-100 text-emerald-700"
                              : "bg-gray-100 text-gray-400"
                          }`}
                        >
                          {s.label}
                        </div>
                        {idx < arr.length - 1 && (
                          <div className={`w-3 h-0.5 mx-0.5 ${isDone || isActive ? "bg-blue-300" : "bg-gray-200"}`} />
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Formula reference */}
              <div className="bg-cyan-50 rounded-lg border border-cyan-200 p-4">
                <h4 className="text-sm font-semibold text-cyan-800 mb-3">公式</h4>
                <div className="space-y-2 text-cyan-900">
                  <div><InlineMath math={"\\mu = \\dfrac{1}{d} \\sum_{i=1}^{d} x_i"} /></div>
                  <div><InlineMath math={"\\sigma^2 = \\dfrac{1}{d} \\sum_{i=1}^{d} (x_i - \\mu)^2"} /></div>
                  <div><InlineMath math={"\\hat{x}_i = \\dfrac{x_i - \\mu}{\\sqrt{\\sigma^2 + \\varepsilon}}"} /></div>
                  <div><InlineMath math={"y_i = \\gamma \\cdot \\hat{x}_i + \\beta"} /></div>
                </div>
              </div>
            </div>
          );
        },
      }}
    />
  );
}

export default LayerNormVisualizer;
