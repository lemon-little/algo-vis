import { VisualizationStep } from "@/types";

/**
 * Residual Connection (Add & Norm) in Transformer
 *
 * The operation is: Output = LayerNorm(x + Sublayer(x))
 *
 * We visualize:
 *  1. Input x
 *  2. Sublayer output F(x)  (e.g. attention or FFN output)
 *  3. Residual addition: x + F(x)
 *  4. LayerNorm applied to the sum → final output
 *
 * Steps are element-wise so the animation is granular.
 */
export function generateResidualConnectionSteps(
  input: number[],
  sublayerOutput: number[],
  epsilon = 1e-5
): VisualizationStep[] {
  const steps: VisualizationStep[] = [];
  let stepId = 0;
  const dim = input.length;

  // Step 0 – Init
  steps.push({
    id: stepId++,
    description: `初始化：输入 x 维度 ${dim}，子层输出 F(x) 维度 ${sublayerOutput.length}。残差连接公式：Output = LayerNorm(x + F(x))`,
    data: { input, sublayerOutput },
    variables: {
      phase: "init",
      input,
      sublayerOutput,
    },
  });

  // Step 1 – show sublayer output
  steps.push({
    id: stepId++,
    description: `子层（如注意力或 FFN）已完成计算，输出 F(x) = [${sublayerOutput.map((v) => v.toFixed(3)).join(", ")}]`,
    data: { input, sublayerOutput },
    variables: {
      phase: "sublayer",
      input,
      sublayerOutput,
    },
  });

  // Step 2 – residual addition (element-wise), one element at a time
  const added: number[] = [];
  for (let i = 0; i < dim; i++) {
    added.push(Number((input[i] + sublayerOutput[i]).toFixed(4)));
    steps.push({
      id: stepId++,
      description: `残差相加第 ${i + 1}/${dim} 位：x[${i}] + F(x)[${i}] = ${input[i].toFixed(3)} + ${sublayerOutput[i].toFixed(3)} = ${added[i].toFixed(3)}`,
      data: { input, sublayerOutput, added: [...added] },
      variables: {
        phase: "add",
        input,
        sublayerOutput,
        added: [...added],
        addIdx: i,
      },
    });
  }

  // Step 3 – all elements added, show full sum
  steps.push({
    id: stepId++,
    description: `残差相加完成：x + F(x) = [${added.map((v) => v.toFixed(3)).join(", ")}]`,
    data: { input, sublayerOutput, added },
    variables: {
      phase: "addDone",
      input,
      sublayerOutput,
      added,
    },
  });

  // Step 4 – LayerNorm: compute mean
  const mean = Number((added.reduce((s, v) => s + v, 0) / dim).toFixed(4));
  steps.push({
    id: stepId++,
    description: `LayerNorm — 计算均值 μ = (${added.map((v) => v.toFixed(2)).join(" + ")}) / ${dim} = ${mean}`,
    data: { input, sublayerOutput, added, mean },
    variables: {
      phase: "lnMean",
      input,
      sublayerOutput,
      added,
      mean,
    },
  });

  // Step 5 – LayerNorm: compute variance
  const variance = Number(
    (added.reduce((s, v) => s + (v - mean) ** 2, 0) / dim).toFixed(4)
  );
  steps.push({
    id: stepId++,
    description: `LayerNorm — 计算方差 σ² = Σ(aᵢ - μ)² / ${dim} = ${variance}`,
    data: { input, sublayerOutput, added, mean, variance },
    variables: {
      phase: "lnVar",
      input,
      sublayerOutput,
      added,
      mean,
      variance,
    },
  });

  // Step 6 – LayerNorm: normalize
  const normalized = added.map((v) =>
    Number(((v - mean) / Math.sqrt(variance + epsilon)).toFixed(4))
  );
  steps.push({
    id: stepId++,
    description: `LayerNorm — 标准化：x̂ᵢ = (aᵢ - μ) / √(σ² + ε)，结果：[${normalized.map((v) => v.toFixed(3)).join(", ")}]`,
    data: { input, sublayerOutput, added, mean, variance, normalized },
    variables: {
      phase: "lnNorm",
      input,
      sublayerOutput,
      added,
      mean,
      variance,
      normalized,
    },
  });

  // Step 7 – done (gamma=1, beta=0 default, output = normalized)
  steps.push({
    id: stepId++,
    description: `残差连接完成！Output = LayerNorm(x + F(x)) = [${normalized.map((v) => v.toFixed(3)).join(", ")}]。跳跃路径使梯度可直接反传至输入层。`,
    data: { input, sublayerOutput, added, mean, variance, normalized },
    variables: {
      phase: "done",
      input,
      sublayerOutput,
      added,
      mean,
      variance,
      normalized,
      finished: true,
    },
  });

  return steps;
}
