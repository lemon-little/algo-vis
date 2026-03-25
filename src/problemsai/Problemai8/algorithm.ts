import { VisualizationStep } from "@/types";

export function generateLayerNormSteps(
  input: number[],
  gamma: number,
  beta: number,
  epsilon: number
): VisualizationStep[] {
  const steps: VisualizationStep[] = [];
  let stepId = 0;

  const dim = input.length;

  if (dim === 0) {
    steps.push({
      id: stepId++,
      description: "请输入有效的输入向量。",
      data: { input },
      variables: { finished: true },
    });
    return steps;
  }

  steps.push({
    id: stepId++,
    description: `初始化：输入维度 ${dim}，γ=${gamma}，β=${beta}，ε=${epsilon}`,
    data: { input, gamma, beta, epsilon },
    variables: { input, dim, gamma, beta, epsilon, phase: "init", batch: [input] },
  });

  // Step 1: compute mean across the entire vector (layer dimension)
  const mean = Number((input.reduce((s, v) => s + v, 0) / dim).toFixed(4));

  steps.push({
    id: stepId++,
    description: `计算均值 μ = (${input.map((v) => v.toFixed(2)).join(" + ")}) / ${dim} = ${mean}`,
    data: { input, mean },
    variables: { input, mean, phase: "mean", batch: [input] },
  });

  // Step 2: compute variance across the vector
  const variance = Number(
    (input.reduce((s, v) => s + (v - mean) ** 2, 0) / dim).toFixed(4)
  );

  steps.push({
    id: stepId++,
    description: `计算方差 σ² = Σ(xᵢ - μ)² / ${dim} = ${variance}`,
    data: { input, mean, variance },
    variables: { input, mean, variance, phase: "variance", batch: [input] },
  });

  // Step 3: normalize each element
  const normalized = input.map((v) =>
    Number(((v - mean) / Math.sqrt(variance + epsilon)).toFixed(4))
  );

  steps.push({
    id: stepId++,
    description: `标准化：x̂ᵢ = (xᵢ - μ) / √(σ² + ε)，结果：[${normalized.map((v) => v.toFixed(3)).join(", ")}]`,
    data: { input, mean, variance, normalized },
    variables: {
      input,
      mean,
      variance,
      normalized,
      phase: "normalize",
      batch: [input],
      normalizedBatch: [normalized],
    },
  });

  // Step 4: scale and shift with gamma / beta
  const output = normalized.map((v) =>
    Number((gamma * v + beta).toFixed(4))
  );

  steps.push({
    id: stepId++,
    description: `缩放平移：yᵢ = γ × x̂ᵢ + β，γ=${gamma}，β=${beta}，结果：[${output.map((v) => v.toFixed(3)).join(", ")}]`,
    data: { input, mean, variance, normalized, output, gamma, beta },
    variables: {
      input,
      mean,
      variance,
      normalized,
      output,
      gamma,
      beta,
      phase: "scale",
      batch: [input],
      normalizedBatch: [normalized],
      outputBatch: [output],
    },
  });

  steps.push({
    id: stepId++,
    description: "层归一化完成！输出向量均值≈0、方差≈1，再经 γ/β 恢复表达能力。",
    data: { input, mean, variance, normalized, output },
    variables: {
      input,
      mean,
      variance,
      normalized,
      output,
      gamma,
      beta,
      phase: "done",
      finished: true,
      batch: [input],
      normalizedBatch: [normalized],
      outputBatch: [output],
    },
  });

  return steps;
}
