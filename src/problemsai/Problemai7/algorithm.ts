import { VisualizationStep } from "@/types";

/**
 * ReLU activation function
 */
function relu(x: number): number {
  return Math.max(0, x);
}

/**
 * Apply ReLU to a vector
 */
function reluVec(vec: number[]): number[] {
  return vec.map((v) => Number(relu(v).toFixed(4)));
}

/**
 * Matrix × vector multiplication: W (out×in) · x (in) → out
 */
function matVecMul(W: number[][], x: number[]): number[] {
  return W.map((row) =>
    Number(row.reduce((acc, w, i) => acc + w * (x[i] ?? 0), 0).toFixed(4))
  );
}

/**
 * Add bias vector to a vector
 */
function addBias(vec: number[], bias: number[]): number[] {
  return vec.map((v, i) => Number((v + (bias[i] ?? 0)).toFixed(4)));
}

export function generateFFNSteps(
  x: number[],
  W1: number[][],
  b1: number[],
  W2: number[][],
  b2: number[]
): VisualizationStep[] {
  const steps: VisualizationStep[] = [];
  let stepId = 0;

  const dModel = x.length;
  const dFF = W1.length;
  const dOut = W2.length;

  if (dModel === 0 || dFF === 0 || dOut === 0) {
    steps.push({
      id: 0,
      description: "请输入有效的输入向量和权重矩阵。",
      data: {},
      variables: { phase: "error", finished: true },
    });
    return steps;
  }

  // Step 1: 初始化
  steps.push({
    id: stepId++,
    description: `初始化：输入维度 d_model=${dModel}，中间层维度 d_ff=${dFF}，输出维度=${dOut}。FFN 通过两次线性变换 + ReLU 激活，对每个 Token 独立进行非线性特征变换。`,
    data: { x, W1, b1, W2, b2 },
    variables: {
      phase: "init",
      x: [...x],
      W1: W1.map((r) => [...r]),
      b1: [...b1],
      W2: W2.map((r) => [...r]),
      b2: [...b2],
      dModel,
      dFF,
      dOut,
    },
  });

  // Step 2: 第一层线性变换 W1·x（逐行展开）
  const linear1: number[] = [];
  for (let i = 0; i < dFF; i++) {
    let sum = 0;
    for (let d = 0; d < dModel; d++) {
      sum += (W1[i]?.[d] ?? 0) * (x[d] ?? 0);
    }
    linear1[i] = Number(sum.toFixed(4));

    steps.push({
      id: stepId++,
      description: `第一层线性变换：计算 W1[${i}] · x = ${W1[i]?.map((w, d) => `${w.toFixed(2)}×${(x[d] ?? 0).toFixed(2)}`).join(" + ")} = ${linear1[i].toFixed(4)}（第 ${i} 个神经元，尚未加偏置）`,
      data: {},
      variables: {
        phase: "linear1",
        x: [...x],
        W1: W1.map((r) => [...r]),
        b1: [...b1],
        linear1: [...linear1],
        currentNeuron: i,
        dModel,
        dFF,
        dOut,
      },
    });
  }

  // Step 3: 加偏置 b1
  const z1 = addBias(linear1, b1);
  steps.push({
    id: stepId++,
    description: `加上偏置 b1：z1 = W1·x + b1，得到第一层线性输出 z1 ∈ ℝ^${dFF}。偏置为每个神经元提供独立的激活阈值。`,
    data: {},
    variables: {
      phase: "bias1",
      x: [...x],
      linear1: [...linear1],
      b1: [...b1],
      z1: [...z1],
      dModel,
      dFF,
      dOut,
    },
  });

  // Step 4: ReLU 激活（逐元素展开）
  const h: number[] = [];
  for (let i = 0; i < dFF; i++) {
    h[i] = Number(relu(z1[i]).toFixed(4));

    steps.push({
      id: stepId++,
      description: `ReLU 激活：h[${i}] = max(0, ${z1[i].toFixed(4)}) = ${h[i].toFixed(4)}${z1[i] < 0 ? "（负值被截断为 0，神经元不激活）" : "（正值保留，神经元激活）"}`,
      data: {},
      variables: {
        phase: "relu",
        z1: [...z1],
        h: [...h],
        currentNeuron: i,
        dModel,
        dFF,
        dOut,
      },
    });
  }

  // Step 5: 第二层线性变换 W2·h（逐行展开）
  const linear2: number[] = [];
  for (let i = 0; i < dOut; i++) {
    let sum = 0;
    for (let d = 0; d < dFF; d++) {
      sum += (W2[i]?.[d] ?? 0) * (h[d] ?? 0);
    }
    linear2[i] = Number(sum.toFixed(4));

    steps.push({
      id: stepId++,
      description: `第二层线性变换：计算 W2[${i}] · h = ${W2[i]?.map((w, d) => `${w.toFixed(2)}×${(h[d] ?? 0).toFixed(2)}`).slice(0, 4).join(" + ")}${dFF > 4 ? "..." : ""} = ${linear2[i].toFixed(4)}（第 ${i} 个输出神经元）`,
      data: {},
      variables: {
        phase: "linear2",
        h: [...h],
        W2: W2.map((r) => [...r]),
        b2: [...b2],
        linear2: [...linear2],
        currentNeuron: i,
        dModel,
        dFF,
        dOut,
      },
    });
  }

  // Step 6: 加偏置 b2，得到最终输出
  const output = addBias(linear2, b2);
  steps.push({
    id: stepId++,
    description: `加上偏置 b2：output = W2·h + b2，得到最终输出 ∈ ℝ^${dOut}。FFN 将激活后的中间表示重新投影回模型维度，完成非线性特征融合。`,
    data: {},
    variables: {
      phase: "bias2",
      h: [...h],
      linear2: [...linear2],
      b2: [...b2],
      output: [...output],
      dModel,
      dFF,
      dOut,
    },
  });

  // Step 7: 完成
  steps.push({
    id: stepId++,
    description: `计算完成！FFN 通过 "扩展-激活-压缩" 的两层结构，对 Attention 输出进行非线性变换：第一层将维度从 d_model(${dModel}) 扩展到 d_ff(${dFF})（通常是 4 倍），ReLU 引入非线性，第二层再压缩回 d_model(${dOut})。`,
    data: {},
    variables: {
      phase: "complete",
      x: [...x],
      z1: [...z1],
      h: [...h],
      output: [...output],
      dModel,
      dFF,
      dOut,
      finished: true,
    },
  });

  return steps;
}

/**
 * 生成随机 FFN 权重（用于测试用例）
 */
export function makeDefaultFFN(dModel: number, dFF: number) {
  const W1 = Array.from({ length: dFF }, () =>
    Array.from({ length: dModel }, () => Number((Math.random() * 2 - 1).toFixed(2)))
  );
  const b1 = Array.from({ length: dFF }, () => Number((Math.random() * 0.5 - 0.25).toFixed(2)));
  const W2 = Array.from({ length: dModel }, () =>
    Array.from({ length: dFF }, () => Number((Math.random() * 2 - 1).toFixed(2)))
  );
  const b2 = Array.from({ length: dModel }, () => Number((Math.random() * 0.5 - 0.25).toFixed(2)));
  return { W1, b1, W2, b2 };
}

/**
 * Compute linear1 result (W1·x)
 */
export function computeLinear1(W1: number[][], x: number[]): number[] {
  return matVecMul(W1, x);
}

/**
 * Compute relu activation
 */
export function computeRelu(z: number[]): number[] {
  return reluVec(z);
}
