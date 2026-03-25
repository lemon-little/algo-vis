import { VisualizationStep } from "@/types";

function softmaxRowMasked(row: number[], maskIndices: Set<number>): number[] {
  // Apply mask: set masked positions to -Infinity before softmax
  const maskedRow = row.map((v, i) => (maskIndices.has(i) ? -Infinity : v));
  const maxVal = Math.max(...maskedRow.filter((v) => isFinite(v)));
  const exps = maskedRow.map((v) => (isFinite(v) ? Math.exp(v - maxVal) : 0));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => Number((sum === 0 ? 0 : v / sum).toFixed(4)));
}

function matMulWeightsV(weights: number[][], V: number[][]): number[][] {
  const seqLen = weights.length;
  const dv = V[0]?.length || 0;
  const result: number[][] = [];
  for (let i = 0; i < seqLen; i++) {
    result[i] = new Array(dv).fill(0);
    for (let j = 0; j < seqLen; j++) {
      for (let d = 0; d < dv; d++) {
        result[i][d] += weights[i][j] * (V[j]?.[d] ?? 0);
      }
    }
    result[i] = result[i].map((v) => Number(v.toFixed(4)));
  }
  return result;
}

/**
 * Build the causal mask matrix:
 * mask[i][j] = true means position j is MASKED (future token, should not be attended to)
 * For causal attention: query at position i can only attend to positions j <= i
 */
function buildCausalMask(seqLen: number): boolean[][] {
  const mask: boolean[][] = [];
  for (let i = 0; i < seqLen; i++) {
    mask[i] = [];
    for (let j = 0; j < seqLen; j++) {
      mask[i][j] = j > i; // true = masked (future)
    }
  }
  return mask;
}

export function generateCausalAttentionSteps(
  Q: number[][],
  K: number[][],
  V: number[][],
  dk: number,
  tokens?: string[]
): VisualizationStep[] {
  const steps: VisualizationStep[] = [];
  let stepId = 0;

  const seqLen = Q.length;
  const dq = Q[0]?.length || 0;
  const dv = V[0]?.length || 0;
  const safeDk = Math.max(dk, 1);
  const scale = 1 / Math.sqrt(safeDk);

  const tokenLabels = tokens && tokens.length === seqLen
    ? tokens
    : Array.from({ length: seqLen }, (_, i) => `t${i}`);

  if (seqLen === 0 || dq === 0) {
    steps.push({
      id: 0,
      description: "请输入有效的 Q、K、V 矩阵。",
      data: {},
      variables: { phase: "error", finished: true },
    });
    return steps;
  }

  const causalMask = buildCausalMask(seqLen);

  // Step 1: 初始化，展示因果掩码
  steps.push({
    id: stepId++,
    description: `初始化：序列长度 ${seqLen}，Q/K/V 维度 ${dq}/${dq}/${dv}，d_k = ${safeDk}。因果注意力通过下三角掩码矩阵，确保每个位置只能看到当前及之前的 Token。`,
    data: { Q, K, V, dk: safeDk, causalMask, tokenLabels },
    variables: {
      phase: "init",
      Q, K, V,
      dk: safeDk,
      scale,
      causalMask,
      tokenLabels,
    },
  });

  // Step 2: 展示因果掩码矩阵详细说明
  steps.push({
    id: stepId++,
    description: `因果掩码（Causal Mask）：下三角为 0（允许注意），上三角为 -∞（禁止看到未来）。对角线及以下表示当前或过去位置，上三角表示未来位置，通过将其设为 -∞ 后 Softmax 输出为 0。`,
    data: { causalMask },
    variables: {
      phase: "mask",
      causalMask,
      tokenLabels,
      Q, K, V,
    },
  });

  // Step 3: 逐行计算 QKᵀ 点积
  const scores: number[][] = [];
  for (let i = 0; i < seqLen; i++) {
    scores[i] = [];
    for (let j = 0; j < seqLen; j++) {
      let sum = 0;
      for (let d = 0; d < dq; d++) {
        sum += (Q[i]?.[d] ?? 0) * (K[j]?.[d] ?? 0);
      }
      scores[i][j] = Number(sum.toFixed(4));
    }
    steps.push({
      id: stepId++,
      description: `计算 Q[${i}]（${tokenLabels[i]}）与所有 Key 的点积，得到第 ${i} 行原始分数：[${scores[i].map((v) => v.toFixed(2)).join(", ")}]。未来位置（j > ${i}）将被掩码。`,
      data: { Q, K, V, scores: scores.map((r) => [...r]), causalMask },
      variables: {
        phase: "dot-product",
        currentQueryIdx: i,
        scores: scores.map((r) => [...r]),
        causalMask,
        tokenLabels,
        Q, K, V,
      },
    });
  }

  // Step 4: 缩放
  const scaledScores = scores.map((row) =>
    row.map((v) => Number((v * scale).toFixed(4)))
  );
  steps.push({
    id: stepId++,
    description: `将所有分数乘以缩放因子 1/√d_k（= ${scale.toFixed(4)}），防止点积值过大导致 Softmax 梯度消失。`,
    data: { scaledScores, scores, causalMask },
    variables: {
      phase: "scale",
      scores,
      scaledScores,
      scale,
      causalMask,
      tokenLabels,
      Q, K, V,
    },
  });

  // Step 5: 应用掩码（将未来位置设为 -∞）
  const maskedScores = scaledScores.map((row, i) =>
    row.map((v, j) => (causalMask[i][j] ? -Infinity : v))
  );
  // For display we use a large negative number instead of -Infinity
  const maskedScoresDisplay = scaledScores.map((row, i) =>
    row.map((v, j) => (causalMask[i][j] ? null : v))
  );
  steps.push({
    id: stepId++,
    description: `应用因果掩码：将上三角（未来位置）的分数替换为 -∞。Softmax 中 e^(-∞) = 0，确保模型完全忽略未来信息。这是自回归语言模型（GPT 等）的关键设计。`,
    data: { maskedScoresDisplay, causalMask },
    variables: {
      phase: "apply-mask",
      scaledScores,
      maskedScoresDisplay,
      maskedScores,
      causalMask,
      tokenLabels,
      Q, K, V,
    },
  });

  // Step 6: Softmax（每行只对未被掩码的位置计算）
  const attentionWeights = scaledScores.map((row, i) => {
    const maskIndices = new Set<number>(
      causalMask[i].map((masked, j) => (masked ? j : -1)).filter((j) => j >= 0)
    );
    return softmaxRowMasked(row, maskIndices);
  });
  steps.push({
    id: stepId++,
    description: `对掩码后的分数逐行应用 Softmax。由于上三角已置为 -∞，各行权重只在允许的位置（j ≤ i）上分配，且仍满足行和为 1（被掩码位置权重为 0）。`,
    data: { attentionWeights, maskedScoresDisplay, causalMask },
    variables: {
      phase: "softmax",
      scaledScores,
      maskedScoresDisplay,
      attentionWeights,
      causalMask,
      tokenLabels,
      Q, K, V,
    },
  });

  // Step 7: 加权求和
  const output = matMulWeightsV(attentionWeights, V);
  steps.push({
    id: stepId++,
    description: `用注意力权重对 Value 矩阵加权求和，得到最终输出（${seqLen}×${dv}）。每个位置的输出仅融合了当前及之前位置的信息，严格遵循因果顺序。`,
    data: { output, attentionWeights, causalMask },
    variables: {
      phase: "output",
      attentionWeights,
      output,
      causalMask,
      tokenLabels,
      Q, K, V,
    },
  });

  // Step 8: 完成
  steps.push({
    id: stepId++,
    description: `计算完成！因果注意力通过下三角掩码确保自回归生成的合理性——生成第 i 个 Token 时，只能利用位置 0 到 i-1 的上下文信息，不泄露未来信息，是 GPT 等解码器架构的核心机制。`,
    data: { output, attentionWeights, finished: true },
    variables: {
      phase: "complete",
      scores,
      scaledScores,
      attentionWeights,
      output,
      causalMask,
      tokenLabels,
      finished: true,
      Q, K, V,
    },
  });

  return steps;
}
