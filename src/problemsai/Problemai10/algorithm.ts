import { VisualizationStep } from "@/types";

function softmaxRow(row: number[]): number[] {
  const maxVal = Math.max(...row);
  const exps = row.map((v) => Math.exp(v - maxVal));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => Number((v / sum).toFixed(4)));
}

function matMulWeightsV(weights: number[][], V: number[][]): number[][] {
  const rows = weights.length;
  const dv = V[0]?.length || 0;
  const result: number[][] = [];
  for (let i = 0; i < rows; i++) {
    result[i] = new Array(dv).fill(0);
    for (let j = 0; j < V.length; j++) {
      for (let d = 0; d < dv; d++) {
        result[i][d] += weights[i][j] * (V[j]?.[d] ?? 0);
      }
    }
    result[i] = result[i].map((v) => Number(v.toFixed(4)));
  }
  return result;
}

/**
 * 交叉注意力：Query 来自 Decoder，Key/Value 来自 Encoder
 * Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
 *
 * @param Q  Decoder 的查询矩阵 [tgtLen × d_k]
 * @param K  Encoder 的键矩阵   [srcLen × d_k]
 * @param V  Encoder 的值矩阵   [srcLen × d_v]
 * @param dk  Key 向量维度
 */
export function generateCrossAttentionSteps(
  Q: number[][],
  K: number[][],
  V: number[][],
  dk: number
): VisualizationStep[] {
  const steps: VisualizationStep[] = [];
  let stepId = 0;

  const tgtLen = Q.length;
  const srcLen = K.length;
  const dq = Q[0]?.length || 0;
  const dv = V[0]?.length || 0;
  const safeDk = Math.max(dk, 1);
  const scale = 1 / Math.sqrt(safeDk);

  if (tgtLen === 0 || srcLen === 0 || dq === 0) {
    steps.push({
      id: 0,
      description: "请输入有效的 Q（Decoder）、K、V（Encoder）矩阵。",
      data: {},
      variables: { phase: "error", finished: true },
    });
    return steps;
  }

  // Step 1: 初始化
  steps.push({
    id: stepId++,
    description: `初始化：Q 来自 Decoder（${tgtLen}×${dq}），K/V 来自 Encoder（${srcLen}×${dq} / ${srcLen}×${dv}）。d_k = ${safeDk}，缩放因子 1/√d_k = ${scale.toFixed(4)}。`,
    data: { Q, K, V, dk: safeDk },
    variables: { phase: "init", Q, K, V, dk: safeDk, scale, tgtLen, srcLen },
  });

  // Step 2: 逐行计算 Q[i] · Kᵀ（每个 Decoder 位置与所有 Encoder 位置的相关性）
  const scores: number[][] = [];
  for (let i = 0; i < tgtLen; i++) {
    scores[i] = [];
    for (let j = 0; j < srcLen; j++) {
      let sum = 0;
      for (let d = 0; d < dq; d++) {
        sum += (Q[i]?.[d] ?? 0) * (K[j]?.[d] ?? 0);
      }
      scores[i][j] = Number(sum.toFixed(4));
    }
    steps.push({
      id: stepId++,
      description: `计算 Decoder 位置 ${i} 的 Query 与所有 Encoder Key 的点积，得到第 ${i} 行分数：[${scores[i].map((v) => v.toFixed(2)).join(", ")}]`,
      data: { Q, K, V, dk: safeDk, scores: scores.map((r) => [...r]) },
      variables: {
        phase: "dot-product",
        currentQueryIdx: i,
        scores: scores.map((r) => [...r]),
        Q,
        K,
        V,
      },
    });
  }

  // Step 3: 缩放
  const scaledScores = scores.map((row) =>
    row.map((v) => Number((v * scale).toFixed(4)))
  );
  steps.push({
    id: stepId++,
    description: `将所有分数乘以缩放因子 1/√d_k（= ${scale.toFixed(4)}），防止点积过大导致 Softmax 梯度消失。`,
    data: { scaledScores, scores },
    variables: {
      phase: "scale",
      scores,
      scaledScores,
      scale,
      Q,
      K,
      V,
    },
  });

  // Step 4: Softmax — 得到 Decoder 位置对 Encoder 各位置的注意力权重
  const attentionWeights = scaledScores.map((row) => softmaxRow(row));
  steps.push({
    id: stepId++,
    description: `对缩放后每行应用 Softmax，得到注意力权重矩阵（${tgtLen}×${srcLen}）。每行之和为 1，表示该 Decoder 位置对各 Encoder 位置的关注程度。`,
    data: { attentionWeights, scaledScores },
    variables: {
      phase: "softmax",
      scaledScores,
      attentionWeights,
      Q,
      K,
      V,
    },
  });

  // Step 5: 加权求和 — 用注意力权重对 Encoder Value 加权
  const output = matMulWeightsV(attentionWeights, V);
  steps.push({
    id: stepId++,
    description: `用注意力权重对 Encoder Value 矩阵加权求和，得到最终输出（${tgtLen}×${dv}）。Decoder 的每个位置都融合了 Encoder 的全局上下文信息。`,
    data: { output, attentionWeights },
    variables: {
      phase: "output",
      attentionWeights,
      output,
      Q,
      K,
      V,
    },
  });

  // Step 6: 完成
  steps.push({
    id: stepId++,
    description: `计算完成！CrossAttention(Q, K, V) = softmax(QKᵀ / √d_k) · V。交叉注意力让 Decoder 的每个位置都能动态关注 Encoder 输出，是机器翻译、摘要等 Seq2Seq 任务的核心机制。`,
    data: { output, attentionWeights, finished: true },
    variables: {
      phase: "complete",
      scores,
      scaledScores,
      attentionWeights,
      output,
      finished: true,
      Q,
      K,
      V,
    },
  });

  return steps;
}
