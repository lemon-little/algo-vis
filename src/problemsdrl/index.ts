import { ComponentType } from "react";

/**
 * DRL 可视化组件注册表
 * 键为题目 ID，值为懒加载的 React 组件
 * 示例：
 * import { lazy } from "react";
 * 30001: lazy(() => import("./Problemdrl30001/RLBasicsVisualizer")),
 */
export const drlVisualizerRegistry: Record<number, ComponentType> = {
  // 可视化组件将在后续迭代中逐步添加
};

export function getDrlVisualizer(id: number): ComponentType | undefined {
  return drlVisualizerRegistry[id];
}
