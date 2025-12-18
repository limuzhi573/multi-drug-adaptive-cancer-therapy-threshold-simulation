import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import time

# Set font for English/Chinese display (统一字体)
plt.rcParams["font.family"] = ["Arial", "Helvetica", "DejaVu Sans", "SimHei"]
plt.rcParams['axes.unicode_minus'] = False


class CombinedCancerModel:
    """整合版癌症治疗策略参数分析模型
    包含三个策略：
    1. AT50自适应策略（第一版原有）
    2. 暧昧2,1策略（第一版原有）
    3. D→F循环+失效回升并行策略（第二版核心）
    绘图布局：6行3列，统一x轴范围0-1500，保持风格一致性
    """

    def __init__(self, params=None):
        # 通用控制阈值
        self.control_thresholds = {
            'at50_init_upper_th': 1.0,
            'at50_init_lower_th': 0.5,
            'at50_terminate_th': 1.2,
            'cycle_upper_th': 1.0,
            'cycle_lower_th': 0.5,
            'cycle_terminate_th': 1.2,
            'failure_window': 3,
            'failure_count_threshold': 2,
            'cycle_parallel_failure_threshold': 1
        }

        # 模型基础参数
        self.base_params = {
            'growth_rates': [0.027, 0.021, 0.020, 0.015],
            'death_rates': [0.001, 0.001, 0.001, 0.001],
            'initial_cells': [0.8595, 0.0021, 0.0024, 0.0015],
            'carrying_capacity': 1.5,
            'drug_effects': [1.5, 1.2],
            't_end': 2500,
            'dt': 1,
            'b1_default': 1.0,
            'b2_default': 1.0,
            'target_b1': 1.0,
            'target_b2': 1.0,
            'b1_min': 0.9990,
            'b1_max': 1.0010,
            'b2_min': 0.9990,
            'b2_max': 1.0010,
            'param_step': 0.0001
        }

        # 合并所有参数
        self.params = params or {**self.control_thresholds, **self.base_params}
        self.time = np.arange(0, self.params['t_end'] + self.params['dt'], self.params['dt'])
        self.reset()

        # -------------------------- 策略1：AT50自适应策略结果 --------------------------
        self.at50_results = {}
        self.at50_param_sweep = {}
        self.at50_heatmap = None
        self.at50_b1_grid = None
        self.at50_b2_grid = None
        self.at50_target = None
        self.at50_optimal = None
        self.at50_worst = None
        self.at50_default = None

        # -------------------------- 策略2：暧昧2,1策略结果 --------------------------
        self.ambiguous_results = {}
        self.ambiguous_param_sweep = {}
        self.ambiguous_heatmap = None
        self.ambiguous_b1_grid = None
        self.ambiguous_b2_grid = None
        self.ambiguous_target = None
        self.ambiguous_optimal = None
        self.ambiguous_worst = None
        self.ambiguous_default = None

        # -------------------------- 策略3：D→F循环+失效回升并行策略结果 --------------------------
        self.cycle_parallel_results = {}
        self.cycle_parallel_param_sweep = {}
        self.cycle_parallel_heatmap = None
        self.cycle_parallel_b1_grid = None
        self.cycle_parallel_b2_grid = None
        self.cycle_parallel_target = None
        self.cycle_parallel_optimal = None
        self.cycle_parallel_worst = None
        self.cycle_parallel_default = None

        # 对比基准：D-F循环策略
        self.baseline_cycle_res = None

    def reset(self):
        """重置细胞状态为初始值"""
        self.x = np.array(self.params['initial_cells'], dtype=np.float32)
        self.total_cells = np.sum(self.x)
        return self.x.copy()

    def step(self, d_drug, f_drug):
        """模型一步更新（细胞状态演化）"""
        r = self.params['growth_rates']
        d = self.params['death_rates']
        K = self.params['carrying_capacity']
        a1, a2 = self.params['drug_effects']
        total = self.total_cells

        dx = np.zeros(4)
        # x1: 对D、F双敏感细胞
        dx[0] = r[0] * self.x[0] * (1 - total / K) * (1 - a1 * d_drug - a2 * f_drug) - d[0] * self.x[0]
        # x2: 仅对F敏感细胞
        dx[1] = r[1] * self.x[1] * (1 - total / K) * (1 - a2 * f_drug) - d[1] * self.x[1]
        # x3: 仅对D敏感细胞
        dx[2] = r[2] * self.x[2] * (1 - total / K) * (1 - a1 * d_drug) - d[2] * self.x[2]
        # x4: 双耐药细胞（不受药物影响）
        dx[3] = r[3] * self.x[3] * (1 - total / K) - d[3] * self.x[3]

        self.x = np.maximum(0, self.x + dx * self.params['dt'])
        self.total_cells = np.sum(self.x)
        return self.x.copy(), self.total_cells

    # -------------------------- 策略1：AT50自适应策略（第一版原有） --------------------------
    def simulate_at50_strategy(self, b1=None, b2=None):
        """Simulate AT50 adaptive strategy (dynamic threshold update)"""
        self.reset()
        num_steps = len(self.time)
        cell_terminate_th = self.params['at50_terminate_th']
        current_b1 = b1 if b1 is not None else self.params['b1_default']
        current_b2 = b2 if b2 is not None else self.params['b2_default']

        current_b1 = np.clip(current_b1, self.params['b1_min'], self.params['b1_max'])
        current_b2 = np.clip(current_b2, self.params['b2_min'], self.params['b2_max'])

        current_upper_th = self.params['at50_init_upper_th']
        current_lower_th = self.params['at50_init_lower_th']

        cells = np.zeros((num_steps, 4))
        total = np.zeros(num_steps)
        d_drugs = np.zeros(num_steps)
        f_drugs = np.zeros(num_steps)
        upper_ths = np.zeros(num_steps)
        lower_ths = np.zeros(num_steps)
        threshold_valid = np.ones(num_steps, dtype=bool)

        cells[0] = self.x
        total[0] = self.total_cells
        upper_ths[0] = current_upper_th
        lower_ths[0] = current_lower_th
        stop_idx = num_steps
        stop_reason = "Not terminated (reached max time)"
        threshold_failed = False

        for t_idx in range(1, num_steps):
            current_upper_th = max(current_upper_th, 1e-6)
            current_lower_th = max(current_lower_th, 1e-6)
            if current_lower_th >= current_upper_th and not threshold_failed:
                threshold_failed = True
                threshold_valid[t_idx:] = False
                stop_reason = f"Threshold failed (lower_th={current_lower_th:.5f} ≥ upper_th={current_upper_th:.5f})"

            if self.total_cells >= cell_terminate_th and stop_idx == num_steps:
                stop_idx = t_idx
                stop_reason = f"Cell count target reached ({self.total_cells:.3f} ≥ termination_th={cell_terminate_th})"
                break

            if not threshold_failed:
                current_upper_th = upper_ths[t_idx - 1] * current_b1
                current_lower_th = lower_ths[t_idx - 1] * current_b2
            else:
                current_upper_th = upper_ths[t_idx - 1]
                current_lower_th = lower_ths[t_idx - 1]

            upper_ths[t_idx] = current_upper_th
            lower_ths[t_idx] = current_lower_th

            prev_d = d_drugs[t_idx - 1]
            if threshold_valid[t_idx]:
                if self.total_cells >= current_upper_th:
                    current_d, current_f = 1.0, 1.0
                elif self.total_cells < current_lower_th:
                    current_d, current_f = 0.0, 0.0
                else:
                    current_d, current_f = prev_d, prev_d
            else:
                current_d, current_f = 0.0, 0.0

            d_drugs[t_idx] = current_d
            f_drugs[t_idx] = current_f
            self.step(current_d, current_f)
            cells[t_idx] = self.x
            total[t_idx] = self.total_cells

        result = {
            'time': self.time[:stop_idx],
            'cells': cells[:stop_idx],
            'total': total[:stop_idx],
            'd_drugs': d_drugs[:stop_idx],
            'f_drugs': f_drugs[:stop_idx],
            'upper_ths': upper_ths[:stop_idx],
            'lower_ths': lower_ths[:stop_idx],
            'threshold_valid': threshold_valid[:stop_idx],
            'stop_time': self.time[stop_idx - 1] if stop_idx < num_steps else self.params['t_end'],
            'stop_reason': stop_reason,
            'terminate_th': cell_terminate_th,
            'b1': current_b1,
            'b2': current_b2,
            'param_label': f'b1={current_b1:.5f},b2={current_b2:.5f}',
            'avg_n': np.mean(total[:stop_idx]),
            'threshold_failed': threshold_failed
        }

        if (b1 is not None and np.isclose(b1, self.params['target_b1']) and
                b2 is not None and np.isclose(b2, self.params['target_b2'])):
            self.at50_target = result

        if b1 is None and b2 is None:
            self.at50_results['AT50 Default Parameter Strategy'] = result
            self.at50_default = result
            self.at50_target = result

        return result

    def sweep_at50_parameters(self):
        """AT50策略参数扫描"""
        print("\n===== Starting AT50 Strategy b1-b2 Parameter Sweep =====")
        num_b1 = int((self.params['b1_max'] - self.params['b1_min']) / self.params['param_step']) + 1
        num_b2 = int((self.params['b2_max'] - self.params['b2_min']) / self.params['param_step']) + 1

        b1_values = np.linspace(self.params['b1_min'], self.params['b1_max'], num_b1)
        b2_values = np.linspace(self.params['b2_min'], self.params['b2_max'], num_b2)

        b1_values = np.round(b1_values, 5)
        b2_values = np.round(b2_values, 5)
        b1_values = np.unique(b1_values)
        b2_values = np.unique(b2_values)
        b1_values = b1_values[(b1_values >= self.params['b1_min']) & (b1_values <= self.params['b1_max'])]
        b2_values = b2_values[(b2_values >= self.params['b2_min']) & (b2_values <= self.params['b2_max'])]

        n_rows, n_cols = len(b2_values), len(b1_values)
        self.at50_heatmap = np.zeros((n_rows, n_cols))
        self.at50_b1_grid = b1_values
        self.at50_b2_grid = b2_values

        all_results = []
        total_combinations = len(b1_values) * len(b2_values)
        print(f"AT50 Strategy: Actual parameter combinations: {total_combinations}")

        start_time = time.time()
        for i, b2 in enumerate(b2_values):
            for j, b1 in enumerate(b1_values):
                res = self.simulate_at50_strategy(b1=b1, b2=b2)
                all_results.append(res)
                self.at50_heatmap[i, j] = res['stop_time']

                if (i * len(b1_values) + j + 1) % 50 == 0:
                    progress = (i * len(b1_values) + j + 1) / total_combinations * 100
                    elapsed = time.time() - start_time
                    print(f"AT50 Progress: {i * len(b1_values) + j + 1}/{total_combinations} ({progress:.1f}%)")

        self.at50_param_sweep['all_combinations'] = all_results
        self.at50_param_sweep['b1_values'] = b1_values
        self.at50_param_sweep['b2_values'] = b2_values
        self.at50_optimal = max(all_results, key=lambda x: x['stop_time'])
        self.at50_worst = min(all_results, key=lambda x: x['stop_time'])

        if self.at50_default is None:
            self.at50_default = self.simulate_at50_strategy()

        print(f"AT50 Strategy Sweep Completed. Optimal stop time: {self.at50_optimal['stop_time']:.1f}")

    # -------------------------- 策略2：暧昧2,1策略（第一版原有） --------------------------
    def simulate_ambiguous_strategy(self, b1=None, b2=None):
        """模拟暧昧2,1策略（D→F→并行）"""
        self.reset()
        num_steps = len(self.time)
        cell_terminate_th = self.params['at50_terminate_th']
        current_b1 = b1 if b1 is not None else self.params['b1_default']
        current_b2 = b2 if b2 is not None else self.params['b2_default']

        current_b1 = np.clip(current_b1, self.params['b1_min'], self.params['b1_max'])
        current_b2 = np.clip(current_b2, self.params['b2_min'], self.params['b2_max'])

        current_upper_th = self.params['at50_init_upper_th']
        current_lower_th = self.params['at50_init_lower_th']
        failure_window = self.params['failure_window']
        failure_count_threshold = self.params['failure_count_threshold']

        cells = np.zeros((num_steps, 4))
        total = np.zeros(num_steps)
        d_drugs = np.zeros(num_steps)
        f_drugs = np.zeros(num_steps)
        upper_ths = np.zeros(num_steps)
        lower_ths = np.zeros(num_steps)
        strategy_phase = np.zeros(num_steps)
        total_history = []
        failure_count = 0
        current_drug = 'D'
        is_treating = False
        use_combination = False
        first_failure_time = None
        threshold_failed = False

        cells[0] = self.x
        total[0] = self.total_cells
        upper_ths[0] = current_upper_th
        lower_ths[0] = current_lower_th
        stop_idx = num_steps
        stop_reason = "Not terminated (reached max time)"

        for t_idx in range(1, num_steps):
            current_upper_th = max(current_upper_th, 1e-6)
            current_lower_th = max(current_lower_th, 1e-6)
            if current_lower_th >= current_upper_th and not threshold_failed:
                threshold_failed = True
                stop_reason = f"Threshold failed (lower_th={current_lower_th:.5f} ≥ upper_th={current_upper_th:.5f})"
                is_treating = False
                use_combination = False
                strategy_phase[t_idx:] = -1

            if self.total_cells >= cell_terminate_th and stop_idx == num_steps:
                stop_idx = t_idx
                stop_reason = f"Cell count target reached ({self.total_cells:.3f} ≥ termination_th={cell_terminate_th})"
                break

            if not threshold_failed:
                current_upper_th = upper_ths[t_idx - 1] * current_b1
                current_lower_th = lower_ths[t_idx - 1] * current_b2

            upper_ths[t_idx] = current_upper_th
            lower_ths[t_idx] = current_lower_th

            if threshold_failed:
                current_d, current_f = 0.0, 0.0
            else:
                if not use_combination:
                    if not is_treating:
                        if self.total_cells >= current_upper_th:
                            is_treating = True
                            total_history = [self.total_cells]
                    else:
                        total_history.append(self.total_cells)
                        if len(total_history) > failure_window:
                            total_history = total_history[-failure_window:]

                        if len(total_history) == failure_window and total_history[-1] >= total_history[0]:
                            failure_count += 1
                            is_treating = False
                            if failure_count == 1:
                                first_failure_time = self.time[t_idx]
                            if failure_count >= failure_count_threshold:
                                use_combination = True
                            else:
                                current_drug = 'F' if current_drug == 'D' else 'D'
                        elif self.total_cells <= current_lower_th:
                            is_treating = False
                else:
                    if not is_treating:
                        if self.total_cells >= current_upper_th:
                            is_treating = True
                    else:
                        if self.total_cells <= current_lower_th:
                            is_treating = False

                if use_combination:
                    current_d, current_f = (1.0, 1.0) if is_treating else (0.0, 0.0)
                    strategy_phase[t_idx] = 1
                else:
                    current_d, current_f = (1.0, 0.0) if (is_treating and current_drug == 'D') else \
                        (0.0, 1.0) if (is_treating and current_drug == 'F') else (0.0, 0.0)
                    strategy_phase[t_idx] = 0

            d_drugs[t_idx] = current_d
            f_drugs[t_idx] = current_f
            self.step(current_d, current_f)
            cells[t_idx] = self.x
            total[t_idx] = self.total_cells

        result = {
            'time': self.time[:stop_idx],
            'cells': cells[:stop_idx],
            'total': total[:stop_idx],
            'd_drugs': d_drugs[:stop_idx],
            'f_drugs': f_drugs[:stop_idx],
            'upper_ths': upper_ths[:stop_idx],
            'lower_ths': lower_ths[:stop_idx],
            'strategy_phase': strategy_phase[:stop_idx],
            'stop_time': self.time[stop_idx - 1] if stop_idx < num_steps else self.params['t_end'],
            'stop_reason': stop_reason,
            'terminate_th': cell_terminate_th,
            'b1': current_b1,
            'b2': current_b2,
            'param_label': f'b1={current_b1:.5f},b2={current_b2:.5f}',
            'avg_n': np.mean(total[:stop_idx]),
            'failure_count': failure_count,
            'first_failure_time': first_failure_time,
            'use_combination': use_combination,
            'threshold_failed': threshold_failed,
            'threshold_failure_time': self.time[t_idx] if threshold_failed else None
        }

        if (b1 is not None and np.isclose(b1, self.params['target_b1']) and
                b2 is not None and np.isclose(b2, self.params['target_b2'])):
            self.ambiguous_target = result

        if b1 is None and b2 is None:
            self.ambiguous_results['Ambiguous 2,1 Strategy (Default Params)'] = result
            self.ambiguous_default = result
            self.ambiguous_target = result

        return result

    def sweep_ambiguous_parameters(self):
        """暧昧2,1策略参数扫描"""
        print("\n===== Starting Ambiguous 2,1 Strategy b1-b2 Parameter Sweep =====")
        num_b1 = int((self.params['b1_max'] - self.params['b1_min']) / self.params['param_step']) + 1
        num_b2 = int((self.params['b2_max'] - self.params['b2_min']) / self.params['param_step']) + 1

        b1_values = np.linspace(self.params['b1_min'], self.params['b1_max'], num_b1)
        b2_values = np.linspace(self.params['b2_min'], self.params['b2_max'], num_b2)

        b1_values = np.round(b1_values, 5)
        b2_values = np.round(b2_values, 5)
        b1_values = np.unique(b1_values)
        b2_values = np.unique(b2_values)
        b1_values = b1_values[(b1_values >= self.params['b1_min']) & (b1_values <= self.params['b1_max'])]
        b2_values = b2_values[(b2_values >= self.params['b2_min']) & (b2_values <= self.params['b2_max'])]

        n_rows, n_cols = len(b2_values), len(b1_values)
        self.ambiguous_heatmap = np.zeros((n_rows, n_cols))
        self.ambiguous_b1_grid = b1_values
        self.ambiguous_b2_grid = b2_values

        all_results = []
        total_combinations = len(b1_values) * len(b2_values)
        print(f"Ambiguous Strategy: Actual parameter combinations: {total_combinations}")

        start_time = time.time()
        for i, b2 in enumerate(b2_values):
            for j, b1 in enumerate(b1_values):
                res = self.simulate_ambiguous_strategy(b1=b1, b2=b2)
                all_results.append(res)
                self.ambiguous_heatmap[i, j] = res['stop_time']

                if (i * len(b1_values) + j + 1) % 50 == 0:
                    progress = (i * len(b1_values) + j + 1) / total_combinations * 100
                    elapsed = time.time() - start_time
                    print(f"Ambiguous Progress: {i * len(b1_values) + j + 1}/{total_combinations} ({progress:.1f}%)")

        self.ambiguous_param_sweep['all_combinations'] = all_results
        self.ambiguous_param_sweep['b1_values'] = b1_values
        self.ambiguous_param_sweep['b2_values'] = b2_values
        self.ambiguous_optimal = max(all_results, key=lambda x: x['stop_time'])
        self.ambiguous_worst = min(all_results, key=lambda x: x['stop_time'])

        if self.ambiguous_default is None:
            self.ambiguous_default = self.simulate_ambiguous_strategy()

        print(f"Ambiguous Strategy Sweep Completed. Optimal stop time: {self.ambiguous_optimal['stop_time']:.1f}")

    # -------------------------- 策略3：D→F循环+失效回升并行策略（第二版核心） --------------------------
    def simulate_cycle_parallel_strategy(self, b1=None, b2=None):
        """模拟核心策略：D→F→D→F循环+失效后回升上阈值并行"""
        self.reset()
        num_steps = len(self.time)
        cell_terminate_th = self.params['at50_terminate_th']
        current_b1 = b1 if b1 is not None else self.params['b1_default']
        current_b2 = b2 if b2 is not None else self.params['b2_default']

        current_b1 = np.clip(current_b1, self.params['b1_min'], self.params['b1_max'])
        current_b2 = np.clip(current_b2, self.params['b2_min'], self.params['b2_max'])

        current_upper_th = self.params['at50_init_upper_th']
        current_lower_th = self.params['at50_init_lower_th']

        cells = np.zeros((num_steps, 4))
        total = np.zeros(num_steps)
        d_drugs = np.zeros(num_steps)
        f_drugs = np.zeros(num_steps)
        upper_ths = np.zeros(num_steps)
        lower_ths = np.zeros(num_steps)
        strategy_phase = np.zeros(num_steps)
        failure_nodes = []
        switch_parallel_time = None
        threshold_failure_time = None

        failure_count = 0
        first_failure_time = None
        in_wait_switch = False
        use_parallel = False
        current_drug = 'D'
        is_treating = False
        prev_total = self.total_cells
        threshold_failed = False

        cells[0] = self.x
        total[0] = self.total_cells
        upper_ths[0] = current_upper_th
        lower_ths[0] = current_lower_th
        stop_idx = num_steps
        stop_reason = "Not terminated (reached max time)"

        for t_idx in range(1, num_steps):
            current_upper_th = max(current_upper_th, 1e-6)
            current_lower_th = max(current_lower_th, 1e-6)
            if current_lower_th >= current_upper_th:
                threshold_failed = True
                threshold_failure_time = self.time[t_idx]
                strategy_phase[t_idx:] = 5
                break

            if self.total_cells >= cell_terminate_th and stop_idx == num_steps:
                stop_idx = t_idx
                stop_reason = f"Cell count target reached ({self.total_cells:.3f} ≥ termination_th={cell_terminate_th})"
                break

            current_upper_th = upper_ths[t_idx - 1] * current_b1
            current_lower_th = lower_ths[t_idx - 1] * current_b2

            upper_ths[t_idx] = current_upper_th
            lower_ths[t_idx] = current_lower_th

            if not threshold_failed:
                if not in_wait_switch and not use_parallel:
                    if not is_treating:
                        if self.total_cells >= current_upper_th:
                            is_treating = True
                            prev_total = self.total_cells
                            strategy_phase[t_idx] = 1 if current_drug == 'D' else 2
                        else:
                            strategy_phase[t_idx] = 0
                    else:
                        if self.total_cells > prev_total:
                            failure_count += 1
                            failure_nodes.append(self.time[t_idx])
                            is_treating = False
                            in_wait_switch = True
                            if first_failure_time is None:
                                first_failure_time = self.time[t_idx]
                            strategy_phase[t_idx] = 4
                        elif self.total_cells < current_lower_th:
                            is_treating = False
                            current_drug = 'F' if current_drug == 'D' else 'D'
                            strategy_phase[t_idx] = 0
                        else:
                            strategy_phase[t_idx] = 1 if current_drug == 'D' else 2

                    if is_treating:
                        d_drugs[t_idx] = 1.0 if current_drug == 'D' else 0.0
                        f_drugs[t_idx] = 1.0 if current_drug == 'F' else 0.0
                    else:
                        d_drugs[t_idx] = 0.0
                        f_drugs[t_idx] = 0.0

                    prev_total = self.total_cells

                elif in_wait_switch and not use_parallel:
                    strategy_phase[t_idx] = 4
                    d_drugs[t_idx] = 0.0
                    f_drugs[t_idx] = 0.0

                    if self.total_cells >= current_upper_th:
                        use_parallel = True
                        in_wait_switch = False
                        switch_parallel_time = self.time[t_idx]
                        is_treating = True
                        d_drugs[t_idx] = 1.0
                        f_drugs[t_idx] = 1.0
                        strategy_phase[t_idx] = 3

                else:
                    strategy_phase[t_idx] = 3
                    if not is_treating:
                        if self.total_cells >= current_upper_th:
                            is_treating = True
                            d_drugs[t_idx] = 1.0
                            f_drugs[t_idx] = 1.0
                        else:
                            d_drugs[t_idx] = 0.0
                            f_drugs[t_idx] = 0.0
                    else:
                        if self.total_cells < current_lower_th:
                            is_treating = False
                            d_drugs[t_idx] = 0.0
                            f_drugs[t_idx] = 0.0
                        else:
                            d_drugs[t_idx] = 1.0
                            f_drugs[t_idx] = 1.0

                self.step(d_drugs[t_idx], f_drugs[t_idx])
                cells[t_idx] = self.x
                total[t_idx] = self.total_cells
            else:
                strategy_phase[t_idx] = 5
                d_drugs[t_idx] = 0.0
                f_drugs[t_idx] = 0.0

                self.step(0.0, 0.0)
                cells[t_idx] = self.x
                total[t_idx] = self.total_cells

                if self.total_cells >= cell_terminate_th and stop_idx == num_steps:
                    stop_idx = t_idx
                    stop_reason = f"No treatment after threshold failure, cell count target reached ({self.total_cells:.3f} ≥ termination_th={cell_terminate_th})"
                    break

        if threshold_failed and stop_idx == num_steps:
            for t_idx in range(t_idx + 1, num_steps):
                strategy_phase[t_idx] = 5
                d_drugs[t_idx] = 0.0
                f_drugs[t_idx] = 0.0

                self.step(0.0, 0.0)
                cells[t_idx] = self.x
                total[t_idx] = self.total_cells
                upper_ths[t_idx] = current_upper_th
                lower_ths[t_idx] = current_lower_th

                if self.total_cells >= cell_terminate_th and stop_idx == num_steps:
                    stop_idx = t_idx
                    stop_reason = f"No treatment after threshold failure, cell count target reached ({self.total_cells:.3f} ≥ termination_th={cell_terminate_th})"
                    break

        result = {
            'time': self.time[:stop_idx],
            'cells': cells[:stop_idx],
            'total': total[:stop_idx],
            'd_drugs': d_drugs[:stop_idx],
            'f_drugs': f_drugs[:stop_idx],
            'upper_ths': upper_ths[:stop_idx],
            'lower_ths': lower_ths[:stop_idx],
            'strategy_phase': strategy_phase[:stop_idx],
            'stop_time': self.time[stop_idx - 1] if stop_idx < num_steps else self.params['t_end'],
            'stop_reason': stop_reason,
            'terminate_th': cell_terminate_th,
            'b1': current_b1,
            'b2': current_b2,
            'param_label': f'b1={current_b1:.5f},b2={current_b2:.5f}',
            'avg_n': np.mean(total[:stop_idx]),
            'failure_count': failure_count,
            'first_failure_time': first_failure_time,
            'failure_nodes': failure_nodes,
            'switch_parallel_time': switch_parallel_time,
            'in_wait_switch': in_wait_switch,
            'use_parallel': use_parallel,
            'threshold_failed': threshold_failed,
            'threshold_failure_time': threshold_failure_time
        }

        if (b1 is not None and np.isclose(b1, self.params['target_b1']) and
                b2 is not None and np.isclose(b2, self.params['target_b2'])):
            self.cycle_parallel_target = result

        if b1 is None and b2 is None:
            self.cycle_parallel_results['Cycle+Recovery Combination Strategy (Default Params)'] = result
            self.cycle_parallel_default = result
            self.cycle_parallel_target = result

        return result

    def sweep_cycle_parallel_parameters(self):
        """D→F循环+失效回升并行策略参数扫描"""
        print("\n===== Starting Cycle+Recovery Combination Strategy b1-b2 Parameter Sweep =====")
        num_b1 = int((self.params['b1_max'] - self.params['b1_min']) / self.params['param_step']) + 1
        num_b2 = int((self.params['b2_max'] - self.params['b2_min']) / self.params['param_step']) + 1

        b1_values = np.linspace(self.params['b1_min'], self.params['b1_max'], num_b1)
        b2_values = np.linspace(self.params['b2_min'], self.params['b2_max'], num_b2)

        b1_values = np.round(b1_values, 5)
        b2_values = np.round(b2_values, 5)
        b1_values = np.unique(b1_values)
        b2_values = np.unique(b2_values)
        b1_values = b1_values[(b1_values >= self.params['b1_min']) & (b1_values <= self.params['b1_max'])]
        b2_values = b2_values[(b2_values >= self.params['b2_min']) & (b2_values <= self.params['b2_max'])]

        n_rows, n_cols = len(b2_values), len(b1_values)
        self.cycle_parallel_heatmap = np.zeros((n_rows, n_cols))
        self.cycle_parallel_b1_grid = b1_values
        self.cycle_parallel_b2_grid = b2_values

        all_results = []
        total_combinations = len(b1_values) * len(b2_values)
        print(f"Cycle+Recovery Strategy: Actual parameter combinations: {total_combinations}")

        start_time = time.time()
        for i, b2 in enumerate(b2_values):
            for j, b1 in enumerate(b1_values):
                res = self.simulate_cycle_parallel_strategy(b1=b1, b2=b2)
                all_results.append(res)
                self.cycle_parallel_heatmap[i, j] = res['stop_time']

                if (i * len(b1_values) + j + 1) % 50 == 0:
                    progress = (i * len(b1_values) + j + 1) / total_combinations * 100
                    elapsed = time.time() - start_time
                    print(f"Cycle+Recovery Progress: {i * len(b1_values) + j + 1}/{total_combinations} ({progress:.1f}%)")

        self.cycle_parallel_param_sweep['all_combinations'] = all_results
        self.cycle_parallel_param_sweep['b1_values'] = b1_values
        self.cycle_parallel_param_sweep['b2_values'] = b2_values
        self.cycle_parallel_optimal = max(all_results, key=lambda x: x['stop_time'])
        self.cycle_parallel_worst = min(all_results, key=lambda x: x['stop_time'])

        if self.cycle_parallel_default is None:
            self.cycle_parallel_default = self.simulate_cycle_parallel_strategy()

        print(f"Cycle+Recovery Strategy Sweep Completed. Optimal stop time: {self.cycle_parallel_optimal['stop_time']:.1f}")

    # -------------------------- 基准策略：D-F循环策略 --------------------------
    def simulate_baseline_cycle_strategy(self):
        """Simulate D-F cycle strategy (as comparison baseline)"""
        self.reset()
        num_steps = len(self.time)
        upper_th = self.params['cycle_upper_th']
        lower_th = self.params['cycle_lower_th']
        terminate_th = self.params['cycle_terminate_th']

        cells = np.zeros((num_steps, 4))
        total = np.zeros(num_steps)
        d_drugs = np.zeros(num_steps)
        f_drugs = np.zeros(num_steps)

        current_drug = 'D'
        is_treating = False
        cells[0] = self.x
        total[0] = self.total_cells
        stop_idx = num_steps

        for t_idx in range(1, num_steps):
            if self.total_cells >= terminate_th and stop_idx == num_steps:
                stop_idx = t_idx
                break

            if not is_treating:
                if self.total_cells >= upper_th:
                    is_treating = True
            else:
                if self.total_cells <= lower_th:
                    is_treating = False
                    current_drug = 'F' if current_drug == 'D' else 'D'

            if is_treating:
                current_d, current_f = (1.0, 0.0) if current_drug == 'D' else (0.0, 1.0)
            else:
                current_d, current_f = 0.0, 0.0

            d_drugs[t_idx] = current_d
            f_drugs[t_idx] = current_f
            self.step(current_d, current_f)
            cells[t_idx] = self.x
            total[t_idx] = self.total_cells

        result = {
            'time': self.time[:stop_idx],
            'cells': cells[:stop_idx],
            'total': total[:stop_idx],
            'd_drugs': d_drugs[:stop_idx],
            'f_drugs': f_drugs[:stop_idx],
            'stop_time': self.time[stop_idx - 1] if stop_idx < num_steps else self.params['t_end'],
            'terminate_th': terminate_th,
            'upper_th': upper_th,
            'lower_th': lower_th,
            'avg_n': np.mean(total[:stop_idx])
        }

        self.baseline_cycle_res = result
        return result

    # -------------------------- 通用绘图工具函数 --------------------------
    def _add_letter_annotation(self, fig, ax, letter):
        """子图左上角内部添加字母标注"""
        ax.text(0.02, 0.98, letter,
                fontsize=18, fontweight='bold', color='black',
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    def _get_color_map(self):
        """统一颜色映射"""
        return LinearSegmentedColormap.from_list('custom_cmap',
                                                 ['#4575b4', '#91bfdb', '#e0f3f8',
                                                  '#ffffbf', '#fee090', '#fc8d59', '#d73027'])

    def _plot_strategy_detail(self, fig, ax, res, title, letter, strategy_type='at50',
                              is_target=False, is_optimal=False, is_worst=False, show_legend=True):
        """绘制策略详细曲线"""
        self._add_letter_annotation(fig, ax, letter)

        cell_labels = ['x1(Dual-sensitive)', 'x2(F-sensitive only)', 'x3(D-sensitive only)', 'x4(Dual-resistant)']
        cell_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
        terminate_th = res['terminate_th']

        # 绘制细胞类型曲线
        for i in range(4):
            if len(res['time']) == len(res['cells'][:, i]):
                ax.plot(res['time'], res['cells'][:, i],
                        color=cell_colors[i], linewidth=1.2, label=cell_labels[i])

        # 绘制总细胞量曲线
        if len(res['time']) == len(res['total']):
            if is_target:
                ax.plot(res['time'], res['total'], color='black', linewidth=2.5, label='Total cells', alpha=0.9)
            elif is_optimal:
                ax.plot(res['time'], res['total'], color='#2ca02c', linewidth=2.5, label='Total cells', alpha=0.9)
            elif is_worst:
                ax.plot(res['time'], res['total'], color='#ff4444', linewidth=2.5, label='Total cells', alpha=0.9)
            else:
                ax.plot(res['time'], res['total'], color='#d62728', linewidth=1.8, label='Total cells', alpha=0.9)

        # 绘制阈值曲线
        if 'upper_ths' in res and 'lower_ths' in res:
            if 'threshold_valid' in res and strategy_type == 'at50':
                valid_mask = res['threshold_valid']
                valid_times = res['time'][valid_mask]
                valid_upper = res['upper_ths'][valid_mask]
                valid_lower = res['lower_ths'][valid_mask]
                if len(valid_times) > 0:
                    ax.plot(valid_times, valid_upper, color='#ff4444', linestyle='--', linewidth=1.5,
                            label=f'Upper threshold (b1={res["b1"]:.5f})')
                    ax.plot(valid_times, valid_lower, color='#4444ff', linestyle='--', linewidth=1.5,
                            label=f'Lower threshold (b2={res["b2"]:.5f})')
            else:
                ax.plot(res['time'], res['upper_ths'], color='#ff4444', linestyle='--', linewidth=1.5,
                        label=f'Upper threshold (b1={res.get("b1", 1.0):.5f})')
                ax.plot(res['time'], res['lower_ths'], color='#4444ff', linestyle='--', linewidth=1.5,
                        label=f'Lower threshold (b2={res.get("b2", 1.0):.5f})')

        # 绘制药物区域
        if strategy_type == 'at50':
            for t_idx, t in enumerate(res['time']):
                if t_idx < len(res['d_drugs']) and t_idx < len(res['f_drugs']):
                    d_dose = res['d_drugs'][t_idx]
                    f_dose = res['f_drugs'][t_idx]
                    if d_dose > 0:
                        ax.add_patch(Rectangle((t - 0.5, terminate_th),
                                               1, 0.1, facecolor=(255 / 255, 192 / 255, 192 / 255),
                                               alpha=1, edgecolor='none'))
                    if f_dose > 0:
                        ax.add_patch(Rectangle((t - 0.5, terminate_th + 0.1),
                                               1, 0.1, facecolor=(176 / 255, 176 / 255, 255 / 255),
                                               alpha=1, edgecolor='none'))
        elif strategy_type == 'ambiguous':
            for t_idx, t in enumerate(res['time']):
                if t_idx < len(res['d_drugs']) and t_idx < len(res['f_drugs']) and t_idx < len(res['strategy_phase']):
                    d_dose = res['d_drugs'][t_idx]
                    f_dose = res['f_drugs'][t_idx]
                    phase = res['strategy_phase'][t_idx]

                    if d_dose > 0 and f_dose == 0 and phase == 0:
                        ax.add_patch(Rectangle((t - 0.5, terminate_th),
                                               1, 0.1, facecolor=(255 / 255, 192 / 255, 192 / 255),
                                               alpha=1.0, edgecolor='none'))
                    elif f_dose > 0 and d_dose == 0 and phase == 0:
                        ax.add_patch(Rectangle((t - 0.5, terminate_th + 0.1),
                                               1, 0.1, facecolor=(176 / 255, 176 / 255, 255 / 255),
                                               alpha=1.0, edgecolor='none'))
                    elif d_dose > 0 and f_dose > 0 and phase == 1:
                        ax.add_patch(Rectangle((t - 0.5, terminate_th),
                                               1, 0.1, facecolor=(255 / 255, 192 / 255, 192 / 255),
                                               alpha=1.0, edgecolor='none'))
                        ax.add_patch(Rectangle((t - 0.5, terminate_th + 0.1),
                                               1, 0.1, facecolor=(176 / 255, 176 / 255, 255 / 255),
                                               alpha=1.0, edgecolor='none'))
                    elif phase == -1:
                        ax.add_patch(Rectangle((t - 0.5, terminate_th - 0.1),
                                               1, 0.1, facecolor=(200 / 255, 200 / 255, 200 / 255),
                                               alpha=1.0, edgecolor='none'))
        elif strategy_type == 'cycle_parallel':
            for t_idx, t in enumerate(res['time']):
                if t_idx < len(res['d_drugs']) and t_idx < len(res['f_drugs']) and t_idx < len(res['strategy_phase']):
                    phase = res['strategy_phase'][t_idx]
                    if phase == 1:  # D单药
                        ax.add_patch(Rectangle((t - 0.5, terminate_th),
                                               1, 0.1, facecolor=(255 / 255, 192 / 255, 192 / 255),
                                               alpha=1.0, edgecolor='none'))
                    elif phase == 2:  # F单药
                        ax.add_patch(Rectangle((t - 0.5, terminate_th + 0.1),
                                               1, 0.1, facecolor=(176 / 255, 176 / 255, 255 / 255),
                                               alpha=1.0, edgecolor='none'))
                    elif phase == 3:  # 双药并行
                        ax.add_patch(Rectangle((t - 0.5, terminate_th),
                                               1, 0.1, facecolor=(255 / 255, 192 / 255, 192 / 255),
                                               alpha=1.0, edgecolor='none'))
                        ax.add_patch(Rectangle((t - 0.5, terminate_th + 0.1),
                                               1, 0.1, facecolor=(176 / 255, 176 / 255, 255 / 255),
                                               alpha=1.0, edgecolor='none'))
                    elif phase == 5:  # 阈值失效后不治疗
                        ax.add_patch(Rectangle((t - 0.5, terminate_th - 0.1),
                                               1, 0.1, facecolor=(200 / 255, 200 / 255, 200 / 255),
                                               alpha=1.0, edgecolor='none'))
        elif strategy_type == 'baseline_cycle':
            for t_idx, t in enumerate(res['time']):
                if t_idx < len(res['d_drugs']) and t_idx < len(res['f_drugs']):
                    d_dose = res['d_drugs'][t_idx]
                    f_dose = res['f_drugs'][t_idx]
                    if d_dose > 0:
                        ax.add_patch(Rectangle((t - 0.5, terminate_th),
                                               1, 0.1, facecolor=(255 / 255, 192 / 255, 192 / 255),
                                               alpha=0.5, edgecolor='none'))
                    if f_dose > 0:
                        ax.add_patch(Rectangle((t - 0.5, terminate_th + 0.1),
                                               1, 0.1, facecolor=(176 / 255, 176 / 255, 255 / 255),
                                               alpha=0.5, edgecolor='none'))

        # 辅助线
        if is_target or is_optimal or is_worst:  # 最优和最差也显示终止阈值线
            ax.axhline(y=terminate_th, color='orange', linestyle=':', linewidth=1.2,
                       label=f'Termination threshold: {terminate_th}')

        # 终止时间线
        stop_time = res['stop_time']
        ax.axvline(x=stop_time, color='red', linestyle='--', linewidth=1.5)

        # 阈值失效标记
        if res.get('threshold_failed', False) and res.get('threshold_failure_time') is not None:
            failure_time = res['threshold_failure_time']
            ax.axvline(x=failure_time, color='gray', linestyle=':', linewidth=1.5,
                       label=f'Threshold failed ({failure_time:.1f})')

        # 仅显示终止时间标注
        annotate_text = f'Termination time: {stop_time:.1f}'
        ax.annotate(annotate_text,
                    xy=(0.95, 0.90), xycoords='axes fraction',
                    horizontalalignment='right', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                    fontsize=10, fontweight='bold')

        # 子图属性
        ax.set_title(title, fontsize=11, pad=15, fontweight='bold')
        ax.set_ylabel('Cell count', fontsize=10)
        ax.set_ylim(0, min(self.params['carrying_capacity'], max(res['total']) * 1.2) if len(res['total']) > 0 else 1.5)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim(0, 1500)

        # 图例 - 修改：所有关键子图（target/optimal/worst）都显示图例，仅调整位置避免重叠
        if show_legend or is_optimal or is_worst:
            if strategy_type == 'at50' or strategy_type == 'baseline_cycle':
                drug_patches = [
                    Rectangle((0, 0), 1, 1, facecolor=(255 / 255, 192 / 255, 192 / 255), alpha=1, edgecolor='none'),
                    Rectangle((0, 0), 1, 1, facecolor=(176 / 255, 176 / 255, 255 / 255), alpha=1, edgecolor='none')
                ]
                drug_labels = ['Drug D', 'Drug F']
            elif strategy_type == 'ambiguous':
                drug_patches = [
                    Rectangle((0, 0), 1, 1, facecolor=(255 / 255, 192 / 255, 192 / 255), alpha=1.0, edgecolor='none'),
                    Rectangle((0, 0), 1, 1, facecolor=(176 / 255, 176 / 255, 255 / 255), alpha=1.0, edgecolor='none'),
                    Rectangle((0, 0), 1, 1, facecolor=(200 / 255, 200 / 255, 200 / 255), alpha=1.0, edgecolor='none'),
                    Rectangle((0, 0), 1, 1, facecolor=(255/255,192/255,192/255), alpha=1.0, edgecolor='none')  # 双药并行的D部分
                ]
                drug_labels = ['Drug D (single)', 'Drug F (single)', 'No treatment (threshold failed)', 'D+F (combination)']
            elif strategy_type == 'cycle_parallel':
                drug_patches = [
                    Rectangle((0, 0), 1, 1, facecolor=(255 / 255, 192 / 255, 192 / 255), alpha=1.0, edgecolor='none'),
                    Rectangle((0, 0), 1, 1, facecolor=(176 / 255, 176 / 255, 255 / 255), alpha=1.0, edgecolor='none'),
                    Rectangle((0, 0), 1, 1, facecolor=(200 / 255, 200 / 255, 200 / 255), alpha=1.0, edgecolor='none'),
                    Rectangle((0, 0), 1, 1, facecolor=(255/255,192/255,192/255), alpha=1.0, edgecolor='none')  # 双药并行的D部分
                ]
                drug_labels = ['Drug D (single)', 'Drug F (single)', 'No treatment (threshold failed)', 'D+F (combination)']
            else:
                drug_patches = []
                drug_labels = []

            handles, labels = ax.get_legend_handles_labels()
            unique_handles = []
            unique_labels = []
            for h, l in zip(handles, labels):
                if l not in unique_labels:
                    unique_labels.append(l)
                    unique_handles.append(h)

            # 调整图例位置，避免与A/B/C子图重叠
            # 强制所有子图（包括最优参数）的图例统一在左下角
            ax.legend(handles=drug_patches + unique_handles,
                      labels=drug_labels + unique_labels,
                      loc='lower left', fontsize=7, bbox_to_anchor=(0.01, 0.01),
                      framealpha=0.9, ncol=2, columnspacing=0.5, handlelength=1.5)

        # 边框高亮
        if is_target:
            for spine in ax.spines.values():
                spine.set_linewidth(2)
                spine.set_color('black')
        elif is_optimal:
            for spine in ax.spines.values():
                spine.set_linewidth(2)
                spine.set_color('green')
        elif is_worst:
            for spine in ax.spines.values():
                spine.set_linewidth(2)
                spine.set_color('darkred')

    def _plot_heatmap(self, fig, ax, letter, strategy_type='at50'):
        """绘制热力图"""
        self._add_letter_annotation(fig, ax, letter)

        if strategy_type == 'at50':
            heatmap_data = self.at50_heatmap
            b1_grid = self.at50_b1_grid
            b2_grid = self.at50_b2_grid
            target_res = self.at50_target
            optimal_res = self.at50_optimal
            title = 'AT50 Strategy: b1-b2 Parameter Heatmap (Color=Termination Time)'
        elif strategy_type == 'ambiguous':
            heatmap_data = self.ambiguous_heatmap
            b1_grid = self.ambiguous_b1_grid
            b2_grid = self.ambiguous_b2_grid
            target_res = self.ambiguous_target
            optimal_res = self.ambiguous_optimal
            title = 'Ambiguous 2,1 Strategy: b1-b2 Parameter Heatmap (Color=Termination Time)'
        elif strategy_type == 'cycle_parallel':
            heatmap_data = self.cycle_parallel_heatmap
            b1_grid = self.cycle_parallel_b1_grid
            b2_grid = self.cycle_parallel_b2_grid
            target_res = self.cycle_parallel_target
            optimal_res = self.cycle_parallel_optimal
            title = 'Cycle+Recovery Combination Strategy: b1-b2 Parameter Heatmap (Color=Termination Time)'
        else:
            return

        cmap = self._get_color_map()
        vmin = heatmap_data.min()
        vmax = heatmap_data.max()

        im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto',
                       origin='lower', extent=[b1_grid.min() - 0.00005, b1_grid.max() + 0.00005,
                                               b2_grid.min() - 0.00005, b2_grid.max() + 0.00005])

        target_b1 = target_res['b1']
        target_b2 = target_res['b2']
        optimal_b1 = optimal_res['b1']
        optimal_b2 = optimal_res['b2']

        ax.scatter(target_b1, target_b2, color='yellow', s=120, marker='*',
                   edgecolor='black', linewidth=1.5,
                   label=f'Default/Target\n({target_b1:.5f},{target_b2:.5f})\nThreshold failed: {"Yes" if target_res["threshold_failed"] else "No"}')
        ax.scatter(optimal_b1, optimal_b2, color='white', s=100, marker='o',
                   edgecolor='green', linewidth=2,
                   label=f'Optimal\n({optimal_b1:.5f},{optimal_b2:.5f})\nThreshold failed: {"Yes" if optimal_res["threshold_failed"] else "No"}')

        cbar = plt.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label('Termination time', fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        step = max(1, len(b1_grid) // 8)
        ax.set_xticks(b1_grid[::step])
        ax.set_xticklabels([f'{x:.5f}' for x in b1_grid[::step]], rotation=45, ha='right', fontsize=8)

        step = max(1, len(b2_grid) // 8)
        ax.set_yticks(b2_grid[::step])
        ax.set_yticklabels([f'{y:.5f}' for y in b2_grid[::step]], fontsize=8)

        ax.set_title(title, fontsize=11, pad=15, fontweight='bold')
        ax.set_xlabel('Upper threshold decay param b1', fontsize=10)
        ax.set_ylabel('Lower threshold decay param b2', fontsize=10)
        ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0.)

    def _plot_stop_time_bar(self, fig, ax, letter, strategy_type='at50'):
        """绘制终止时间柱状图"""
        self._add_letter_annotation(fig, ax, letter)

        if strategy_type == 'at50':
            target_res = self.at50_target
            optimal_res = self.at50_optimal
            worst_res = self.at50_worst
            heatmap_data = self.at50_heatmap
            title = 'AT50 Strategy: Key Params Termination Time Comparison'
        elif strategy_type == 'ambiguous':
            target_res = self.ambiguous_target
            optimal_res = self.ambiguous_optimal
            worst_res = self.ambiguous_worst
            heatmap_data = self.ambiguous_heatmap
            title = 'Ambiguous 2,1 Strategy: Key Params Termination Time Comparison'
        elif strategy_type == 'cycle_parallel':
            target_res = self.cycle_parallel_target
            optimal_res = self.cycle_parallel_optimal
            worst_res = self.cycle_parallel_worst
            heatmap_data = self.cycle_parallel_heatmap
            title = 'Cycle+Recovery Combination Strategy: Key Params Termination Time Comparison'
        else:
            return

        cmap = self._get_color_map()
        vmin = heatmap_data.min()
        vmax = heatmap_data.max()
        norm = plt.Normalize(vmin, vmax)

        labels = [f'Default/Target\nparams\nThreshold failed: {"Yes" if target_res["threshold_failed"] else "No"}',
                  f'Optimal\nparams\nThreshold failed: {"Yes" if optimal_res["threshold_failed"] else "No"}',
                  f'Worst\nparams\nThreshold failed: {"Yes" if worst_res["threshold_failed"] else "No"}']
        stop_times = [target_res['stop_time'], optimal_res['stop_time'], worst_res['stop_time']]
        bar_colors = [cmap(norm(t)) for t in stop_times]

        bars = ax.bar(labels, stop_times, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.0, width=0.6)

        for bar, time_val in zip(bars, stop_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 2,
                    f'{time_val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        max_idx = np.argmax(stop_times)
        bars[max_idx].set_alpha(1.0)
        bars[max_idx].set_edgecolor('green')
        bars[max_idx].set_linewidth(2.0)
        ax.annotate('Optimal', xy=(max_idx, stop_times[max_idx]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='green', linewidth=1.0))

        min_idx = np.argmin(stop_times)
        bars[min_idx].set_alpha(1.0)
        bars[min_idx].set_edgecolor('darkred')
        bars[min_idx].set_linewidth(2.0)
        ax.annotate('Worst', xy=(min_idx, stop_times[min_idx]),
                    xytext=(0, -15), textcoords='offset points',
                    ha='center', va='top', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='darkred', linewidth=1.0))

        ax.set_title(title, fontsize=11, pad=15, fontweight='bold')
        ax.set_ylabel('Termination time', fontsize=10)
        ax.set_ylim(0, min(max(stop_times) * 1.1, 2500 * 1.05))
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.tick_params(labelsize=9)

    def _plot_all_params_comparison(self, fig, ax, letter, strategy_type='at50'):
        """绘制所有参数组合对比图"""
        self._add_letter_annotation(fig, ax, letter)

        if strategy_type == 'at50':
            all_results = self.at50_param_sweep['all_combinations']
            target_res = self.at50_target
            optimal_res = self.at50_optimal
            worst_res = self.at50_worst
            heatmap_data = self.at50_heatmap
            title = 'AT50 Strategy: All Params Total Cell Count Comparison'
        elif strategy_type == 'ambiguous':
            all_results = self.ambiguous_param_sweep['all_combinations']
            target_res = self.ambiguous_target
            optimal_res = self.ambiguous_optimal
            worst_res = self.ambiguous_worst
            heatmap_data = self.ambiguous_heatmap
            title = 'Ambiguous 2,1 Strategy: All Params Total Cell Count Comparison'
        elif strategy_type == 'cycle_parallel':
            all_results = self.cycle_parallel_param_sweep['all_combinations']
            target_res = self.cycle_parallel_target
            optimal_res = self.cycle_parallel_optimal
            worst_res = self.cycle_parallel_worst
            heatmap_data = self.cycle_parallel_heatmap
            title = 'Cycle+Recovery Combination Strategy: All Params Total Cell Count Comparison'
        else:
            return

        cmap = self._get_color_map()
        vmin = heatmap_data.min()
        vmax = heatmap_data.max()
        norm = plt.Normalize(vmin, vmax)

        ax.set_xlim(0, 1500)
        full_time = np.arange(0, 1501, 1)

        for res in all_results:
            is_key_res = (np.isclose(res['b1'], target_res['b1']) and np.isclose(res['b2'], target_res['b2'])) or \
                         (np.isclose(res['b1'], worst_res['b1']) and np.isclose(res['b2'], worst_res['b2'])) or \
                         (np.isclose(res['b1'], optimal_res['b1']) and np.isclose(res['b2'], optimal_res['b2']))
            if is_key_res:
                continue

            total_data = np.full_like(full_time, res['total'][-1], dtype=np.float32)
            valid_len = min(len(res['total']), len(full_time))
            total_data[:valid_len] = res['total'][:valid_len]

            if res.get('threshold_failed', False) and res.get('threshold_failure_time') is not None:
                failure_idx = int(min(res['threshold_failure_time'], 1500))
                if failure_idx < len(total_data):
                    total_data[failure_idx:] = res['total'][-1]

            line_color = cmap(norm(res['stop_time']))
            ax.plot(full_time, total_data, color=line_color, linewidth=0.5, alpha=0.3)

        key_results = [
            (worst_res, '#ff4444', 'dashed', 2.0, 'Worst strategy', '◀ Worst'),
            (target_res, 'black', 'solid', 2.5, 'Target params', '● Target'),
            (optimal_res, '#2ca02c', 'dashdot', 2.0, 'Optimal strategy', '▲ Optimal')
        ]

        key_handles = []
        for res, color, linestyle, linewidth, label, legend_label in key_results:
            total_data = np.full_like(full_time, res['total'][-1], dtype=np.float32)
            valid_len = min(len(res['total']), len(full_time))
            total_data[:valid_len] = res['total'][:valid_len]

            if res.get('threshold_failed', False) and res.get('threshold_failure_time') is not None:
                failure_idx = int(min(res['threshold_failure_time'], 1500))
                if failure_idx < len(total_data):
                    total_data[failure_idx:] = res['total'][-1]

            line = ax.plot(full_time, total_data,
                           color=color, linestyle=linestyle, linewidth=linewidth,
                           alpha=0.9, label=legend_label)
            key_handles.extend(line)

            end_time = min(res['stop_time'], 1500)
            end_value = res['total'][-1] if end_time >= res['stop_time'] else total_data[int(end_time)]
            ax.annotate(label, xy=(end_time, end_value),
                        xytext=(5, 5), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9),
                        fontsize=8, fontweight='bold', color=color)

            if res.get('threshold_failed', False) and res.get('threshold_failure_time') is not None:
                failure_time = min(res['threshold_failure_time'], 1500)
                ax.axvline(x=failure_time, color=color, linestyle=':', linewidth=1.5, alpha=0.7)
                ax.annotate('Threshold failed', xy=(failure_time, end_value),
                            xytext=(5, -10), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', alpha=0.8),
                            fontsize=7, color=color)

        # 终止阈值线
        ax.axhline(y=target_res['terminate_th'], color='orange', linestyle=':', linewidth=1.5,
                   label=f'Termination threshold ({target_res["terminate_th"]})')

        ax.set_title(title, fontsize=11, pad=15, fontweight='bold')
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Total cell count', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=8, loc='lower left', bbox_to_anchor=(0.01, 0.01), framealpha=0.9)

    def plot_all_strategies_comprehensive(self):
        """绘制所有策略的综合对比图（6行3列布局）- 子图填满画布"""
        plt.style.use('default')

        # 1. 关闭默认边距（关键）：rcParams控制全局子图边距为0
        plt.rcParams['figure.subplot.left'] = 0.02  # 左侧边距（按需微调，0为无）
        plt.rcParams['figure.subplot.right'] = 0.98  # 右侧边距（按需微调，1为无）
        plt.rcParams['figure.subplot.bottom'] = 0.03  # 底部边距（留少量空间放总标题）
        plt.rcParams['figure.subplot.top'] = 0.95  # 顶部边距（留少量空间放总标题）
        plt.rcParams['figure.subplot.wspace'] = 0.05  # 子图水平间距（0为无缝，按需微调）
        plt.rcParams['figure.subplot.hspace'] = 0.05  # 子图垂直间距（0为无缝，按需微调）

        # 2. 画布大小适配：保持24x24，确保子图密度合适
        fig = plt.figure(figsize=(24, 24))

        # -------------------------- 第1行：策略1（AT50）详细曲线 --------------------------
        ax1 = plt.subplot(4, 3, 1)
        self._plot_strategy_detail(fig, ax1, self.at50_target,
                                   'AT50 Strategy (Default/Target Params)',
                                   'A', strategy_type='at50', is_target=True, show_legend=True)

        # -------------------------- 第2行：策略2（暧昧2,1）详细曲线 --------------------------
        ax4 = plt.subplot(4, 3, 2)
        self._plot_strategy_detail(fig, ax4, self.ambiguous_target,
                                   'Ambiguous 2,1 Strategy (Default/Target Params)',
                                   'B', strategy_type='ambiguous', is_target=True, show_legend=True)

        # -------------------------- 第3行：策略3（D→F循环+失效回升）详细曲线 --------------------------
        ax7 = plt.subplot(4, 3, 3)
        self._plot_strategy_detail(fig, ax7, self.cycle_parallel_target,
                                   'Cycle+Recovery Combination Strategy (Default/Target Params)',
                                   'C', strategy_type='cycle_parallel', is_target=True, show_legend=True)

        ax2 = plt.subplot(4, 3, 4)
        self._plot_strategy_detail(fig, ax2, self.at50_optimal,
                                   'AT50 Strategy (Optimal Params)',
                                   'D', strategy_type='at50', is_optimal=True, show_legend=True)  # 修改为show_legend=True

        ax5 = plt.subplot(4, 3, 5)
        self._plot_strategy_detail(fig, ax5, self.ambiguous_optimal,
                                   'Ambiguous 2,1 Strategy (Optimal Params)',
                                   'E', strategy_type='ambiguous', is_optimal=True,
                                   show_legend=True)  # 修改为show_legend=True

        ax8 = plt.subplot(4, 3, 6)
        self._plot_strategy_detail(fig, ax8, self.cycle_parallel_optimal,
                                   'Cycle+Recovery Combination Strategy (Optimal Params)',
                                   'F', strategy_type='cycle_parallel', is_optimal=True,
                                   show_legend=True)  # 修改为show_legend=True

        # -------------------------- 第4行：热力图对比 --------------------------
        ax10 = plt.subplot(4, 3, 7)
        self._plot_heatmap(fig, ax10, 'G', strategy_type='at50')

        ax11 = plt.subplot(4, 3, 8)
        self._plot_heatmap(fig, ax11, 'H', strategy_type='ambiguous')

        ax12 = plt.subplot(4, 3, 9)
        self._plot_heatmap(fig, ax12, 'I', strategy_type='cycle_parallel')

        # -------------------------- 第6行：所有参数组合对比图 --------------------------
        ax16 = plt.subplot(4, 3, 10)
        self._plot_all_params_comparison(fig, ax16, 'J', strategy_type='at50')

        ax17 = plt.subplot(4, 3, 11)
        self._plot_all_params_comparison(fig, ax17, 'K', strategy_type='ambiguous')

        ax18 = plt.subplot(4, 3, 12)
        self._plot_all_params_comparison(fig, ax18, 'M', strategy_type='cycle_parallel')

        # 3. 关键调整：禁用tight_layout（会自动留边），仅用subplots_adjust精调
        # plt.tight_layout()  # 注释掉！否则会覆盖rcParams的边距设置
        plt.subplots_adjust(
            top=0.95,  # 与rcParams一致，确保总标题不遮挡子图
            bottom=0.03,  # 底部留少量空间，避免子图被截断
            left=0.02,  # 左侧留少量空间，避免y轴标签被截断
            right=0.98,  # 右侧留少量空间，避免colorbar被截断
            hspace=0.05,  # 垂直间距（可改为0实现无缝，需确保子图标题不重叠）
            wspace=0.05  # 水平间距（可改为0实现无缝，需确保子图标题不重叠）
        )

        # 4. 保存时移除多余空白：bbox_inches='tight'改为'None'，配合facecolor='white'
        plt.savefig(
            'cancer_treatment_strategies_comprehensive_analysis.png',
            dpi=300,
            bbox_inches=None,  # 关键：不裁剪画布，保留完整布局
            facecolor='white',
            pad_inches=0.0  # 额外padding设为0
        )
        plt.show()

        # 重置rcParams（可选，避免影响后续绘图）
        plt.rcParams.update(plt.rcParamsDefault)

    def run_full_analysis(self):
        """运行完整分析流程"""
        print("===== Starting Full Cancer Treatment Strategy Analysis =====")
        start_total = time.time()

        # 1. 运行基准策略
        print("\n1. Simulating Baseline D-F Cycle Strategy...")
        self.simulate_baseline_cycle_strategy()
        print(f"Baseline Strategy Termination Time: {self.baseline_cycle_res['stop_time']:.1f}")

        # 2. 运行策略1：AT50自适应策略
        print("\n2. Running AT50 Strategy Analysis...")
        self.simulate_at50_strategy()  # 默认参数模拟
        self.sweep_at50_parameters()  # 参数扫描

        # 3. 运行策略2：暧昧2,1策略
        print("\n3. Running Ambiguous 2,1 Strategy Analysis...")
        self.simulate_ambiguous_strategy()  # 默认参数模拟
        self.sweep_ambiguous_parameters()  # 参数扫描

        # 4. 运行策略3：D→F循环+失效回升并行策略
        print("\n4. Running Cycle+Recovery Combination Strategy Analysis...")
        self.simulate_cycle_parallel_strategy()  # 默认参数模拟
        self.sweep_cycle_parallel_parameters()  # 参数扫描

        # 5. 生成综合对比图
        print("\n5. Generating Comprehensive Analysis Plots...")
        self.plot_all_strategies_comprehensive()

        total_elapsed = time.time() - start_total
        print(f"\n===== Full Analysis Completed! Total Time Elapsed: {total_elapsed:.2f}s =====")

        # 输出核心结果汇总
        print("\n===== Core Results Summary =====")
        strategies = [
            ('AT50 Adaptive Strategy', self.at50_default, self.at50_optimal, self.at50_worst),
            ('Ambiguous 2,1 Strategy', self.ambiguous_default, self.ambiguous_optimal, self.ambiguous_worst),
            ('Cycle+Recovery Combination Strategy', self.cycle_parallel_default, self.cycle_parallel_optimal,
             self.cycle_parallel_worst),
            ('Baseline D-F Cycle Strategy', self.baseline_cycle_res, self.baseline_cycle_res, self.baseline_cycle_res)
        ]

        for name, default, optimal, worst in strategies:
            print(f"\n{name}:")
            print(f"  - Default Params Termination Time: {default['stop_time']:.1f}")
            print(f"  - Optimal Params Termination Time: {optimal['stop_time']:.1f}")
            print(f"  - Worst Params Termination Time: {worst['stop_time']:.1f}")
            print(f"  - Average Cell Count (Default): {default['avg_n']:.4f}")


if __name__ == "__main__":
    # 初始化模型并运行完整分析
    model = CombinedCancerModel()
    model.run_full_analysis()
