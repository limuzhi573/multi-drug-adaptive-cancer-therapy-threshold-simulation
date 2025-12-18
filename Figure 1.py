import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import itertools
import time

# Font settings (use default English-supported fonts)
plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams['axes.unicode_minus'] = False


class CancerModel:
    """Cancer Drug Strategy Model, Layout Order: Strategy 4 (Baseline) → Strategy 1 → Strategy 2 → Strategy 3 → Comparison Subplots"""

    def __init__(self, params=None):
        # Control thresholds (default values) - upper threshold renamed to a1, lower threshold renamed to a2
        self.control_thresholds = {
            'a1': 1.0,  # Upper threshold (start treatment)
            'a2': 0.5,  # Lower threshold (stop treatment)
            'terminate_threshold': 1.2,  # Simulation termination threshold
            'failure_window': 3,  # Determine failure if no suppression for N consecutive steps
            'failure_count_threshold': 2  # Failure count threshold
        }

        # Other model parameters
        self.base_params = {
            'growth_rates': [0.027, 0.021, 0.020, 0.015],
            'death_rates': [0.001, 0.001, 0.001, 0.001],
            'initial_cells': [0.8595, 0.0021, 0.0024, 0.0015],
            'carrying_capacity': 1.5,
            'drug_effects': [1.5, 1.2],
            't_end': 2500,
            'dt': 1,
        }

        # Merge all parameters
        self.params = params or {**self.control_thresholds, **self.base_params}
        self.time = np.arange(0, self.params['t_end'] + self.params['dt'], self.params['dt'])
        self.reset()
        self.results = {}
        self.threshold_results = {}  # Store threshold variation results
        self.optimal_strategies = {}  # Store optimal plan for each strategy

    def reset(self):
        """Reset cell state to initial values"""
        self.x = np.array(self.params['initial_cells'], dtype=np.float32)
        self.total_cells = np.sum(self.x)
        return self.x.copy()

    def step(self, d_drug, f_drug):
        """Model one-step update"""
        r = self.params['growth_rates']
        d = self.params['death_rates']
        K = self.params['carrying_capacity']
        a1_drug, a2_drug = self.params[
            'drug_effects']  # Rename drug effect parameters to avoid conflict with thresholds a1/a2
        total = self.total_cells

        dx = np.zeros(4)
        dx[0] = r[0] * self.x[0] * (1 - total / K) * (1 - a1_drug * d_drug - a2_drug * f_drug) - d[0] * self.x[0]
        dx[1] = r[1] * self.x[1] * (1 - total / K) * (1 - a2_drug * f_drug) - d[1] * self.x[1]
        dx[2] = r[2] * self.x[2] * (1 - total / K) * (1 - a1_drug * d_drug) - d[2] * self.x[2]
        dx[3] = r[3] * self.x[3] * (1 - total / K) - d[3] * self.x[3]

        self.x = np.maximum(0, self.x + dx * self.params['dt'])
        self.total_cells = np.sum(self.x)
        return self.x.copy(), self.total_cells

    # -------------------------- Strategy Simulation Functions --------------------------
    def simulate_d_to_f_to_combination_strategy(self, a1=None, a2=None):
        self.reset()
        a1_val = a1 if a1 is not None else self.params['a1']  # Upper threshold a1
        a2_val = a2 if a2 is not None else self.params['a2']  # Lower threshold a2
        terminate_th = self.params['terminate_threshold']
        failure_window = self.params['failure_window']
        failure_count_threshold = self.params['failure_count_threshold']
        num_steps = len(self.time)

        cells = np.zeros((num_steps, 4))
        total = np.zeros(num_steps)
        d_drugs = np.zeros(num_steps)
        f_drugs = np.zeros(num_steps)
        strategy_phase = np.zeros(num_steps)
        total_history = []
        failure_count = 0
        current_drug = 'D'
        is_treating = False
        use_combination = False
        first_failure_time = None
        second_failure_time = None  # New: Record second failure time
        cells[0] = self.x
        total[0] = self.total_cells
        stop_idx = num_steps

        for t_idx in range(1, num_steps):
            if self.total_cells >= terminate_th and stop_idx == num_steps:
                stop_idx = t_idx
                break

            if not use_combination:
                if not is_treating:
                    if self.total_cells >= a1_val:  # Upper threshold a1: Treatment initiation condition
                        is_treating = True
                        total_history = [self.total_cells]
                else:
                    total_history.append(self.total_cells)
                    if len(total_history) > failure_window:
                        total_history = total_history[-failure_window:]
                    if len(total_history) == failure_window and total_history[-1] >= total_history[0]:
                        failure_count += 1
                        is_treating = False
                        # Record first and second failure times
                        if failure_count == 1:
                            first_failure_time = self.time[t_idx]
                        elif failure_count == 2:
                            second_failure_time = self.time[t_idx]
                        if failure_count >= failure_count_threshold:
                            use_combination = True
                        else:
                            current_drug = 'F' if current_drug == 'D' else 'D'
                        total_history = []  # Reset history to avoid duplicate counting
                    elif self.total_cells <= a2_val:  # Lower threshold a2: Treatment termination condition
                        is_treating = False
            else:
                if not is_treating:
                    if self.total_cells >= a1_val:  # Upper threshold a1: Combination treatment initiation condition
                        is_treating = True
                else:
                    if self.total_cells <= a2_val:  # Lower threshold a2: Combination treatment termination condition
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
            'time': self.time[:stop_idx], 'cells': cells[:stop_idx], 'total': total[:stop_idx],
            'd_drugs': d_drugs[:stop_idx], 'f_drugs': f_drugs[:stop_idx],
            'strategy_phase': strategy_phase[:stop_idx],
            'stop_time': self.time[stop_idx - 1] if stop_idx < num_steps else self.params['t_end'],
            'failure_count': failure_count,
            'first_failure_time': first_failure_time,
            'second_failure_time': second_failure_time,  # New: Return second failure time
            'a1': a1_val, 'a2': a2_val  # Store upper threshold a1 and lower threshold a2 used in current strategy
        }

        if a1 is None and a2 is None:
            self.results['Strategy 1: D→F→Combination'] = result
        return result

    def simulate_f_to_d_to_combination_strategy(self, a1=None, a2=None):
        self.reset()
        a1_val = a1 if a1 is not None else self.params['a1']  # Upper threshold a1
        a2_val = a2 if a2 is not None else self.params['a2']  # Lower threshold a2
        terminate_th = self.params['terminate_threshold']
        failure_window = self.params['failure_window']
        failure_count_threshold = self.params['failure_count_threshold']
        num_steps = len(self.time)

        cells = np.zeros((num_steps, 4))
        total = np.zeros(num_steps)
        d_drugs = np.zeros(num_steps)
        f_drugs = np.zeros(num_steps)
        strategy_phase = np.zeros(num_steps)
        total_history = []
        failure_count = 0
        current_drug = 'F'
        is_treating = False
        use_combination = False
        first_failure_time = None
        second_failure_time = None  # New: Record second failure time
        cells[0] = self.x
        total[0] = self.total_cells
        stop_idx = num_steps

        for t_idx in range(1, num_steps):
            if self.total_cells >= terminate_th and stop_idx == num_steps:
                stop_idx = t_idx
                break

            if not use_combination:
                if not is_treating:
                    if self.total_cells >= a1_val:  # Upper threshold a1: Treatment initiation condition
                        is_treating = True
                        total_history = [self.total_cells]
                else:
                    total_history.append(self.total_cells)
                    if len(total_history) > failure_window:
                        total_history = total_history[-failure_window:]
                    if len(total_history) == failure_window and total_history[-1] >= total_history[0]:
                        failure_count += 1
                        is_treating = False
                        # Record first and second failure times
                        if failure_count == 1:
                            first_failure_time = self.time[t_idx]
                        elif failure_count == 2:
                            second_failure_time = self.time[t_idx]
                        if failure_count >= failure_count_threshold:
                            use_combination = True
                        else:
                            current_drug = 'D' if current_drug == 'F' else 'F'
                        total_history = []  # Reset history to avoid duplicate counting
                    elif self.total_cells <= a2_val:  # Lower threshold a2: Treatment termination condition
                        is_treating = False
            else:
                if not is_treating:
                    if self.total_cells >= a1_val:  # Upper threshold a1: Combination treatment initiation condition
                        is_treating = True
                else:
                    if self.total_cells <= a2_val:  # Lower threshold a2: Combination treatment termination condition
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
            'time': self.time[:stop_idx], 'cells': cells[:stop_idx], 'total': total[:stop_idx],
            'd_drugs': d_drugs[:stop_idx], 'f_drugs': f_drugs[:stop_idx],
            'strategy_phase': strategy_phase[:stop_idx],
            'stop_time': self.time[stop_idx - 1] if stop_idx < num_steps else self.params['t_end'],
            'failure_count': failure_count,
            'first_failure_time': first_failure_time,
            'second_failure_time': second_failure_time,  # New: Return second failure time
            'a1': a1_val, 'a2': a2_val  # Store upper threshold a1 and lower threshold a2 used in current strategy
        }

        if a1 is None and a2 is None:
            self.results['Strategy 2: F→D→Combination'] = result
        return result

    def simulate_adaptive_strategy(self, a1=None, a2=None):
        self.reset()
        a1_val = a1 if a1 is not None else self.params['a1']  # Upper threshold a1
        a2_val = a2 if a2 is not None else self.params['a2']  # Lower threshold a2
        terminate_th = self.params['terminate_threshold']
        failure_window = self.params['failure_window']
        num_steps = len(self.time)

        cells = np.zeros((num_steps, 4))
        total = np.zeros(num_steps)
        d_drugs = np.zeros(num_steps)
        f_drugs = np.zeros(num_steps)
        strategy_phase = np.zeros(num_steps)
        total_history = []
        current_drug = 'D'
        is_treating = False
        use_combination = False
        failure_count = 0
        failure_times = []  # Store first two failure times
        cells[0] = self.x
        total[0] = self.total_cells
        stop_idx = num_steps

        for t_idx in range(1, num_steps):
            if self.total_cells >= terminate_th and stop_idx == num_steps:
                stop_idx = t_idx
                break

            if not use_combination:
                if not is_treating:
                    if self.total_cells >= a1_val:  # Upper threshold a1: Treatment initiation condition
                        is_treating = True
                        total_history = [self.total_cells]
                else:
                    total_history.append(self.total_cells)
                    if len(total_history) > failure_window:
                        total_history = total_history[-failure_window:]
                    if len(total_history) == failure_window and total_history[-1] >= total_history[0]:
                        failure_count += 1
                        is_treating = False
                        if len(failure_times) < 2:  # Only record first two failures
                            failure_times.append(self.time[t_idx])
                        if failure_count == 1:
                            current_drug = 'F'
                        elif failure_count >= 2:
                            use_combination = True
                        total_history = []  # Reset history to avoid duplicate counting
                    elif self.total_cells <= a2_val:  # Lower threshold a2: Treatment termination condition
                        is_treating = False
                        current_drug = 'F' if current_drug == 'D' else 'D'
            else:
                if not is_treating:
                    if self.total_cells >= a1_val:  # Upper threshold a1: Combination treatment initiation condition
                        is_treating = True
                else:
                    if self.total_cells <= a2_val:  # Lower threshold a2: Combination treatment termination condition
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
            'time': self.time[:stop_idx], 'cells': cells[:stop_idx], 'total': total[:stop_idx],
            'd_drugs': d_drugs[:stop_idx], 'f_drugs': f_drugs[:stop_idx],
            'strategy_phase': strategy_phase[:stop_idx],
            'stop_time': self.time[stop_idx - 1] if stop_idx < num_steps else self.params['t_end'],
            'failure_count': failure_count,
            'failure_times': failure_times,
            'a1': a1_val, 'a2': a2_val  # Store upper threshold a1 and lower threshold a2 used in current strategy
        }

        if a1 is None and a2 is None:
            self.results['Strategy 3: Adaptive Single→Combination'] = result
        return result

    def simulate_at50_strategy(self, a1=None, a2=None):
        self.reset()
        a1_val = a1 if a1 is not None else self.params['a1']  # Upper threshold a1
        a2_val = a2 if a2 is not None else self.params['a2']  # Lower threshold a2
        terminate_th = self.params['terminate_threshold']
        failure_window = self.params['failure_window']  # Failure window (3 steps)
        num_steps = len(self.time)

        cells = np.zeros((num_steps, 4))
        total = np.zeros(num_steps)
        d_drugs = np.zeros(num_steps)
        f_drugs = np.zeros(num_steps)
        cells[0] = self.x
        total[0] = self.total_cells
        stop_idx = num_steps

        # New: Record first failure time (consistent with other strategies' failure logic)
        total_history = []
        failure_count = 0
        first_failure_time = None

        for t_idx in range(1, num_steps):
            if self.total_cells >= terminate_th and stop_idx == num_steps:
                stop_idx = t_idx
                break

            prev_d = d_drugs[t_idx - 1]
            if self.total_cells >= a1_val:  # Upper threshold a1: Treatment initiation condition
                current_d, current_f = 1.0, 1.0
            elif self.total_cells < a2_val:  # Lower threshold a2: Treatment termination condition
                current_d, current_f = 0.0, 0.0
            else:
                current_d, current_f = prev_d, prev_d

            d_drugs[t_idx] = current_d
            f_drugs[t_idx] = current_f
            self.step(current_d, current_f)
            cells[t_idx] = self.x
            total[t_idx] = self.total_cells

            # Failure judgment logic (judge as failure if total cells do not decrease for 3 consecutive steps)
            total_history.append(self.total_cells)
            if len(total_history) > failure_window:
                total_history = total_history[-failure_window:]
            if len(total_history) == failure_window and total_history[-1] >= total_history[0]:
                failure_count += 1
                if failure_count == 1:
                    first_failure_time = self.time[t_idx]
                total_history = []  # Reset history to avoid duplicate counting

        result = {
            'time': self.time[:stop_idx], 'cells': cells[:stop_idx], 'total': total[:stop_idx],
            'd_drugs': d_drugs[:stop_idx], 'f_drugs': f_drugs[:stop_idx],
            'stop_time': self.time[stop_idx - 1] if stop_idx < num_steps else self.params['t_end'],
            'first_failure_time': first_failure_time,  # New: Return first failure time
            'a1': a1_val, 'a2': a2_val  # Store upper threshold a1 and lower threshold a2 used in current strategy
        }

        if a1 is None and a2 is None:
            self.results['Strategy 4: Original AT50 Strategy'] = result
        return result

    # -------------------------- Threshold Variation Impact Analysis --------------------------
    def analyze_threshold_effects(self):
        a1_range = np.round(np.arange(0.6, 1.11, 0.01), 2)  # Upper threshold a1 range: 0.6~1.1, step 0.01
        a2_range = np.round(np.arange(0.1, 0.51, 0.01), 2)  # Lower threshold a2 range: 0.1~0.5, step 0.01
        threshold_combinations = list(itertools.product(a1_range, a2_range))  # All combinations of a1 and a2
        total_combinations = len(threshold_combinations)
        print(
            f"Total threshold combinations: {total_combinations} (a1 ranges: {len(a1_range)}, a2 ranges: {len(a2_range)})")

        strategies = {
            'Strategy 1: D→F→Combination': self.simulate_d_to_f_to_combination_strategy,
            'Strategy 2: F→D→Combination': self.simulate_f_to_d_to_combination_strategy,
            'Strategy 3: Adaptive Single→Combination': self.simulate_adaptive_strategy,
            'Strategy 4: Original AT50 Strategy': self.simulate_at50_strategy
        }

        for strategy_name, sim_func in strategies.items():
            print(f"\n===== Starting analysis for {strategy_name} =====")
            start_time = time.time()
            self.threshold_results[strategy_name] = []

            for i, (a1_val, a2_val) in enumerate(threshold_combinations):
                if i % 100 == 0:
                    progress = (i / total_combinations) * 100
                    elapsed = time.time() - start_time
                    if i == 0:
                        est_remaining = 0.0
                    else:
                        est_remaining = (elapsed / i) * (total_combinations - i)
                    print(
                        f"Progress: {i}/{total_combinations} ({progress:.1f}%) | Elapsed: {elapsed:.1f}s | Estimated remaining: {est_remaining:.1f}s")

                result = sim_func(a1=a1_val, a2=a2_val)  # Pass a1 and a2 parameters
                self.threshold_results[strategy_name].append({
                    'a1': a1_val,  # Store upper threshold a1
                    'a2': a2_val,  # Store lower threshold a2
                    'result': result
                })

            max_stop_time = -1
            optimal_result = None
            for item in self.threshold_results[strategy_name]:
                if item['result']['stop_time'] > max_stop_time:
                    max_stop_time = item['result']['stop_time']
                    optimal_result = item
            self.optimal_strategies[strategy_name] = optimal_result

            total_time = time.time() - start_time
            print(
                f"{strategy_name} analysis completed, time consumed: {total_time:.2f} seconds, optimal stop time: {max_stop_time:.1f}, optimal a1: {optimal_result['a1']:.2f}, optimal a2: {optimal_result['a2']:.2f}")

    # -------------------------- Comparison Subplot Drawing Functions --------------------------
    def _plot_optimal_strategies_comparison(self, ax, label_letter):
        # Add subplot label (inside subplot)
        self._add_subplot_label(ax, label_letter)

        # Fixed strategy style mapping: strictly correspond to ADGJ subplots
        strategy_styles = {
            'Strategy 1 (AT50)': {'color': '#1f77b4', 'linestyle': '-', 'linewidth': 2, 'marker': 'o', 'markevery': 50},
            'Strategy 2 (D→F Combination)': {'color': '#2ca02c', 'linestyle': '--', 'linewidth': 2, 'marker': 's',
                                             'markevery': 50},
            'Strategy 3 (F→D Combination)': {'color': '#ff7f0e', 'linestyle': '-.', 'linewidth': 2, 'marker': '^',
                                             'markevery': 50},
            'Strategy 4 (D→F Cyclic Combination)': {'color': '#9467bd', 'linestyle': ':', 'linewidth': 2, 'marker': 'd',
                                                    'markevery': 50}
        }

        # Fixed full strategy name mapping: strictly correspond to ADGJ subplots
        strategy_full_names = {
            'Strategy 1 (AT50)': 'Strategy 4: Original AT50 Strategy',
            'Strategy 2 (D→F Combination)': 'Strategy 1: D→F→Combination',
            'Strategy 3 (F→D Combination)': 'Strategy 2: F→D→Combination',
            'Strategy 4 (D→F Cyclic Combination)': 'Strategy 3: Adaptive Single→Combination'
        }

        terminate_th = self.params['terminate_threshold']

        for short_name, style in strategy_styles.items():
            full_name = strategy_full_names[short_name]
            optimal = self.optimal_strategies.get(full_name)
            if optimal:
                res = optimal['result']
                # Use simplified legend name after correction
                ax.plot(res['time'], res['total'], label=short_name, **style)
                # Keep scatter marker for stop point
                ax.scatter(res['stop_time'], terminate_th, color=style['color'], s=100, zorder=5)

        ax.axhline(y=terminate_th, color='orange', linestyle=':', linewidth=1.5,
                   label=f'Termination threshold: {terminate_th}')
        # Subtitle is English translation "Comparison of Optimal Strategies"
        ax.set_title('Comparison of Optimal Strategies', fontsize=15, pad=15, fontweight='bold')
        ax.set_xlabel('Time (days)', fontsize=12, labelpad=8)
        ax.set_ylabel('Cell Quantity', fontsize=12, labelpad=8)
        ax.set_ylim(0, self.params['carrying_capacity'])
        ax.grid(True, linestyle='--', alpha=0.5)
        # Move legend to the right to avoid overlapping with letters
        ax.legend(loc='upper left', fontsize=10, bbox_to_anchor=(0.1, 1))

    def _plot_survival_time_bar(self, ax, label_letter):
        # Add subplot label (inside subplot)
        self._add_subplot_label(ax, label_letter)

        # Key fix: x-axis labels are exactly consistent with N subplot annotations
        strategy_names = ['Strategy 1 (AT50)', 'Strategy 2 (D→F Combination)', 'Strategy 3 (F→D Combination)',
                          'Strategy 4 (D→F Cyclic Combination)']
        optimal_survival_times = []  # Optimal strategy survival time
        original_survival_times = []  # Original strategy (default threshold) survival time
        optimal_a1 = []  # Optimal a1 value
        optimal_a2 = []  # Optimal a2 value
        original_a1 = self.params['a1']  # Default a1
        original_a2 = self.params['a2']  # Default a2

        # Two main color schemes: blue for original strategy, green for optimal strategy (clear distinction)
        color_original = '#4682B4'  # Steel blue (unified color for original strategy)
        color_optimal = '#2E8B57'  # Sea green (unified color for optimal strategy)

        # Key fix: strategy order strictly corresponds to N subplot
        strategy_order = [
            'Strategy 4: Original AT50 Strategy',  # Strategy 1 (AT50)
            'Strategy 1: D→F→Combination',  # Strategy 2 (D→F Combination)
            'Strategy 2: F→D→Combination',  # Strategy 3 (F→D Combination)
            'Strategy 3: Adaptive Single→Combination'  # Strategy 4 (D→F Cyclic Combination)
        ]

        for name in strategy_order:
            # Get optimal strategy survival time and optimal a1/a2
            optimal = self.optimal_strategies.get(name)
            if optimal:
                optimal_survival_times.append(optimal['result']['stop_time'])
                optimal_a1.append(optimal['a1'])
                optimal_a2.append(optimal['a2'])
            else:
                optimal_survival_times.append(0)
                optimal_a1.append(0)
                optimal_a2.append(0)

            # Get original strategy (default a1/a2) survival time
            original = self.results.get(name)
            if original:
                original_survival_times.append(original['stop_time'])
            else:
                original_survival_times.append(0)

        # Draw grouped bar chart
        x = np.arange(len(strategy_names))
        width = 0.35  # Bar width

        bars1 = ax.bar(x - width / 2, original_survival_times, width,
                       label=f'Original (a1={original_a1:.2f}, a2={original_a2:.2f})',
                       color=color_original, alpha=0.8, edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x + width / 2, optimal_survival_times, width, label='Optimal (a1/a2 optimized)',
                       color=color_optimal, alpha=0.8, edgecolor='black', linewidth=1.2)

        # Add value labels to each bar (including survival time and optimal a1/a2)
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            if height > 0:  # Only display label when there is valid data
                ax.text(bar.get_x() + bar.get_width() / 2., height + 5,
                        f'{height:.1f}d', ha='center', va='bottom', fontsize=8)

        for i, bar in enumerate(bars2):
            height = bar.get_height()
            if height > 0:  # Only display label when there is valid data
                ax.text(bar.get_x() + bar.get_width() / 2., height + 5,
                        f'{height:.1f}d\n(a1={optimal_a1[i]:.2f}, a2={optimal_a2[i]:.2f})',
                        ha='center', va='bottom', fontsize=7)

        # Modify title to "Strategy Benefit Comparison"
        ax.set_title('Strategy Benefit Comparison', fontsize=15, pad=15, fontweight='bold')
        ax.set_ylabel('Survival Time (days)', fontsize=12, labelpad=8)
        ax.set_xlabel('Strategy (Consistent with Top Subplots)', fontsize=12, labelpad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(strategy_names, fontsize=10)

        # Adjust y-axis range to ensure labels do not overflow
        max_height = max(max(optimal_survival_times), max(original_survival_times))
        ax.set_ylim(0, max_height * 1.3)

        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        # Move legend to the right
        ax.legend(loc='upper left', fontsize=10, bbox_to_anchor=(0.1, 1))

    # -------------------------- Unified Legend Drawing Function (Fixed: No overflow) --------------------------
    def _plot_unified_legend(self, ax, label_letter):
        # Add subplot label (inside subplot)
        self._add_subplot_label(ax, label_letter)

        ax.axis('off')

        # 1. Fixed strategy description text: strictly correspond to ADGJ subplots
        strategy_labels = [
            'Strategy 1: AT50 (Baseline)',
            'Strategy 2: D→F→Combination',
            'Strategy 3: F→D→Combination',
            'Strategy 4: D→F Cyclic Combination (Adaptive Single→Combination)'
        ]

        # 2. Cell type legend (unchanged)
        cell_labels = ['x1(Dual-sensitive)', 'x2(F-only sensitive)', 'x3(D-only sensitive)', 'x4(Dual-resistant)']
        cell_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']

        # 3. Threshold legend (updated to a1/a2)
        other_labels = ['Total Cell Quantity', 'Termination Threshold', 'Upper Threshold (a1)', 'Lower Threshold (a2)',
                        'First Failure (0.5 opacity)', 'Second Failure (1 opacity)', 'Termination Point (Purple)']
        other_colors = ['#d62728', 'orange', 'gray', 'gray', 'red', 'red', 'purple']
        other_linestyles = ['-', ':', ':', ':', '--', '--', '--']
        other_alphas = [1.0, 1.0, 0.8, 0.8, 0.5, 1.0, 1.0]

        # 4. Drug marker legend (unchanged)
        drug_labels = ['D Drug Usage', 'F Drug Usage']
        drug_colors = [(255 / 255, 192 / 255, 192 / 255), (176 / 255, 176 / 255, 255 / 255)]

        # Create legend elements
        legend_elements = []

        # Add strategy description (reduced font size)
        for label in strategy_labels:
            legend_elements.append(plt.Line2D([0], [0], color='none', linewidth=0, label=label))

        # Add separator
        legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=1, label='—— Legend Explanation ——'))

        # Add cell types
        for label, color in zip(cell_labels, cell_colors):
            legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=1.2, label=label))

        # Add other line elements
        for i in range(len(other_labels)):
            if i < 4:
                legend_elements.append(plt.Line2D([0], [0], color=other_colors[i],
                                                  linestyle=other_linestyles[i],
                                                  linewidth=1.2 if i == 0 else 1.0,
                                                  alpha=other_alphas[i], label=other_labels[i]))
            else:
                if i == 4:  # First failure
                    legend_elements.append(plt.Line2D([0], [0], color=other_colors[i],
                                                      linestyle=other_linestyles[i],
                                                      linewidth=1.5, alpha=other_alphas[i],
                                                      label=other_labels[i]))
                elif i == 5:  # Second failure
                    legend_elements.append(plt.Line2D([0], [0], color=other_colors[i],
                                                      linestyle=other_linestyles[i],
                                                      linewidth=1.5, alpha=other_alphas[i],
                                                      label=other_labels[i]))
                else:  # Termination point
                    legend_elements.append(plt.Line2D([0], [0], color=other_colors[i],
                                                      linestyle=other_linestyles[i],
                                                      linewidth=1.5, alpha=other_alphas[i],
                                                      label=other_labels[i]))

        # Add drug usage markers
        from matplotlib.patches import Patch
        for label, color in zip(drug_labels, drug_colors):
            legend_elements.append(Patch(color=color, alpha=0.5, label=label))

        # Move legend to the right to avoid overlapping with letters
        ax.legend(handles=legend_elements, loc='center', ncol=1, fontsize=9,
                  bbox_to_anchor=(0.6, 0.5), frameon=True, fancybox=True, shadow=True,
                  handlelength=1.0, handletextpad=0.4, labelspacing=0.3,
                  borderpad=0.6,
                  bbox_transform=ax.transAxes)

        ax.set_title('Unified Legend Explanation', fontsize=13, pad=15, fontweight='bold')

    # -------------------------- Add Subplot Label Function (Inside Subplot) --------------------------
    def _add_subplot_label(self, ax, letter):
        """Add letter label inside the subplot (top-left corner)"""
        ax.text(0.02, 0.98, letter,
                transform=ax.transAxes,
                fontsize=20, fontweight='black',
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='black',
                          linewidth=1.5, alpha=0.9),
                zorder=10)

    # -------------------------- Three-Column Layout Plotting --------------------------
    def plot_strategies_with_three_columns(self):
        required = ['Strategy 1: D→F→Combination', 'Strategy 2: F→D→Combination',
                    'Strategy 3: Adaptive Single→Combination', 'Strategy 4: Original AT50 Strategy']
        if not all(k in self.results for k in required) or not self.threshold_results:
            print("Please run all strategy simulations and threshold impact analysis first")
            return

        # Further increase chart height to leave more space for ADGJ
        fig = plt.figure(figsize=(22, 30))
        # Increase row spacing, especially for rows containing ADGJ
        gs = fig.add_gridspec(5, 3, hspace=0.3, wspace=0.15, left=0.03, right=0.97, top=0.94, bottom=0.03)

        cell_labels = ['x1(Dual-sensitive)', 'x2(F-only sensitive)', 'x3(D-only sensitive)', 'x4(Dual-resistant)']
        cell_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
        terminate_th = self.params['terminate_threshold']
        a1_default = self.params['a1']  # Default upper threshold a1
        a2_default = self.params['a2']  # Default lower threshold a2
        a1_range = np.round(np.arange(0.6, 1.11, 0.01), 2)  # Upper threshold a1 range
        a2_range = np.round(np.arange(0.1, 0.51, 0.01), 2)  # Lower threshold a2 range

        # Key setting: Unified x-axis range for subplots A, D, G, J is [0, 1300] (strictly limited)
        unified_x_max = 1300  # Unified maximum x-axis length (exactly 1300)
        print(f"Unified x-axis range for subplots A, D, G, J: 0 - {unified_x_max} days (strictly limited)")

        # 1. Draw original strategy curves with adjusted layout for ADGJ
        def plot_strategy(ax, res, title, label_letter, is_combination=True,
                          first_failure=None, second_failure=None, failure_times=None, is_baseline=False):
            # Add subplot label (inside)
            self._add_subplot_label(ax, label_letter)

            # Increase top margin for ADGJ
            if label_letter in ['A', 'D', 'G', 'J']:
                ax.margins(x=0.02, y=0.1)
            else:
                ax.margins(x=0.02, y=0.05)

            for i in range(4):
                ax.plot(res['time'], res['cells'][:, i],
                        color=cell_colors[i], linewidth=1.2)
            ax.plot(res['time'], res['total'],
                    color='#d62728', linewidth=1.8)

            for t_idx, t in enumerate(res['time']):
                # Only draw drug markers where t <= 1300 (avoid exceeding x-axis range)
                if t > unified_x_max:
                    break
                d_dose = res['d_drugs'][t_idx]
                f_dose = res['f_drugs'][t_idx]

                # D drug usage (including combination cases) - Keep original position and color
                if d_dose > 0:
                    ax.add_patch(Rectangle((t - 0.5, terminate_th), 1, 0.1,
                                           facecolor=(255 / 255, 192 / 255, 192 / 255),
                                           alpha=0.2 + 0.6 * d_dose, edgecolor='none'))
                # F drug usage (including combination cases) - Keep original position and color
                if f_dose > 0:
                    ax.add_patch(Rectangle((t - 0.5, terminate_th + 0.1), 1, 0.1,
                                           facecolor=(176 / 255, 176 / 255, 255 / 255),
                                           alpha=0.2 + 0.6 * f_dose, edgecolor='none'))

            ax.axhline(y=terminate_th, color='orange', linestyle=':', linewidth=1.2)
            ax.axhline(y=a1_default, color='gray', linestyle=':', linewidth=1.0,
                       alpha=0.8, label=f'Upper Threshold (a1={a1_default:.2f})')
            ax.axhline(y=a2_default, color='gray', linestyle=':', linewidth=1.0,
                       alpha=0.8, label=f'Lower Threshold (a2={a2_default:.2f})')

            # First failure node: Keep only red dashed line, remove text annotation
            if first_failure is not None and first_failure <= unified_x_max:
                ax.axvline(x=first_failure, color='red', linestyle='--',
                           linewidth=1.5, alpha=0.5)

            # Second failure node: Keep only red dashed line, remove text annotation
            if second_failure is not None and second_failure <= unified_x_max:
                ax.axvline(x=second_failure, color='red', linestyle='--',
                           linewidth=1.5, alpha=1.0)

            # Failure times for adaptive strategy: Keep only red dashed lines, remove text annotations
            if failure_times and len(failure_times) >= 1 and failure_times[0] <= unified_x_max:
                ax.axvline(x=failure_times[0], color='red', linestyle='--',
                           linewidth=1.5, alpha=0.5)
            if failure_times and len(failure_times) >= 2 and failure_times[1] <= unified_x_max:
                ax.axvline(x=failure_times[1], color='red', linestyle='--',
                           linewidth=1.5, alpha=1.0)

            # Termination point: Purple dashed line (subplots A, D, G, J) - If stop time exceeds 1300, display at 1300
            stop_time = min(res['stop_time'], unified_x_max)
            ax.axvline(x=stop_time, color='purple', linestyle='--', linewidth=1.5)
            ax.text(stop_time, terminate_th, f"Termination: {stop_time:.1f}d",
                    color='purple', fontsize=8, verticalalignment='center',
                    horizontalalignment='right' if label_letter in ['A', 'D', 'G', 'J'] else 'left')

            # Adjust annotation position to avoid overlapping with letters
            annotate_y = 0.85 if label_letter in ['A', 'D', 'G', 'J'] else 0.95
            ax.annotate(f'Stop: {res["stop_time"]:.1f}d | a1={res["a1"]:.2f}, a2={res["a2"]:.2f}',
                        xy=(0.95, annotate_y), xycoords='axes fraction',
                        horizontalalignment='right', verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                        fontsize=9, fontweight='bold')

            # Increase title spacing for ADGJ
            title_pad = 20 if label_letter in ['A', 'D', 'G', 'J'] else 10
            ax.set_title(title, fontsize=13, pad=title_pad, fontweight='bold')
            ax.set_ylabel('Cell Quantity', fontsize=11, labelpad=8)
            ax.set_ylim(0, self.params['carrying_capacity'])
            ax.grid(True, linestyle='--', alpha=0.5)

            # Adjust legend position
            if label_letter in ['A', 'D', 'G', 'J']:
                ax.legend(loc='upper right', fontsize=7, bbox_to_anchor=(0.98, 0.8))
            else:
                ax.legend(loc='upper right', fontsize=7, bbox_to_anchor=(0.95, 0.95))

            # Core correction: Force x-axis range of subplots A, D, G, J to [0, 1300], ticks exactly match
            if label_letter in ['A', 'D', 'G', 'J']:
                ax.set_xlim(0, unified_x_max)
                # Set ticks to 0, 200, 400, 600, 800, 1000, 1200, 1300 (not exceeding 1300)
                ax.set_xticks([0, 200, 400, 600, 800, 1000, 1200, unified_x_max])
                ax.set_xticklabels([f'{int(tick)}' for tick in ax.get_xticks()], fontsize=9)
                # Increase x label spacing
                ax.set_xlabel('Time (days)', fontsize=11, labelpad=10)

            # Key modification: Change baseline strategy border to black (consistent with other subplots)
            if is_baseline:
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
                    spine.set_color('black')

            # Reduce tick label font size
            ax.tick_params(axis='both', which='major', labelsize=9)

        # 2. Draw threshold impact trajectory chart - Remove legend
        def plot_threshold_trajectories(ax, strategy_name, title, label_letter, is_baseline=False):
            # Add subplot label (inside)
            self._add_subplot_label(ax, label_letter)

            # Increase subplot margin
            ax.margins(x=0.02, y=0.05)

            results = self.threshold_results[strategy_name]
            sample_rate = 5
            sampled_a1 = a1_range[::sample_rate]  # Sampled a1 values
            sampled_a2 = a2_range[::sample_rate]  # Sampled a2 values

            colors = plt.cm.viridis(np.linspace(0, 1, len(sampled_a1)))

            for i, a1_val in enumerate(sampled_a1):
                for j, a2_val in enumerate(sampled_a2):
                    for r in results:
                        if np.isclose(r['a1'], a1_val) and np.isclose(r['a2'], a2_val):
                            # Only draw trajectory part where t <= 1300
                            time_mask = r['result']['time'] <= unified_x_max
                            ax.plot(r['result']['time'][time_mask], r['result']['total'][time_mask],
                                    color=colors[i], alpha=0.3 + 0.7 * (j / (len(sampled_a2) - 1)),
                                    linewidth=1.0)
                            break

            for r in results:
                if np.isclose(r['a1'], a1_default) and np.isclose(r['a2'], a2_default):
                    # Only draw trajectory part where t <= 1300
                    time_mask = r['result']['time'] <= unified_x_max
                    ax.plot(r['result']['time'][time_mask], r['result']['total'][time_mask],
                            color='black', linewidth=2.0, label=f'Default (a1={a1_default:.2f}, a2={a2_default:.2f})')
                    break

            ax.axhline(y=terminate_th, color='orange', linestyle=':', linewidth=1.2)
            ax.set_title(title, fontsize=13, pad=10, fontweight='bold')
            ax.set_ylabel('Cell Quantity', fontsize=11, labelpad=8)
            ax.set_ylim(0, self.params['carrying_capacity'])
            ax.grid(True, linestyle='--', alpha=0.5)
            # Move legend to the right
            ax.legend(loc='upper right', fontsize=7, bbox_to_anchor=(0.95, 0.95))

            # Key modification: Change baseline strategy border to black (consistent with other subplots)
            if is_baseline:
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
                    spine.set_color('black')

            # Reduce tick label font size
            ax.tick_params(axis='both', which='major', labelsize=9)

        # 3. Draw stop time heatmap (x-axis: a2, y-axis: a1)
        def plot_stop_time_heatmap(ax, strategy_name, title, label_letter, is_baseline=False):
            # Add subplot label (inside)
            self._add_subplot_label(ax, label_letter)

            # Increase subplot margin
            ax.margins(x=0.02, y=0.05)

            results = self.threshold_results[strategy_name]
            result_dict = {(r['a1'], r['a2']): r['result']['stop_time'] for r in results}
            stop_time_matrix = np.zeros((len(a1_range), len(a2_range)))

            for i, a1_val in enumerate(a1_range):
                for j, a2_val in enumerate(a2_range):
                    stop_time_matrix[i, j] = result_dict.get((a1_val, a2_val), 0)

            im = ax.imshow(stop_time_matrix, cmap='viridis', aspect='auto',
                           origin='lower', extent=[0.09, 0.51, 0.59, 1.11])

            default_a1_idx = np.argmin(np.abs(a1_range - a1_default))
            default_a2_idx = np.argmin(np.abs(a2_range - a2_default))
            ax.scatter(a2_default, a1_default, color='red', s=80, marker='*',
                       label=f'Default (a1={a1_default:.2f}, a2={a2_default:.2f})')

            ax.set_xticks(np.arange(0.1, 0.51, 0.1))
            ax.set_yticks(np.arange(0.6, 1.11, 0.1))
            ax.set_xticklabels([f'{x:.1f}' for x in np.arange(0.1, 0.51, 0.1)], rotation=45, fontsize=9)
            ax.set_yticklabels([f'{x:.1f}' for x in np.arange(0.6, 1.11, 0.1)], fontsize=9)
            ax.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.3)

            ax.set_xlabel('Lower Threshold (a2)', fontsize=11, labelpad=8)
            ax.set_ylabel('Upper Threshold (a1)', fontsize=11, labelpad=8)
            ax.set_title(title, fontsize=13, pad=10, fontweight='bold')
            # Move legend to the right
            ax.legend(loc='upper right', fontsize=7, bbox_to_anchor=(0.95, 0.95))

            cbar = fig.colorbar(im, ax=ax, shrink=0.6)
            cbar.set_label('Stop Time (days)', fontsize=9)
            cbar.ax.tick_params(labelsize=8)

            # Key modification: Change baseline strategy border to black (consistent with other subplots)
            if is_baseline:
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
                    spine.set_color('black')

        # Define subplot label letter sequence (row-major order)
        labels = ['A', 'B', 'C',  # Row 0: Three subplots for Strategy 4
                  'D', 'E', 'F',  # Row 1: Three subplots for Strategy 1
                  'G', 'H', 'I',  # Row 2: Three subplots for Strategy 2
                  'J', 'K', 'L',  # Row 3: Three subplots for Strategy 3
                  'M', 'N', 'O']  # Row 4: Legend, Comparison, Bar Chart

        label_idx = 0

        # Strategy 4 (Row 0 - Baseline Strategy) - A subplot
        ax4_col1 = fig.add_subplot(gs[0, 0])
        res4 = self.results['Strategy 4: Original AT50 Strategy']
        plot_strategy(ax4_col1, res4, 'Strategy 1: AT50 (Baseline)', labels[label_idx],
                      first_failure=res4['first_failure_time'], is_baseline=True)
        label_idx += 1

        ax4_col2 = fig.add_subplot(gs[0, 1])
        plot_threshold_trajectories(ax4_col2, 'Strategy 4: Original AT50 Strategy',
                                    'Strategy 1: Threshold Screening (a1/a2)',
                                    labels[label_idx], is_baseline=True)
        label_idx += 1

        ax4_col3 = fig.add_subplot(gs[0, 2])
        plot_stop_time_heatmap(ax4_col3, 'Strategy 4: Original AT50 Strategy',
                               'Strategy 1: Threshold Heatmap (a1/a2)', labels[label_idx], is_baseline=True)
        label_idx += 1

        # Strategy 1 (Row 1) - D subplot
        ax1_col1 = fig.add_subplot(gs[1, 0])
        res1 = self.results['Strategy 1: D→F→Combination']
        plot_strategy(ax1_col1, res1, 'Strategy 2: D→F→Combination', labels[label_idx],
                      first_failure=res1['first_failure_time'],
                      second_failure=res1['second_failure_time'])
        label_idx += 1

        ax1_col2 = fig.add_subplot(gs[1, 1])
        plot_threshold_trajectories(ax1_col2, 'Strategy 1: D→F→Combination', 'Strategy 2: Threshold Screening (a1/a2)',
                                    labels[label_idx])
        label_idx += 1

        ax1_col3 = fig.add_subplot(gs[1, 2])
        plot_stop_time_heatmap(ax1_col3, 'Strategy 1: D→F→Combination', 'Strategy 2: Threshold Heatmap (a1/a2)',
                               labels[label_idx])
        label_idx += 1

        # Strategy 2 (Row 2) - G subplot
        ax2_col1 = fig.add_subplot(gs[2, 0])
        res2 = self.results['Strategy 2: F→D→Combination']
        plot_strategy(ax2_col1, res2, 'Strategy 3: F→D→Combination', labels[label_idx],
                      first_failure=res2['first_failure_time'],
                      second_failure=res2['second_failure_time'])
        label_idx += 1

        ax2_col2 = fig.add_subplot(gs[2, 1])
        plot_threshold_trajectories(ax2_col2, 'Strategy 2: F→D→Combination', 'Strategy 3: Threshold Screening (a1/a2)',
                                    labels[label_idx])
        label_idx += 1

        ax2_col3 = fig.add_subplot(gs[2, 2])
        plot_stop_time_heatmap(ax2_col3, 'Strategy 2: F→D→Combination', 'Strategy 3: Threshold Heatmap (a1/a2)',
                               labels[label_idx])
        label_idx += 1

        # Strategy 3 (Row 3) - J subplot
        ax3_col1 = fig.add_subplot(gs[3, 0])
        res3 = self.results['Strategy 3: Adaptive Single→Combination']
        plot_strategy(ax3_col1, res3, 'Strategy 4: D→F Cyclic Combination (Adaptive)', labels[label_idx],
                      failure_times=res3['failure_times'])
        label_idx += 1

        ax3_col2 = fig.add_subplot(gs[3, 1])
        plot_threshold_trajectories(ax3_col2, 'Strategy 3: Adaptive Single→Combination',
                                    'Strategy 4: Threshold Screening (a1/a2)', labels[label_idx])
        label_idx += 1

        ax3_col3 = fig.add_subplot(gs[3, 2])
        plot_stop_time_heatmap(ax3_col3, 'Strategy 3: Adaptive Single→Combination',
                               'Strategy 4: Threshold Heatmap (a1/a2)', labels[label_idx])
        label_idx += 1

        # Row 4 layout
        # Position 4,0: Unified Legend (fixed no overflow)
        ax_legend = fig.add_subplot(gs[4, 0])
        self._plot_unified_legend(ax_legend, labels[label_idx])
        label_idx += 1

        # Position 4,1: Optimal Strategy Simulation Comparison (Subplot N)
        ax_compare = fig.add_subplot(gs[4, 1])
        self._plot_optimal_strategies_comparison(ax_compare, labels[label_idx])
        label_idx += 1

        # Position 4,2: Strategy Benefit Comparison Bar Chart (Subplot O)
        ax_bar = fig.add_subplot(gs[4, 2])
        self._plot_survival_time_bar(ax_bar, labels[label_idx])
        label_idx += 1

        # Set x-axis labels for all subplots (exclude heatmap and bar chart, they have their own x-axis labels)
        for ax, letter in zip(
                [ax4_col1, ax1_col1, ax2_col1, ax3_col1, ax4_col2, ax1_col2, ax2_col2, ax3_col2, ax_compare],
                ['A', 'D', 'G', 'J', 'B', 'E', 'H', 'K', 'N']):
            if letter in ['A', 'D', 'G', 'J']:
                ax.set_xlabel('Time (days)', fontsize=11, labelpad=10)
            else:
                ax.set_xlabel('Time (days)', fontsize=11, labelpad=8)

        # Final layout adjustment
        plt.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.03, hspace=0.3, wspace=0.15)
        plt.show()


# Main function
def main():
    print("Note: High-precision threshold analysis may take a long time, please be patient...")
    model = CancerModel()

    # Run four strategy simulations (default thresholds a1=1.0, a2=0.5)
    print("\nRunning Strategy 1: D→F→Combination...")
    model.simulate_d_to_f_to_combination_strategy()

    print("Running Strategy 2: F→D→Combination...")
    model.simulate_f_to_d_to_combination_strategy()

    print("Running Strategy 3: Adaptive Single→Combination...")
    model.simulate_adaptive_strategy()

    print("Running Strategy 4: Original AT50 Strategy...")
    model.simulate_at50_strategy()

    # Analyze high-precision threshold variation impact (a1:0.6~1.1, a2:0.1~0.5, step=0.01)
    model.analyze_threshold_effects()

    # Draw three-column comparison chart
    print("\nSimulation completed, drawing result chart...")
    model.plot_strategies_with_three_columns()


if __name__ == "__main__":
    main()
