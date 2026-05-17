"""
CSC 301 - Algorithm Analysis and Design
Group: Efficiency Experts | Section: 6FS3
Milestone 2: Implementation of Interval Scheduling Algorithms

Algorithms:
  1. Greedy Algorithm (Earliest Finish Time)
  2. Dynamic Programming (Weighted Job Scheduling)

Dataset: Cloud Task Scheduling Dataset (20,000 tasks)
         https://www.kaggle.com/datasets/ziya07/cloud-task-scheduling-dataset
"""

import pandas as pd
import numpy as np
import time
import bisect


# ─────────────────────────────────────────────
# STEP 1: LOAD & PREPROCESS DATASET
# ─────────────────────────────────────────────

def load_dataset(filepath):
    # read the CSV file from Kaggle
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[INFO] Columns: {list(df.columns)}\n")

    # the dataset doesn't have start/end times directly,
    # so we derive them from Task_ID and Execution_Time
    # start_time = Task_ID (tasks arrive one by one in order)
    # end_time   = Task_ID + Execution_Time (when it finishes)
    # weight     = Priority (how valuable the task is)
    tasks = pd.DataFrame({
        'task_id':    df['Task_ID'],
        'start_time': df['Task_ID'].astype(float),
        'end_time':   df['Task_ID'].astype(float) + df['Execution_Time (s)'],
        'weight':     df['Priority'],
        'cpu':        df['CPU_Usage (%)'],
        'ram':        df['RAM_Usage (MB)']
    })

    # remove any rows where end time is before or equal to start time
    tasks = tasks[tasks['end_time'] > tasks['start_time']].reset_index(drop=True)
    print(f"[INFO] Valid tasks after preprocessing: {len(tasks)}")
    print(f"[INFO] start_time range : {tasks['start_time'].min():.1f} → {tasks['start_time'].max():.1f}")
    print(f"[INFO] end_time range   : {tasks['end_time'].min():.2f} → {tasks['end_time'].max():.2f}")
    print(f"[INFO] weight (priority): {sorted(tasks['weight'].unique())}\n")
    return tasks


# ─────────────────────────────────────────────
# STEP 2: GREEDY ALGORITHM (Earliest Finish Time)
# ─────────────────────────────────────────────

def greedy_scheduling(tasks_df):
    # sort all tasks by their finish time (earliest finish first)
    # this is the key greedy choice — always pick what ends soonest
    tasks = tasks_df.sort_values('end_time').reset_index(drop=True)

    selected    = []   # will hold the indices of tasks we pick
    last_finish = -1   # tracks when the last selected task ends

    for i in range(len(tasks)):
        # only pick this task if it starts after the last one finished
        if tasks.loc[i, 'start_time'] >= last_finish:
            selected.append(i)
            last_finish = tasks.loc[i, 'end_time']

    return tasks.iloc[selected].reset_index(drop=True)


# ─────────────────────────────────────────────
# STEP 3: DYNAMIC PROGRAMMING (Weighted Scheduling)
# ─────────────────────────────────────────────

def find_latest_non_overlapping(finish_times, start_i):
    # use binary search to find the latest task j that finishes
    # before task i starts — this is p(i) in the DP recurrence
    # returns -1 if no such task exists
    pos = bisect.bisect_right(finish_times, start_i) - 1
    return pos


def dp_weighted_scheduling(tasks_df):
    # sort by finish time so the DP recurrence works correctly
    tasks = tasks_df.sort_values('end_time').reset_index(drop=True)
    n = len(tasks)

    # precompute finish times list so binary search is fast
    finish_times = tasks['end_time'].tolist()

    # M[i] = best total weight we can get using the first i tasks
    M = [0] * (n + 1)

    for i in range(1, n + 1):
        idx = i - 1  # 0-based index for the tasks dataframe

        # find the latest task that doesn't overlap with task i
        p = find_latest_non_overlapping(finish_times[:idx],
                                        tasks.loc[idx, 'start_time'])

        # two choices: include task i or skip it, take the better one
        include = tasks.loc[idx, 'weight'] + M[p + 1]  # include task i
        exclude = M[i - 1]                              # skip task i
        M[i] = max(include, exclude)

    # backtrack through M to figure out which tasks were actually selected
    selected = []
    i = n
    while i >= 1:
        idx = i - 1
        p = find_latest_non_overlapping(finish_times[:idx],
                                        tasks.loc[idx, 'start_time'])
        # if including task i gave a better or equal result, it was selected
        if tasks.loc[idx, 'weight'] + M[p + 1] >= M[i - 1]:
            selected.append(idx)
            i = p + 1  # jump back to the last compatible task
        else:
            i -= 1

    selected.reverse()
    result = tasks.iloc[selected].reset_index(drop=True)
    return result, M[n]


# ─────────────────────────────────────────────
# STEP 4: THREE SCREENSHOT CASES
# ─────────────────────────────────────────────

def get_case_samples(tasks_df):
    """
    BEST CASE    : Tasks are perfectly non-overlapping → all get selected
    WORST CASE   : All tasks overlap completely → only 1 gets selected
    AVERAGE CASE : Random sample from the real dataset → mixed results
    """
    cases = {}

    # best case: tasks are back to back with no overlap at all
    best = pd.DataFrame({
        'task_id':    range(1, 11),
        'start_time': [0,  5, 10, 15, 20, 25, 30, 35, 40, 45],
        'end_time':   [4,  9, 14, 19, 24, 29, 34, 39, 44, 49],
        'weight':     [2,  1,  3,  2,  1,  3,  2,  1,  3,  2]
    })
    cases['BEST'] = best

    # worst case: every task runs at the exact same time, maximum overlap
    worst = pd.DataFrame({
        'task_id':    range(1, 11),
        'start_time': [0.0] * 10,
        'end_time':   [5.0] * 10,
        'weight':     [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
    })
    cases['WORST'] = worst

    # average case: 20 random tasks from the real Kaggle dataset
    cases['AVERAGE'] = tasks_df.sample(20, random_state=42).reset_index(drop=True)

    return cases


def run_and_time(algorithm, tasks, label, case_name):
    # start the timer right before running the algorithm
    start = time.perf_counter()

    if algorithm == 'greedy':
        result = greedy_scheduling(tasks)
        extra  = None
    else:
        result, extra = dp_weighted_scheduling(tasks)

    # stop timer and convert to milliseconds
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"{'='*55}")
    print(f"  Algorithm  : {label}")
    print(f"  Case       : {case_name}")
    print(f"  Input size : {len(tasks)} tasks")
    print(f"  Selected   : {len(result)} tasks")
    if extra is not None:
        print(f"  Max weight : {extra}")
    print(f"  Time       : {elapsed_ms:.4f} ms")
    print(f"{'='*55}\n")

    return result, elapsed_ms


# ─────────────────────────────────────────────
# STEP 5: SCALABILITY EXPERIMENT
# ─────────────────────────────────────────────

def scalability_experiment(tasks_df, sizes=None):
    # test both algorithms on different input sizes to see how they scale
    if sizes is None:
        sizes = [100, 500, 1000, 2000, 5000]

    print("\n" + "="*55)
    print("  SCALABILITY EXPERIMENT")
    print("="*55)
    print(f"{'n':>8} | {'Greedy (ms)':>14} | {'DP (ms)':>14}")
    print("-"*42)

    records = []
    for n in sizes:
        # take a random sample of size n from the full dataset
        sample = tasks_df.sample(min(n, len(tasks_df)),
                                 random_state=1).reset_index(drop=True)

        # time the greedy algorithm
        t0 = time.perf_counter()
        greedy_scheduling(sample)
        greedy_ms = (time.perf_counter() - t0) * 1000

        # time the DP algorithm
        t0 = time.perf_counter()
        dp_weighted_scheduling(sample)
        dp_ms = (time.perf_counter() - t0) * 1000

        print(f"{n:>8} | {greedy_ms:>14.4f} | {dp_ms:>14.4f}")
        records.append({'n': n, 'greedy_ms': greedy_ms, 'dp_ms': dp_ms})

    print()
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*55)
    print("  CSC 301 — Interval Scheduling Project")
    print("  Group: Efficiency Experts | Section: 6FS3")
    print("="*55 + "\n")

    # load the dataset — make sure the CSV is in the same folder as this file
    CSV_PATH = "cloud_task_scheduling_dataset_20k.csv"
    tasks_df = load_dataset(CSV_PATH)

    # show a small preview of what the data looks like after preprocessing
    print("Sample of preprocessed dataset (first 10 rows):")
    print(tasks_df[['task_id', 'start_time', 'end_time', 'weight']].head(10).to_string(index=False))
    print()

    # run best, worst, and average cases
    print("=" * 55)
    print("  SCREENSHOT CASES")
    print("=" * 55 + "\n")

    cases = get_case_samples(tasks_df)
    for case_name, case_data in cases.items():
        run_and_time('greedy', case_data, 'Greedy (EFT)',  case_name)
        run_and_time('dp',     case_data, 'DP (Weighted)', case_name)

    # run both algorithms on a sample of 1000 tasks from the real dataset
    print("=" * 55)
    print("  FULL DATASET RUN (sample of 1000 tasks)")
    print("=" * 55 + "\n")

    sample_1000 = tasks_df.sample(1000, random_state=42).reset_index(drop=True)
    greedy_result, _     = run_and_time('greedy', sample_1000, 'Greedy (EFT)',  'FULL')
    dp_result,     max_w = run_and_time('dp',     sample_1000, 'DP (Weighted)', 'FULL')

    print("Greedy — first 5 selected tasks:")
    print(greedy_result[['task_id','start_time','end_time','weight']].head().to_string(index=False))
    print()
    print("DP — first 5 selected tasks:")
    print(dp_result[['task_id','start_time','end_time','weight']].head().to_string(index=False))
    print()

    # run scalability test and save results for Milestone 3 graphs
    scale_df = scalability_experiment(tasks_df)
    scale_df.to_csv("scalability_results.csv", index=False)
    print("[INFO] Scalability results saved → scalability_results.csv")
