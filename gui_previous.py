"""Read the saved 100-seed benchmark and display comparisons in Tkinter.

Place this next to gui.py and the results/ directory.
No new simulation or extra package is required.
"""
import csv
import math
import tkinter as tk
from collections import defaultdict
from pathlib import Path
from tkinter import ttk, messagebox

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / 'results' / 'final_marl_dynamic_benchmark.csv'

PALETTE = {
    'Auction': '#2563eb',
    'Hungarian V3': '#0d9488',
    'Hungarian V4': '#7c3aed',
    'MARL Original': '#64748b',
    'Dynamic 10': '#f59e0b',
    'Dynamic 20': '#db2777',
}
METRICS = (
    ('finish', 'Completion time', 'simulation steps'),
    ('distance', 'Total distance', 'grid cells travelled'),
    ('p1_avg', 'Normal delivery time (P1)', 'steps per normal task'),
    ('p3_avg', 'Urgent delivery time (P3)', 'steps per urgent task'),
)


def read_results(path=CSV_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f'Benchmark CSV not found:\n{path}\n\n'
            'Run experiments/final_marl_dynamic_benchmark.py first '
            'or copy the CSV into results/.'
        )
    groups = defaultdict(list)
    with path.open(newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        needed = {'algorithm', 'seed', 'completed', 'total_tasks', 'collisions',
                  'error', *(field for field, _, _ in METRICS)}
        missing = needed - set(reader.fieldnames or [])
        if missing:
            raise ValueError('Missing CSV columns: ' + ', '.join(sorted(missing)))
        for row in reader:
            name = row['algorithm'].strip()
            if not name:
                continue
            try:
                completed = int(row['completed'])
                total = int(row['total_tasks'])
                collisions = int(row['collisions'])
            except (TypeError, ValueError):
                raise ValueError(f'Invalid counts in CSV row for {name}') from None
            success = (row.get('error', '').strip() in ('', 'None')
                       and total == 40 and completed == 40 and collisions == 0)
            item = {'success': success, 'seed': row['seed']}
            for field, _, _ in METRICS:
                try:
                    item[field] = float(row[field]) if row[field] else None
                except ValueError:
                    item[field] = None
            groups[name].append(item)
    if not groups:
        raise ValueError('Benchmark CSV has no results')
    return dict(groups)


def metric_mean(rows, key):
    values = [r[key] for r in rows if r['success'] and r[key] is not None
              and math.isfinite(r[key])]
    return sum(values) / len(values) if values else None


class ComparisonWindow:
    def __init__(self, parent):
        self.data = read_results()
        self.window = tk.Toplevel(parent)
        self.window.title('Algorithm Comparison | 100-seed benchmark')
        self.window.geometry('1080x790')
        self.window.minsize(850, 620)
        self.window.configure(bg='#e2e8f0')

        tk.Label(self.window, text='Algorithm Comparison',
                 bg='#e2e8f0', fg='#0f172a',
                 font=('Segoe UI', 19, 'bold')).pack(pady=(14, 2))
        tk.Label(self.window,
                 text='Saved benchmark results · 20×20 grid · 5 robots · 40 dynamic tasks',
                 bg='#e2e8f0', fg='#475569', font=('Segoe UI', 10)
                 ).pack(pady=(0, 10))

        selection = tk.Frame(self.window, bg='white', padx=12, pady=10)
        selection.pack(fill='x', padx=16, pady=(0, 8))
        tk.Label(selection, text='Show algorithms:', bg='white', fg='#334155',
                 font=('Segoe UI', 10, 'bold')).pack(side='left', padx=(0, 10))
        order = [name for name in PALETTE if name in self.data]
        order.extend(name for name in self.data if name not in order)
        self.selected = {}
        for name in order:
            var = tk.BooleanVar(value=True)
            self.selected[name] = var
            tk.Checkbutton(selection, text=name, variable=var,
                           command=self.render, bg='white',
                           fg=PALETTE.get(name, '#334155'),
                           activebackground='white',
                           font=('Segoe UI', 9)).pack(side='left', padx=2)

        self.scroller = tk.Canvas(self.window, bg='#e2e8f0',
                                  highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.window, orient='vertical',
                                   command=self.scroller.yview)
        self.scroller.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y', padx=(0, 12), pady=(0, 12))
        self.scroller.pack(side='left', fill='both', expand=True,
                           padx=(16, 0), pady=(0, 12))
        self.content = tk.Frame(self.scroller, bg='#e2e8f0')
        self.content_id = self.scroller.create_window(
            (0, 0), window=self.content, anchor='nw')
        self.content.bind('<Configure>', lambda event: self.scroller.configure(
            scrollregion=self.scroller.bbox('all')))
        self.scroller.bind('<Configure>', lambda event: self.scroller.itemconfigure(
            self.content_id, width=event.width))
        self.render()

    def render(self):
        for child in self.content.winfo_children():
            child.destroy()
        names = [name for name, var in self.selected.items() if var.get()]
        if not names:
            tk.Label(self.content, text='Select at least one algorithm.',
                     bg='#e2e8f0', fg='#475569').pack(pady=25)
            return

        counts = tk.Frame(self.content, bg='white', padx=16, pady=11)
        counts.pack(fill='x', pady=(0, 10))
        tk.Label(counts, text='Completed runs (success / total)',
                 bg='white', fg='#0f172a',
                 font=('Segoe UI', 11, 'bold')).pack(anchor='w')
        items = []
        for name in names:
            rows = self.data[name]
            good = sum(row['success'] for row in rows)
            items.append(f'{name}: {good}/{len(rows)}')
        tk.Label(counts, text='     |     '.join(items), wraplength=950,
                 justify='left', bg='white', fg='#334155',
                 font=('Segoe UI', 9)).pack(anchor='w', pady=(5, 0))

        for key, title, unit in METRICS:
            means = [(name, metric_mean(self.data[name], key)) for name in names]
            available = [(name, value) for name, value in means if value is not None]
            card = tk.Frame(self.content, bg='white', padx=15, pady=11)
            card.pack(fill='x', pady=(0, 10))
            tk.Label(card, text=f'{title}  ·  {unit}',
                     bg='white', fg='#0f172a',
                     font=('Segoe UI', 11, 'bold')).pack(anchor='w')
            tk.Label(card, text='Lower is better · averages of successful runs',
                     bg='white', fg='#64748b',
                     font=('Segoe UI', 9)).pack(anchor='w', pady=(1, 8))

            if not available:
                tk.Label(card, text='No successful runs with this metric',
                         bg='white', fg='#64748b').pack()
                continue
            max_value = max(value for _, value in available)
            best_value = min(value for _, value in available)
            for name, value in means:
                line = tk.Frame(card, bg='white')
                line.pack(fill='x', pady=3)
                tk.Label(line, text=name, width=18, anchor='w',
                         bg='white', fg='#334155',
                         font=('Segoe UI', 9)).pack(side='left')
                bar = tk.Canvas(line, width=530, height=20, bg='#f1f5f9',
                                highlightthickness=0)
                bar.pack(side='left', fill='x', expand=True, padx=(5, 9))
                if value is not None and max_value > 0:
                    # Draw bar relative to zero, with a common scale per metric.
                    def redraw(event, *, current=value, algorithm=name,
                               maximum=max_value, widget=bar):
                        widget.delete('all')
                        width = max(1, event.width)
                        length = max(1, int(width * current / maximum))
                        widget.create_rectangle(0, 0, length, 20, width=0,
                            fill=PALETTE.get(algorithm, '#475569'))
                    bar.bind('<Configure>', redraw)
                else:
                    bar.create_text(8, 10, text='—', anchor='w', fill='#64748b')
                label = '—' if value is None else f'{value:.2f}'
                if value is not None and abs(value - best_value) < 1e-9:
                    label += '  ★'
                tk.Label(line, text=label, width=12, anchor='e',
                         bg='white', fg='#0f172a',
                         font=('Segoe UI', 9, 'bold')).pack(side='right')

        tk.Label(self.content,
                 text='Descriptive comparison only. Small differences do not imply statistical significance. '
                      'The CSV contains the individual runs for deeper analysis.',
                 bg='#e2e8f0', fg='#475569', wraplength=950,
                 font=('Segoe UI', 9)).pack(anchor='w', pady=(0, 12))


def open_comparison(parent):
    try:
        return ComparisonWindow(parent)
    except (OSError, ValueError) as exc:
        messagebox.showerror('Algorithm comparison', str(exc), parent=parent)
        return None


if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    dialog = open_comparison(root)
    if dialog is not None:
        dialog.window.protocol('WM_DELETE_WINDOW', root.destroy)
        root.mainloop()
    else:
        root.destroy()
