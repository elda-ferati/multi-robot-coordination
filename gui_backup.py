import sys
import random
import tkinter as tk
from functools import lru_cache
from pathlib import Path
from tkinter import ttk, messagebox

ROOT = Path(__file__).resolve().parent
# This file belongs in the project root, alongside src/ and models_v4_dynamic/.
sys.path.insert(0, str(ROOT / 'src'))

import torch
import simulation
from dynamic_tasks import DynamicTaskGenerator
from auction_allocation import allocate_tasks_auction
from hungarian_v3 import allocate_tasks_hungarian_v3
from hungarian_v4 import allocate_tasks_hungarian_v4
from marl_features_v2 import get_team_features
from marl_model_v2 import MARLQNetworkV2
from marl_task_policy_v4 import select_task_actions

GRID = 20
CELL = 30
MAX_STEPS = 500
COLORS = {
    'background': '#e2e8f0',
    'floor': '#f1f5f9',
    'shelf': '#475569',
    'robots': ['#2563eb', '#059669', '#7c3aed', '#db2777', '#0891b2'],
    'dynamic': '#ef4444',
    'normal': '#eab308',
    'urgent': '#f97316',
    'delivery': '#22c55e',
}


@lru_cache(maxsize=1)
def load_dynamic_model():
    path = ROOT / 'models_v4_dynamic' / 'marl_v4_dynamic_episode_20.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = MARLQNetworkV2()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def allocate_marl_dynamic(robots, tasks, environment):
    pending = sorted(
        (task for task in tasks if task.status == 'pending'),
        key=lambda task: (-task.priority, task.task_id),
    )
    idle = [
        robot for robot in robots
        if robot.current_task is None and robot.goal is None
    ]
    if not pending or not idle:
        return

    observations = {}
    for robot in sorted(idle, key=lambda r: r.robot_id):
        observations[robot.robot_id] = [
            {
                'task_id': task.task_id,
                'features': get_team_features(
                    robot, task, robots, tasks, environment
                ),
            }
            for task in pending[:8]
        ]

    actions = select_task_actions(load_dynamic_model(), observations)
    robot_lookup = {r.robot_id: r for r in robots}
    task_lookup = {t.task_id: t for t in tasks}

    for rid, tid in actions.items():
        if tid is None:
            continue
        task = task_lookup[tid]
        if task.status == 'pending':
            robot_lookup[rid].assign_task(task, environment)


ALGORITHMS = {
    'Auction': allocate_tasks_auction,
    'Hungarian V3': allocate_tasks_hungarian_v3,
    'Hungarian V4': allocate_tasks_hungarian_v4,
    'MARL Dynamic 20': allocate_marl_dynamic,
}


def create_scenario(seed):
    random.seed(seed)
    sim = simulation.Simulation(rows=GRID, cols=GRID)

    for row in (5, 10, 15):
        for col in range(2, 18):
            if col % 6 not in (0, 1):
                sim.environment.add_obstacle(row, col)

    positions = [(0, 0), (0, 19), (19, 0), (19, 19), (9, 9)]
    for rid, position in enumerate(positions, 1):
        assert sim.add_robot(rid, position)

    for _ in range(5):
        for _attempt in range(100):
            row, col = random.randrange(GRID), random.randrange(GRID)
            if sim.environment.grid[row][col] != 0:
                continue
            if (row, col) in [r.position for r in sim.robots]:
                continue
            sim.environment.add_dynamic_obstacle(row, col)
            break

    generator = DynamicTaskGenerator(
        simulation=sim,
        seed=seed + 10000,
        interval=5,
        tasks_per_interval=4,
        urgent_probability=0.45,
        max_new_tasks=40,
    )
    return sim, generator


class WarehouseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Multi-Robot Coordination Simulator')
        self.root.geometry('1240x920')
        self.root.minsize(1050, 830)
        self.root.configure(bg=COLORS['background'])

        torch.set_num_threads(2)
        self.running = False
        self.finished = False
        self.sim = None
        self.generator = None
        self.after_id = None
        self.original_allocator = simulation.allocate_tasks

        self.algorithm = tk.StringVar(value='MARL Dynamic 20')
        self.seed = tk.StringVar(value='100')
        self.speed = tk.IntVar(value=100)
        self.metrics = {}
        self.robot_labels = {}

        self.build_ui()
        self.reset()
        self.root.protocol('WM_DELETE_WINDOW', self.close)

    def section_label(self, parent, title):
        tk.Label(
            parent, text=title, bg='white', fg='#334155',
            font=('Segoe UI', 9, 'bold'),
        ).pack(anchor='w')

    def build_ui(self):
        tk.Label(
            self.root, text='AI-Powered Multi-Robot Coordination',
            font=('Segoe UI', 20, 'bold'),
            bg=COLORS['background'], fg='#0f172a',
        ).pack(pady=(12, 2))
        tk.Label(
            self.root, text='Dynamic Warehouse Simulation | 20x20 Grid | 5 Robots',
            font=('Segoe UI', 10),
            bg=COLORS['background'], fg='#475569',
        ).pack(pady=(0, 12))

        # Main area: 20x20 grid left, controls + live metrics right.
        body = tk.Frame(self.root, bg=COLORS['background'])
        body.pack(fill='x', padx=15)

        self.canvas = tk.Canvas(
            body, width=GRID * CELL, height=GRID * CELL,
            bg='white', highlightthickness=0,
        )
        self.canvas.pack(side='left', anchor='n', padx=(0, 15))

        panel_container = tk.Frame(body, bg='white', width=420, height=600)
        panel_container.pack(side='left', fill='both', expand=True)
        panel_container.pack_propagate(False)

        panel_canvas = tk.Canvas(
            panel_container, bg='white', highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(
            panel_container, orient='vertical', command=panel_canvas.yview,
        )
        panel_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        panel_canvas.pack(side='left', fill='both', expand=True)

        panel = tk.Frame(panel_canvas, bg='white', padx=18, pady=15)
        window_id = panel_canvas.create_window((0, 0), window=panel, anchor='nw')
        panel.bind('<Configure>', lambda e: panel_canvas.configure(
            scrollregion=panel_canvas.bbox('all')
        ))
        panel_canvas.bind('<Configure>', lambda e: panel_canvas.itemconfigure(
            window_id, width=e.width
        ))

        self.section_label(panel, 'ALGORITHM')
        self.algo_combo = ttk.Combobox(
            panel, textvariable=self.algorithm,
            values=list(ALGORITHMS), state='readonly',
        )
        self.algo_combo.pack(fill='x', pady=(4, 14))

        self.section_label(panel, 'SCENARIO SEED')
        ttk.Entry(panel, textvariable=self.seed).pack(
            fill='x', pady=(4, 12)
        )

        buttons = tk.Frame(panel, bg='white')
        buttons.pack(fill='x', pady=(0, 12))
        for text, command in (
            ('Start', self.start),
            ('Pause', self.pause),
            ('Step', self.single_step),
        ):
            ttk.Button(buttons, text=text, command=command).pack(
                side='left', fill='x', expand=True, padx=2
            )

        ttk.Button(
            panel, text='Reset Simulation', command=self.reset,
        ).pack(fill='x', pady=(0, 15))

        self.section_label(panel, 'SPEED (DELAY IN MS)')
        ttk.Scale(panel, from_=20, to=500, variable=self.speed).pack(
            fill='x', pady=(4, 15)
        )

        self.section_label(panel, 'LIVE METRICS')
        metric_items = (
            ('step', 'Simulation step'),
            ('completed', 'Completed deliveries'),
            ('urgent', 'Urgent deliveries'),
            ('distance', 'Total distance'),
            ('collisions', 'Collisions'),
            ('astar', 'A* calls'),
            ('yield', 'Yield events'),
        )
        for key, label in metric_items:
            line = tk.Frame(panel, bg='white')
            line.pack(fill='x', pady=5)
            tk.Label(
                line, text=label, bg='white', fg='#475569',
                font=('Segoe UI', 10),
            ).pack(side='left')
            value = tk.Label(
                line, text='0', bg='white', fg='#0f172a',
                font=('Segoe UI', 10, 'bold'),
            )
            value.pack(side='right')
            self.metrics[key] = value

        self.status = tk.Label(
            panel, text='Ready', bg='white', fg='#2563eb',
            font=('Segoe UI', 10, 'bold'), wraplength=340,
        )
        self.status.pack(pady=(20, 10))

        # NEW: Robot status cards are OUTSIDE the scrollable right panel.
        # They remain visible below the map, even if the panel scrolls.
        status_area = tk.Frame(self.root, bg=COLORS['background'])
        status_area.pack(fill='x', padx=15, pady=(10, 2))
        tk.Label(
            status_area, text='ROBOT STATUS',
            bg=COLORS['background'], fg='#334155',
            font=('Segoe UI', 10, 'bold'),
        ).pack(anchor='w', pady=(0, 6))

        cards = tk.Frame(status_area, bg=COLORS['background'])
        cards.pack(fill='x')
        for index in range(5):
            rid = index + 1
            card = tk.Frame(
                cards, bg='white', bd=1, relief='solid',
                padx=8, pady=6,
            )
            card.grid(row=0, column=index, padx=4, sticky='nsew')
            cards.grid_columnconfigure(index, weight=1, uniform='robot')

            tk.Label(
                card, text=f'● Robot {rid}',
                fg=COLORS['robots'][index], bg='white',
                font=('Segoe UI', 10, 'bold'), anchor='w',
            ).pack(fill='x')
            label = tk.Label(
                card, text='Idle\nNo task\nPosition: —\nDistance: 0 | Done: 0',
                bg='white', fg='#334155',
                font=('Segoe UI', 8), anchor='w', justify='left',
            )
            label.pack(fill='x', pady=(4, 0))
            self.robot_labels[rid] = label

        legend = tk.Frame(self.root, bg=COLORS['background'])
        legend.pack(fill='x', padx=16, pady=(9, 8))
        for color, name in (
            (COLORS['robots'][0], 'Robot'),
            (COLORS['shelf'], 'Shelf'),
            (COLORS['dynamic'], 'Moving obstacle'),
            (COLORS['normal'], 'Normal pickup'),
            (COLORS['urgent'], 'Urgent pickup'),
            (COLORS['delivery'], 'Delivery'),
        ):
            tk.Label(
                legend, text=f'● {name}    ', fg=color,
                bg=COLORS['background'],
                font=('Segoe UI', 9, 'bold'),
            ).pack(side='left')

    def cancel_animation(self):
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def reset(self):
        self.running = False
        self.finished = False
        self.cancel_animation()
        try:
            seed = int(self.seed.get())
            self.sim, self.generator = create_scenario(seed)
            self.algo_combo.configure(state='readonly')
            self.status.config(text='Ready to start', fg='#2563eb')
            self.draw()
        except Exception as exc:
            messagebox.showerror('Reset error', str(exc))

    def start(self):
        if self.running or self.finished or self.sim is None:
            return
        self.running = True
        self.algo_combo.configure(state='disabled')
        self.status.config(text='Simulation running', fg='#059669')
        self.animate()

    def pause(self):
        self.running = False
        self.cancel_animation()
        if not self.finished:
            self.status.config(text='Simulation paused', fg='#d97706')

    def single_step(self):
        if self.running or self.finished:
            return
        self.algo_combo.configure(state='disabled')
        self.advance()

    def animate(self):
        self.after_id = None
        if not self.running:
            return
        self.advance()
        if self.running:
            self.after_id = self.root.after(
                max(20, int(self.speed.get())), self.animate
            )

    def advance(self):
        try:
            simulation.allocate_tasks = ALGORITHMS[self.algorithm.get()]
            self.sim.step()
            if self.sim.time_step <= 50:
                self.generator.update()
            self.draw()

            completed = sum(t.status == 'completed' for t in self.sim.tasks)
            done = (
                self.sim.time_step >= 50
                and len(self.sim.tasks) == 40
                and completed == 40
            )
            if done or self.sim.time_step >= MAX_STEPS:
                self.running = False
                self.finished = True
                if done:
                    self.status.config(
                        text='All 40 deliveries completed!', fg='#059669'
                    )
                else:
                    self.status.config(
                        text='Maximum steps reached', fg='#dc2626'
                    )
        except Exception as exc:
            self.running = False
            self.cancel_animation()
            self.status.config(text='Simulation error', fg='#dc2626')
            messagebox.showerror('Simulation error', str(exc))

    def draw_circle(self, row, col, color, radius):
        cx = col * CELL + CELL / 2
        cy = row * CELL + CELL / 2
        self.canvas.create_oval(
            cx - radius, cy - radius, cx + radius, cy + radius,
            fill=color, outline='',
        )

    def draw(self):
        self.canvas.delete('all')
        if self.sim is None:
            return
        for row in range(GRID):
            for col in range(GRID):
                x1, y1 = col * CELL, row * CELL
                color = (
                    COLORS['shelf']
                    if self.sim.environment.grid[row][col] != 0
                    else COLORS['floor']
                )
                self.canvas.create_rectangle(
                    x1, y1, x1 + CELL, y1 + CELL,
                    fill=color, outline='#cbd5e1',
                )

        for row, col in self.sim.environment.dynamic_obstacles:
            self.draw_circle(row, col, COLORS['dynamic'], 7)

        for task in self.sim.tasks:
            if task.status == 'completed':
                continue
            if task.status == 'picked_up':
                row, col = task.delivery
                color = COLORS['delivery']
            else:
                row, col = task.pickup
                color = COLORS['urgent'] if task.priority >= 3 else COLORS['normal']
            self.draw_circle(row, col, color, 5)

        for index, robot in enumerate(self.sim.robots):
            if not robot.path:
                continue
            coords = []
            for row, col in [robot.position, *robot.path]:
                coords.extend((col * CELL + CELL / 2, row * CELL + CELL / 2))
            if len(coords) >= 4:
                self.canvas.create_line(
                    *coords,
                    fill=COLORS['robots'][index % len(COLORS['robots'])],
                    width=3, dash=(5, 3), smooth=False,
                )

        for index, robot in enumerate(self.sim.robots):
            row, col = robot.position
            self.draw_circle(
                row, col, COLORS['robots'][index % len(COLORS['robots'])], 11
            )
            self.canvas.create_text(
                col * CELL + CELL / 2,
                row * CELL + CELL / 2,
                text=str(robot.robot_id), fill='white',
                font=('Segoe UI', 9, 'bold'),
            )

        self.update_metrics()
        self.update_robot_status()

    def update_metrics(self):
        tasks = self.sim.tasks
        completed = sum(t.status == 'completed' for t in tasks)
        urgent = [t for t in tasks if t.priority >= 3]
        urgent_completed = sum(t.status == 'completed' for t in urgent)
        values = {
            'step': self.sim.time_step,
            'completed': f'{completed}/{len(tasks)}',
            'urgent': f'{urgent_completed}/{len(urgent)}',
            'distance': self.sim.total_distance,
            'collisions': self.sim.collision_count,
            'astar': self.sim.astar_calls,
            'yield': self.sim.yield_events,
        }
        for key, value in values.items():
            self.metrics[key].config(text=str(value))

    def update_robot_status(self):
        for robot in self.sim.robots:
            rid = robot.robot_id
            if robot.current_task is None:
                status = 'Idle'
                task_info = 'No task'
                goal = '—'
            else:
                task = robot.current_task
                status = 'Delivering' if robot.carrying_package else 'To pickup'
                task_info = f'Task {task.task_id} (P{task.priority})'
                goal = str(robot.goal)

            self.robot_labels[rid].config(text=(
                f'{status}\n'
                f'{task_info}\n'
                f'Pos: {robot.position} | Goal: {goal}\n'
                f'Dist: {self.sim.robot_distance[rid]} | '
                f'Wait: {self.sim.robot_waiting[rid]} | '
                f'Done: {robot.completed_tasks}'
            ))

    def close(self):
        self.running = False
        self.cancel_animation()
        simulation.allocate_tasks = self.original_allocator
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    WarehouseGUI(root)
    root.mainloop()
