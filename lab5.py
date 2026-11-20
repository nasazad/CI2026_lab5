
import argparse
import random
import time
from pathlib import Path
from types import SimpleNamespace

from cave import EFFECTORS, Cave
from classifiers import HAND_WRITTEN, run_episode, show
from learning import DeadReckoning, LearningCS, compact

SLIDE = [['stench', '-', 'breeze', 'PIT'],
         ['WUMPUS', 'breeze stench glitter', 'PIT', 'breeze'],
         ['stench', '-', 'breeze', '-'],
         ['-', 'breeze', 'PIT', 'breeze']]
BEST = 988                          


def check_cave():
    rows = Cave().percept_map()
    assert rows == SLIDE, rows
    for y, row in zip(range(4, 0, -1), rows):
        print(f'  y={y}  ' + ' | '.join(f'{cell:21}' for cell in row))
    print('        ' + ''.join(f'x={x:<22}' for x in range(1, 5)))


def check_dead_reckoning(episodes=500, seed=0):
    rng, cave, memory = random.Random(seed), Cave(), DeadReckoning()
    for _ in range(episodes):
        sensors = cave.reset()
        memory.reset()
        for _ in range(60):
            truth = {f'x{cave.pos[0]}', f'y{cave.pos[1]}', f'facing_{cave.heading}',
                     'has_gold' if cave.has_gold else 'no_gold'}
            assert memory.messages() == truth, (memory.messages(), truth)
            action = rng.choice(EFFECTORS)
            new_sensors, _, done = cave.step(action)
            memory.update(action, sensors, new_sensors)
            sensors = new_sensors
            if done:
                break


def print_trace(trace, whole_rule=True):
    cells = [str(s.winner) if whole_rule else f'{show(s.winner.action)}  strength {s.winner.strength:6.1f}'
             for s in trace]
    w_msgs, w_cell = max(len(show(s.msgs)) for s in trace), max(map(len, cells))
    title = 'winning classifier' if whole_rule else 'winner'
    print(f"    t  square  facing  {'message list':{w_msgs}}  {title:{w_cell}}  reward")
    for s, cell in zip(trace, cells):
        print(f'   {s.t:>2}  ({s.square[0]},{s.square[1]})   {s.heading:6}  {show(s.msgs):{w_msgs}}  '
              f'{cell:{w_cell}}  {s.reward:+5d}')


def learn(seed, episodes, generations, test_every=25):
    cave, lcs = Cave(), LearningCS(memory=DeadReckoning(), seed=seed)
    scores, tests = [], []
    for e in range(1, episodes + 1):
        scores.append(lcs.episode(cave))
        if e % test_every == 0:
            tests.append(lcs.test(cave)[0])
    score, trace = lcs.test(cave)
    assert score == sum(s.reward for s in trace)
    used = list(dict.fromkeys(s.winner for s in trace))        
    simple, fitness = compact(used, cave, DeadReckoning(), seed=seed, generations=generations)
    stable = len(tests)                                         
    while stable > 0 and tests[stable - 1] == score:            
        stable -= 1
    return SimpleNamespace(seed=seed, score=score, trace=trace, created=len(lcs.rules), used=used,
                           first_win=next((i + 1 for i, s in enumerate(scores) if s > 0), None),
                           stable_from=(stable + 1) * test_every, simple=simple, simple_score=fitness[0],
                           scores=scores, tests=tests, test_every=test_every)


def messages_only(seeds, episodes, tags=(1, 2, 3, 4)):
    rows = []
    for k in tags:
        greedy, wins = [], []
        for seed in seeds:
            cave, lcs = Cave(), LearningCS(tags=k, seed=seed)
            wins.append(sum(lcs.episode(cave) > 0 for _ in range(episodes)))
            greedy.append(lcs.test(cave)[0])
        rows.append((k, greedy, wins))
    return rows


def online_ga(seeds, episodes):
    rows = []
    for seed in seeds:
        cave, lcs = Cave(), LearningCS(memory=DeadReckoning(), seed=seed, online_ga=True)
        wins = sum(lcs.episode(cave) > 0 for _ in range(episodes))
        score, trace = lcs.test(cave)
        rows.append((seed, score, wins, len(lcs.rules), trace))
    return rows


def plot(results, path, window=50):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    surface, ink, ink2, muted, grid, axis, blue = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#2a78d6'
    plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans'], 'font.size': 10, 'text.color': ink,
                         'axes.labelcolor': ink2, 'xtick.color': muted, 'ytick.color': muted})
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.9), facecolor=surface)
    n = len(results)

    every = results[0].test_every
    xs = [every * (i + 1) for i in range(len(results[0].tests))]
    share = [100 * sum(r.tests[i] == BEST for r in results) / n for i in range(len(xs))]
    left.plot(xs, share, color=blue, lw=2, solid_joinstyle='round', solid_capstyle='round')
    left.set_ylim(-3, 105)
    left.set_yticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'])
    left.set_title(f'Greedy run scores {BEST} (the optimum): share of {n} seeds', loc='left', fontsize=11)
    left.set_xlabel('training episode')
    stays = len(share)                                  
    while stays > 0 and share[stays - 1] == 100:
        stays -= 1
    if stays < len(share):
        left.annotate(f'all {n} seeds from episode {xs[stays]} on', (xs[stays], 100), xytext=(6, -18),
                      textcoords='offset points', color=ink2, fontsize=9)

    def moving(values):
        return [sum(values[max(0, i - window + 1):i + 1]) / len(values[max(0, i - window + 1):i + 1])
                for i in range(len(values))]

    curves = [moving(r.scores) for r in results]
    episodes = range(1, len(curves[0]) + 1)
    mean = [sum(c[i] for c in curves) / n for i in range(len(curves[0]))]
    low = [min(c[i] for c in curves) for i in range(len(curves[0]))]
    high = [max(c[i] for c in curves) for i in range(len(curves[0]))]
    right.fill_between(episodes, low, high, color=blue, alpha=0.10, lw=0)
    right.plot(episodes, mean, color=blue, lw=2, solid_joinstyle='round', solid_capstyle='round')
    right.axhline(BEST, color=muted, lw=1)
    right.annotate(f'best possible: {BEST}', (len(mean), BEST), xytext=(-4, 5), textcoords='offset points',
                   ha='right', color=ink2, fontsize=9)
    right.set_title(f'Training score, {window}-episode average (20% random actions)', loc='left', fontsize=11)
    right.set_xlabel('training episode')
    right.set_ylim(-1100, 1100)
    right.set_yticks([-1000, -500, 0, 500, 1000], ['-1,000', '-500', '0', '500', '1,000'])
    right.annotate('band: min to max over the seeds', (0, 0), xycoords='axes fraction', xytext=(4, 6),
                   textcoords='offset points', color=ink2, fontsize=9)

    for ax in (left, right):
        ax.set_facecolor(surface)
        ax.grid(True, color=grid, lw=1)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(axis)
        ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=surface)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seeds', type=int, help='seeds for Solution 2 (default 10, quick 3)')
    parser.add_argument('--episodes', type=int, help='training episodes per seed (default 1500, quick 600)')
    parser.add_argument('--quick', action='store_true', help='fewer seeds and episodes')
    parser.add_argument('--no-plot', action='store_true', help='do not write learning_curve.png')
    args = parser.parse_args()
    seeds = range(args.seeds or (3 if args.quick else 10))
    episodes = args.episodes or (600 if args.quick else 1500)
    ablation_seeds = range(min(len(seeds), 2 if args.quick else 5))
    generations = 60 if args.quick else 150
    started = time.perf_counter()

    print('The cave (what the agent perceives on each square):')
    check_cave()
    check_dead_reckoning()
    print('Checks passed: the percept map is the slide\'s, and dead reckoning always matches the cave.\n')

    print('=== Solution 1: hand-written classifiers, messages only ===')
    for i, c in enumerate(HAND_WRITTEN, 1):
        print(f'  {i:>2}. {c}')
    score, trace = run_episode(HAND_WRITTEN, Cave())
    assert score == sum(s.reward for s in trace)
    print()
    print_trace(trace)
    print(f'  Score: {score} in {len(trace)} actions (best possible: {BEST})\n')

    print(f'=== Solution 2: learning classifier system, {len(seeds)} seeds x {episodes} episodes ===')
    t0 = time.perf_counter()
    results = [learn(seed, episodes, generations) for seed in seeds]
    print('  seed  score  actions  first win  greedy stable  created  | used: classifiers  messages'
          '  | GA: classifiers  messages  score')
    for r in results:
        print(f'  {r.seed:>4}  {r.score:>5}  {len(r.trace):>7}  {r.first_win or "-":>9}  {r.stable_from:>13}  '
              f'{r.created:>7}  | {len(r.used):>17}  {sum(len(c.condition) for c in r.used):>8}  '
              f'| {len(r.simple):>15}  {sum(len(c.condition) for c in r.simple):>8}  {r.simple_score:>5}')
    print(f'  ({time.perf_counter() - t0:.1f} s)')
    first = results[0]
    print(f'\n  Seed {first.seed}, greedy run with the learned classifiers ({first.created} created, {len(first.used)} used):')
    print_trace(first.trace, whole_rule=False)
    print(f'\n  Seed {first.seed}, the same run after the GA: {len(first.simple)} classifiers (strongest first)')
    for c in sorted(first.simple, key=lambda c: -c.strength):
        print(f'     {str(c):42} strength {c.strength:6.1f}')
    score, trace = run_episode(first.simple, Cave(), memory=DeadReckoning())
    print()
    print_trace(trace)
    print(f'  Score: {score} in {len(trace)} actions\n')

    print(f'=== Ablation 1: learning with messages only ({len(ablation_seeds)} seeds x {episodes} episodes) ===')
    print('  Every classifier also posts one of the tags m0..m(K-1); there is no bookkeeping.')
    t0 = time.perf_counter()
    for k, greedy, wins in messages_only(ablation_seeds, episodes):
        print(f'  K={k}: greedy scores {greedy}, winning training episodes {wins}')
    print(f'  ({time.perf_counter() - t0:.1f} s)\n')

    print(f'=== Ablation 2: a GA running while the classifiers learn ({len(ablation_seeds)} seeds x {episodes} episodes) ===')
    t0 = time.perf_counter()
    rows = online_ga(ablation_seeds, episodes)
    print('  seed  greedy score  winning training episodes  classifiers')
    for seed, greedy, wins, size, _ in rows:
        print(f'  {seed:>4}  {greedy:>12}  {wins:>25}  {size:>11}')
    print(f'  ({time.perf_counter() - t0:.1f} s)')
    print(f'\n  Seed {rows[0][0]}, first steps of the greedy run:')
    print_trace(rows[0][4][:6])
    print()

    if not args.no_plot:
        path = Path(__file__).with_name('learning_curve.png')
        plot(results, path)
        print(f'Learning curve written to {path.name}')
    print(f'Total time: {time.perf_counter() - started:.0f} s')


if __name__ == '__main__':
    main()
#Perfect solution