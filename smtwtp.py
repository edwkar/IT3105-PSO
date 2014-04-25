# vim: ts=4:sw=4

from pso import optimize
import numpy

def solve(problem):
    proc_times, weights, due_times, opt_sol = problem

    memo = {}
    def total_weighted_tardiness(schedule):
        key = tuple(schedule)
        if not key in memo:
            res = 0.0
            t = 0
            for j in schedule:
                s_j = t
                T_j = s_j + proc_times[j] - due_times[j]
                if T_j > 0:
                    res += weights[j] * T_j
                t += proc_times[j]
            memo[key] = res
        return memo[key]

    def eval_fitness(solution):
        schedule = numpy.argsort(solution)
        best = total_weighted_tardiness(schedule)
        return best

    def local_search(p):
        if random.random() < .9:
            return
        orig_sol = numpy.array(p.pos, copy=True)
        best_score = eval_fitness(orig_sol)
        best = orig_sol
        for k in range(15):
            v = random.randint(0, len(orig_sol)-1)
            w = random.randint(0, len(orig_sol)-1)
            s = numpy.array(orig_sol)
            s[v], s[w] = s[w], s[v]
            score = eval_fitness(s)
            if score < best_score:
                best_score = score
                best = s
        p.pos = best

    N = len(proc_times)
    best_sol = optimize(eval_fitness,
                        N * ((-1.0, 1.0,),),
                        num_particles=80, neighborhood_size=5,
                        start_inertia=1,
                        trust_self=1.5,
                        trust_neighbors=1.5,
                        max_time=40,
                        #particle_mod_fn=local_search,
                        should_clamp_vel=True,
                        should_clamp_pos=False)[0]

    print 1.05*opt_sol
    #raw_input()
    return eval_fitness(best_sol)



def main():
    import time
    ts = time.time()

    problem_data = map(int, open('data/wt40.txt').read().split())
    opt_sol_data = map(int, open('data/wtopt40.txt').read().split())
    unshift_many = lambda xs, n: [xs.pop(0) for k in range(n)]

    with open('out', 'w') as f:
        f.write('')

    N = 40
    k = 0
    num_good = num_bad = 0
    xs_all = []
    while k < 8:
        proc_times = unshift_many(problem_data, N)
        weights = unshift_many(problem_data, N)
        due_times = unshift_many(problem_data, N)
        opt_sol = opt_sol_data.pop(0)
        problem = proc_times, weights, due_times, opt_sol
        if opt_sol == 0:
            continue

        xs = []
        for o in range(8):
            import time
            time.sleep(0.5)
            r = solve(problem)
            print '----', opt_sol, r
            print '----', opt_sol, r
            print '----', opt_sol, r
            xs.append(r)

        with open('out', 'a') as f:
            f.write(str((opt_sol, xs,)) + '\n')


if __name__ == '__main__':
    import random
    random.seed(42)
    main()

