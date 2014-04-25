# vim: ts=4:sw=4

import time
import random
import numpy
import itertools


def A(xs):
    return numpy.array(xs, copy=True)


def _min_by(f, xs):
    best = xs[0]
    for x in xs:
        if f(x) < f(best):
            best = x
    return best


def _max_by(f, xs):
    best = xs[0]
    for x in xs:
        if f(x) > f(best):
            best = x
    return best


def _random_vector_in(ranges):
    return map(lambda r: random.uniform(*r), ranges)


class Particle(object):
    def __init__(self, domains):
        num_dims = len(domains)
        self.pos = A(_random_vector_in(domains))
        self.vel = A([random.uniform(-1, 1) for k in range(num_dims)])
        self.best = A(self.pos)

    @property
    def vel_len(self):
        return numpy.linalg.norm(self.vel)

    def __str__(self):
        return '%s, %s, %s' % (self.pos, self.vel, self.best,)


def optimize(_obj_fn, domains,
             num_particles, neighborhood_size,
             start_inertia, trust_neighbors, trust_self,
             inertia_dec_factor=0.99, max_vel_scale=1.0, max_time=1e12,
             create_particle=Particle,
             should_clamp_pos=True,
             should_clamp_vel=False,
             particle_mod_fn=None,
             accepted_minima=None,
             max_num_fitness_evaluations=None):

    evaluated = set()
    def obj_fn(xs):
        fingerprint = tuple(xs)
        evaluated.add(fingerprint)
        return _obj_fn(xs)

    swarm = [create_particle(domains) for k in range(num_particles)]
    best_neighbor_pos = [None for p in swarm]

    time_start = time.time()
    last_report_time = 0

    inertia = start_inertia

    for iter_num in itertools.count(0):
        time_now = time.time()

        best_fitness = obj_fn(_min_by(obj_fn, [p.best for p in swarm]))
        worst_fitness = (0 if best_neighbor_pos[0] is None else
                          obj_fn(_max_by(obj_fn, best_neighbor_pos)))
        min_vel = min(p.vel_len for p in swarm)
        avg_vel = sum(p.vel_len for p in swarm)/num_particles
        max_vel = max(p.vel_len for p in swarm)
        def dist(a, b):
            return numpy.linalg.norm(a.pos - b.pos)
        diameter = max(dist(a, b) for a in swarm for b in swarm)

        if iter_num >= 1 and time_now-time_start >= max_time:
            break
        if not accepted_minima is None and best_fitness <= accepted_minima:
            break
        #if max_vel < 1e-10:
        #    break
        num_fitness_evaluations = len(evaluated)
        if (not max_num_fitness_evaluations is None and
           num_fitness_evaluations > max_num_fitness_evaluations):
            break

        if time_now-last_report_time >= 0.1:
            print 'TIME: %-3.2f  ITER_NUM: %-4d  MIN_FIT: %-.7f MAX_FIT: %-7.4f  VEL_MIN:%-.4e VEL_AVG:%-.4e VEL_MAX: %-.4e  DIAMETER: %-.2f %d' % (
                time_now-time_start,
                iter_num,
                best_fitness,
                worst_fitness,
                min_vel,
                avg_vel,
                max_vel,
                diameter,
                num_fitness_evaluations,)
            last_report_time = time_now

        if inertia > 0.6:
            inertia *= inertia_dec_factor

        # Update neighborhood bests
        for i in range(num_particles):
            ids = [(i+j) % num_particles for j in range(neighborhood_size)]
            positions = [swarm[i].best for i in ids]

            alt_best = A( _min_by(obj_fn, positions) )
            if (best_neighbor_pos[i] is None or
                obj_fn(alt_best) < obj_fn(best_neighbor_pos[i])):
                best_neighbor_pos[i] = alt_best

        for i, p in enumerate(swarm):
            # Update velocity
            r1 = A([random.uniform(0, 1) for k in domains])
            r2 = A([random.uniform(0, 1) for k in domains])

            p.vel =  (   inertia * p.vel \
                       + trust_neighbors * r1 * (best_neighbor_pos[i] - p.pos)\
                       + trust_self      * r2 * (p.best - p.pos))

            # Clamp velocity
            if should_clamp_vel:
                for j, (low, high,) in enumerate(domains):
                    max_vel = max_vel_scale * (high-low)
                    alt = max(-max_vel, min(max_vel, p.vel[j]))
                    if p.vel[j] != alt:
                        p.vel[j] = alt + random.uniform(-.1, .1)

            # Update position
            p.pos += p.vel

            # Perform modification function, if enabled
            if particle_mod_fn:
                particle_mod_fn(p)

            # Clamp position, if enabled
            if should_clamp_pos:
                for j, (low, high,) in enumerate(domains):
                    p.pos[j] = max(low, min(p.pos[j], high))

            # Update particle's best
            if obj_fn(p.pos) < obj_fn(p.best):
                p.best = A( p.pos )

    return _min_by(obj_fn, [p.best for p in swarm]), iter_num


def test():
    booth = lambda (x, y): (x+2*y-7)**2 + (2*x+y-5)**2
    rosenbrock = lambda (x, y): (1-x)**2 + 100*(y-x**2)**2

    f = rosenbrock
    sol = optimize(
        f,
        ((-1e4, 1e4), (-1e4, 1e4),),
        num_particles=30, neighborhood_size=8,
        start_inertia=1, trust_self=2, trust_neighbors=2,
        max_time=400,
        inertia_dec_factor=0.99,
        should_clamp_vel=True,
        should_clamp_pos=True,
        accepted_minima=0,
        max_vel_scale=8
    )
    print sol, '%.5e' % f(sol[0])


if __name__ == '__main__':
    test()
