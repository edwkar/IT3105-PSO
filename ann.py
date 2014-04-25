# vim: ts=4:sw=4


import math
import pso
import random
import pso


def compile_feed_fw_network(num_inputs, layers):
    def unit_var_name(i, j):
        return 'a_%d_%d' % (i, j,)

    stmts = []

    for i, units in enumerate(layers, start=1):
        for j, (weights, act_fn_compiler,) in enumerate(units):
            act_in = '+'.join('%e*%s' % (w, unit_var_name(i-1, k),)
                              for k, w in enumerate(weights))
            vn = unit_var_name(i, j)
            stmts.append('%s = %s' % (vn, act_in,))
            stmts.append('%s = %s' % (vn, act_fn_compiler(vn),))

    last_layer_id = len(layers)
    num_outputs = len(layers[last_layer_id-1])
    stmts.append('return (%s,)' % ','.join(unit_var_name(last_layer_id, j)
                                           for j in range(num_outputs)))

    fn_header = 'def FFN(%s): ' % ','.join(unit_var_name(0, j)
                                           for j in range(num_inputs))

    raw_code = fn_header + ('; '.join(stmts))
    byte_code = compile(raw_code, '<string>', 'exec', dont_inherit=True)

    locals_ = {}
    eval(byte_code, {}, locals_)
    ffn = locals_['FFN']

    def ffn_biased(*v):
        return ffn(*([-1.0] + list(v)))
    return ffn_biased


def sigmoid(var_name):
    return '(1.0/(1.0+(%.12e)**-%s))' % (math.e, var_name,)


def linear(var_name):
    return var_name




def ann_classifier_meta_factory(num_inputs, hidden_layer_size, num_classes):
    def build_classifier(_weights):
        weights = [w for w in _weights]
        assert len(weights) == ((num_inputs*hidden_layer_size) +
                                (num_classes*hidden_layer_size))
        hidden_layer = []
        for j in range(hidden_layer_size):
            ws = [weights.pop(0) for k in range(num_inputs)]
            hidden_layer.append((ws, sigmoid,))

        output_layer = []
        for j in range(num_classes):
            ws = [weights.pop(0) for k in range(hidden_layer_size)]
            output_layer.append((ws, sigmoid,))

        assert len(weights) == 0
        layers = [hidden_layer, output_layer]
        return compile_feed_fw_network(num_inputs, layers)

    return build_classifier


def evaluate_classifier(classifier, data_set):
    num_failed = 0
    xnum_failed = 0
    ynum_failed = 0
    for data, actual_class in data_set:
        res = classifier(*data)
        best_class, best_val = 0, res[0]
        a = b = 0
        for class_, val in enumerate(res):
            if val > best_val:
                best_class = class_
                a = val
            else:
                b = max(val, b)
        diff = a-b
        predicted_class = best_class

        if predicted_class != actual_class:
            num_failed += 1
            #ynum_failed += res[actual_class]
        else:
            xnum_failed += diff**2
    v = float(num_failed)#/len(data_set)
    #v = num_failed + ynum_failed/1000.0
    #assert int(v) == num_failed
    return v


def read_and_normalize_file(filepath):
    data = []
    for line in open(filepath):
        ls = line.split(',')
        num_features = len(ls) - 1
        features = map(float, ls[:num_features])
        class_ = int(ls[num_features])
        data.append((features, class_,))
    for k in range(num_features):
        vals = [xs[0][k] for xs in data]
        k_min, k_max = min(vals), max(vals)
        if k_max == 0:
            assert k_min == 0
            continue
        for xs in data:
            xs[0][k] = (xs[0][k]-k_min) / (k_max-k_min)
            xs[0][k] *= 0.8
            xs[0][k] += 0.1
    num_classes = max(x[1] for x in data) + 1
    return data, num_features, num_classes


def test(filepath, max_num_fitness_evaluations, num_hidden):
    data, num_features, num_classes = read_and_normalize_file(filepath)

    num_inputs = num_features+1
    num_outputs = num_classes

    build_classifier = ann_classifier_meta_factory(
                            num_inputs, num_hidden, num_outputs)
    num_weights = num_inputs*num_hidden+num_hidden*num_outputs

    random.shuffle(data)

    cut = int(3*len(data)/5.0)
    training, testing = data[:cut], data[cut:]
    print len(data), cut, len(training), len(testing)

    memo = {}
    def evalit(xs):
        t = tuple(xs)
        if not t in memo:
            c = build_classifier(xs)
            memo[t] = evaluate_classifier(c, training)
        return memo[t]

    weights = pso.optimize(
        evalit,
        domains=num_weights*[[-8, 8]],
        num_particles=60, neighborhood_size=5,
        start_inertia=1, trust_self=2.4, trust_neighbors=2.4,
        #max_time=60,
        #accepted_minima=0,
        max_vel_scale=2.0,
        inertia_dec_factor=0.99,
        should_clamp_vel=False,
        max_num_fitness_evaluations=max_num_fitness_evaluations
    )[0]
    print weights
    c = build_classifier(weights)

    testing_err = evaluate_classifier(c, testing)/float(len(testing))
    training_err = evaluate_classifier(c, training)/float(len(training))
    print 'TRAINING ERRORS:', evaluate_classifier(c, training), '/', len(training), training_err
    print 'TEST     ERRORS:', evaluate_classifier(c, testing), '/', len(testing), testing_err
    return ( (int(evaluate_classifier(c, training)),
              int(evaluate_classifier(c, testing))) )

"""
with open('iris.res', 'w') as f:
    f.write('')
for max_ev in [100, 200, 300, 500, 1000, 2000, 3000, 5000, 7000, 10000, 12500, 15000,
        17500, 20000, 22500, 25000]:
    res = []
    for k in range(100):
        print max_ev
    for v in range(25):
        res.append( test('data/iris.data', max_ev, 3) )
        print res
    with open('iris.res', 'a') as f:
        f.write(str((max_ev, res)))
        f.write('\n')


with open('seeds.res', 'w') as f:
    f.write('')
for max_ev in [100, 200, 300, 500, 1000, 2000, 3000, 5000, 7500, 10000, 12500, 15000,
        17500, 20000, 22500, 25000, 27500, 30000]:
    res = []
    for k in range(100):
        print max_ev
    for v in range(25):
        res.append( test('data/seeds.csv', max_ev, 5) )
        print res
    with open('seeds.res', 'a') as f:
        f.write(str((max_ev, res)))
        f.write('\n')

"""

with open('glass.res', 'w') as f:
    f.write('')
for max_ev in [100, 200, 300, 500, 1000, 2000, 3000, 5000, 7500, 10000, 12500, 15000,
        17500, 20000, 22500, 25000, 27500, 30000]:
    res = []
    for k in range(100):
        print max_ev
    for v in range(10):
        res.append( test('data/glass.data', max_ev, 5) )
        print res
    with open('glass.res', 'a') as f:
        f.write(str((max_ev, res)))
        f.write('\n')
