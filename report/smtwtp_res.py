from math import sqrt

NO_LOC = (
  (913, [956, 956, 956, 956, 930, 930, 956, 956]),
  (1225, [1351, 1264, 1535, 1374, 1374, 1374, 1374, 1374]),
  (537, [573, 573, 573, 573, 573, 573, 573, 573]),
  (2094, [2094, 2127, 2254, 2098, 2094, 2307, 2094, 2121]),
  (990, [990, 990, 990, 990, 990, 990, 990, 990]),
  (6955, [7024, 6955, 7484, 6955, 6955, 6955, 6955, 6955]),
  (6324, [6571, 6571, 6447, 6324, 6571, 6324, 6571, 6571]),
  (6865, [6927, 6865, 6865, 6893, 6865, 6865, 6865, 6865]),
  (16225, [16225, 16472, 16468, 16348, 16758, 16740, 16530, 16706]),
  (9737, [9771, 9899, 9741, 9771, 9771, 9903, 9741, 9771]),
  (17465, [17562, 17465, 17966, 17465, 17876, 17562, 17562, 17562]),
)

LOC = (
  (913, [930, 913, 930, 930, 930, 956, 956, 913]),
  (1225, [1225, 1225, 1225, 1263, 1243, 1225, 1225, 1225]),
  (537, [573, 573, 573, 573, 537, 573, 573, 573]),
  (2094, [2094, 2094, 2094, 2094, 2094, 2094, 2094, 2094]),
  (990, [990, 990, 990, 990, 990, 990, 990, 990]),
  (6955, [6955, 6955, 6955, 6955, 6955, 6955, 6955, 6955]),
  (6324, [6324, 6324, 6324, 6571, 6437, 6324, 6324, 6324]),
  (6865, [6865, 6865, 6865, 6865, 6865, 6865, 6865, 6865]),
  (16225, [16225, 16225, 16225, 16225, 16225, 16225, 16225, 16225]),
  (9737, [9741, 9771, 9737, 9741, 9737, 9741, 9741, 9741]),
  (17465, [17465, 17465, 17465, 17465, 17465, 17465, 17465, 17465]),
)

print r'$\textbf{\#}$ &',
print r'$\mathbf{Alg.}$ &',
print ' & ' .join(r'$\mathbf{g}_\text{'+w+'}$' for w in
  r'opt $\min$ avg $\max$ $\sigma$'.split()), r'\\'

all_data = zip(NO_LOC, LOC,)[:10]
for i, ((opt, xs,), (opt2, xs2,)) in enumerate(all_data, start=1):
  assert len(xs) == len(xs2) == 8

  mean = sum(xs)/len(xs)
  mean2 = sum(xs2)/len(xs2)
  stddev = sqrt( sum((x-mean)**2 for x in xs)/(len(xs)) )
  stddev2 = sqrt( sum((x-mean2)**2 for x in xs2)/(len(xs2)) )
  stddev = '%.2f' % stddev
  stddev2 = '%.2f' % stddev2

  # "decorate if optimum" ;)
  difopt = lambda x: x #(r'\textbf{'+str(x)+'}') if x == opt else x

  def rc(c, v, s):
    return r'\rowcolor{'+c+'}'+s if v == opt else s

  print r'\hline'
  print rc('pink', min(xs),
      ' & '.join(map(str, (r'\textbf{'+str(i)+'}', r'\textsc{pso}', opt,
        difopt(min(xs)), difopt(mean), difopt(max(xs)), stddev,))) + r'\\')
  print r'\hline'
  print rc('pink', min(xs2),
      ' & '.join(map(str, (' ', r'\textsc{pso+ls}', opt, difopt(min(xs2)),
    difopt(mean2), difopt(max(xs2)), stddev2,))) + r'\\')
  print r'\hline'
  if i != len(all_data):
    print r'\scriptsize{~}\\'

