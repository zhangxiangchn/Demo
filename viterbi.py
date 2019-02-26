states = ('Rainy', "Sunny")

observation = ('walk', 'shop', 'clean')

start_probability = {'Rainy':0.6, 'Sunny':0.4}

transfrom_probability = {'Rainy':{'Rainy':0.7, 'Sunny':0.3},
                        "Sunny":{"Rainy":0.4, "Sunny":0.6},}

emission_probability = {'Rainy':{'walk':0.1, "shop":0.4, "clean":0.5},
                        "Sunny":{"walk":0.6, "shop":0.3, "clean":0.1},}

def print_path(V):
    """
    print the path
    """
    print(" ", end=' ')
    for i in range(len(V)):
        print("%7d"%i, end=' ')
    print(' ')

    for y in V[0].keys():
        print('%.5s'%y, end=' ')
        for t in range(len(V)):
            print("%.7s"%("%f" % V[t][y]), end=' ')
        print(' ')

def viterbi(obs, states, start_p, trans_p, emiss_p):
    """
    params:obs:  观察序列
    params:states:   隐状态序列
    params:start_p: 开始序列（隐状态）
    params:trans_p: 状态转移序列（隐状态）
    params:emiss_p: 发射序列（隐->显）
    """
    # 路径概率表， V[时间][隐状态] = 概率
    V = [{}]
    # 中间变量 保存当前的隐状态
    path = {}

    # 计算第一天的概率
    for y in states:
        V[0][y] = start_p[y]*emiss_p[y][obs[0]]
        path[y] = [y]

    # 运行vbt算法，t>0
    for i in range(1, len(obs)):
        V.append({})
        newpath = {}

        for y in states:
            # 概率 隐状态 = 前状态是y0的概率*y0到y1状态的概率*y是当前状态的概率
            (prob, state) = max([(V[i-1][y0]*trans_p[y0][y]*emiss_p[y][obs[i]], y0) for y0 in states])
            V[i][y] = prob

            newpath[y] = path[state] + [y]

        path = newpath

    print_path(V)
    (prob, state) = max([(V[len(obs) - 1][y], y) for y in states])
    return (prob, path[state])

def example():
    return viterbi(observation,
                   states,
                   start_probability,
                   transfrom_probability,
                   emission_probability)

print(example())
