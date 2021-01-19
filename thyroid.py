import math
from mlp import MLP


def load_data_set(file):
    raw = []
    result = []
    with open(file, 'r') as f:
        raw = f.read().split('\n')

    for item in raw:
        if item == '':
            continue

        tmp = [float(sub_item) for sub_item in item.split(', ')]
        result.append({'inputs': tmp[:-1], 'target': tmp[-1]})

    return result


if __name__ == "__main__":
    network = MLP(21, 10, 3, 0.1)

    train_data = []
    test_data = []
    validation_data = []

    data = load_data_set('./thyroid.txt')
    length = len(data)
    train_data = data[:math.floor(length * .7)]  # 70%
    test_data = data[math.floor(length * .7): math.floor(length * .85)]  # 15%
    validation_data = data[math.floor(length * .85):]  # 15%

    del length, data  # prevent memory leak.

    network.backward(train_data)

    good = 0
    bad = 0
    for item in test_data:
        result = network.forward(item['inputs'])
        # print(str(result) + ' ' + str(item['target']))
        if result.index(max(result)) + 1 == item['target']:
            good = good + 1
        else:
            bad = bad + 1

    print(f'good: {str(good)} bad: {str(bad)}')
