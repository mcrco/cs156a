import random
import math

def find_slope(x1, x2):
    return (x1 * math.sin(math.pi * x1) + x2 * math.sin(math.pi * x2)) / (x1 * x1 + x2 * x2)

slopes = []
for _ in range(10000):
    x1 = random.uniform(-1, 1)
    x2 = random.uniform(-1, 1)
    slopes.append(find_slope(x1, x2))

a = sum(slopes) / len(slopes)
print(a)

errs = []
for _ in range(10000):
    x1 = random.uniform(-1, 1)
    x2 = random.uniform(-1, 1)
    err = 0.5 * ((a * x1 - math.sin(math.pi * x1)) ** 2 + (a * x2 - math.sin(math.pi * x2)) ** 2)
    errs.append(err)

bias = sum(errs) / len(errs)
print(bias)

diffs = []
for _ in range(10000):
    x1 = random.uniform(-1, 1)
    x2 = random.uniform(-1, 1)
    x3 = random.uniform(-1, 1)
    x4 = random.uniform(-1, 1)
    m = find_slope(x1, x2)
    diff = 0.5 * ((a * x3 - m * x3) ** 2 + (a * x4 - m * x4) ** 2)
    diffs.append(diff)

var = sum(diffs) / len(diffs)
print(var)
