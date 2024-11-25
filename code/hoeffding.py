import random

import numpy as np
from tqdm import tqdm

def sim():
    coin_flips = [
        [random.choice([True, False]) for _ in range(10)] for _ in range(1000)
    ]
    counts = [coin.count(True) for coin in coin_flips]
    fracs = [counts[0] / 10, counts[random.randint(0, 999)] / 10, min(counts)]

    return fracs


distributions = np.array([sim() for _ in tqdm(range(100000))])
means = np.mean(distributions, 0)
print(means.shape)
print(f"v_1: {means[0]}")  # 0.49982
print(f"v_rand: {means[1]}")  # 0.50055
print(f"v_min: {means[2]}")  # 0.37517
