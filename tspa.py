import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt

def mk_env(n=20, seed=None):
    if seed: np.random.seed(seed)
    return np.random.rand(n, 2) * 100

class hc:
    def __init__(self, pts):
        self.pts = pts
        self.n = len(pts)

    def get_cost(self, p):
        return sum(np.linalg.norm(self.pts[p[i]] - self.pts[p[(i+1)%self.n]]) for i in range(self.n))

    def search(self, t=5):
        st = time.time()
        p = list(range(self.n))
        random.shuffle(p)
        best = p[:]
        best_c = self.get_cost(p)

        while time.time() - st < t:
            i, j = sorted(random.sample(range(self.n), 2))
            p[i:j] = reversed(p[i:j])
            c = self.get_cost(p)
            if c < best_c:
                best = p[:]
                best_c = c
            else:
                p[i:j] = reversed(p[i:j])
        return {"c": best_c, "p": best}

class sa:
    def __init__(self, pts):
        self.pts = pts
        self.n = len(pts)

    def get_cost(self, p):
        return sum(np.linalg.norm(self.pts[p[i]] - self.pts[p[(i+1)%self.n]]) for i in range(self.n))

    def search(self, t=5):
        st = time.time()
        T = 100
        cool = 0.995
        p = list(range(self.n))
        random.shuffle(p)
        best = p[:]
        best_c = self.get_cost(p)

        while time.time() - st < t and T > 1e-3:
            i, j = sorted(random.sample(range(self.n), 2))
            p[i:j] = reversed(p[i:j])
            c = self.get_cost(p)
            d = c - best_c
            if d < 0 or math.exp(-d / T) > random.random():
                if c < best_c:
                    best = p[:]
                    best_c = c
            else:
                p[i:j] = reversed(p[i:j])
            T *= cool
        return {"c": best_c, "p": best}

def draw(c, path, ttl):
    x = [c[i][0] for i in path] + [c[path[0]][0]]
    y = [c[i][1] for i in path] + [c[path[0]][1]]
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, 'o-', markersize=5)
    plt.title(ttl)
    plt.show()

def run_all(cls, c, n=5, t=5):
    cs = []
    for _ in range(n):
        a = cls(c)
        res = a.search(t)
        cs.append(res["c"])
    return np.mean(cs), res["p"]

def main():
    n = 20
    t = 5
    runs = 5
    pts = mk_env(n, seed=42)

    hc_avg, hc_path = run_all(hc, pts, runs, t)
    sa_avg, sa_path = run_all(sa, pts, runs, t)

    print(f"hc avg: {hc_avg:.2f}")
    print(f"sa avg: {sa_avg:.2f}")

    draw(pts, hc_path, "hc tour")
    draw(pts, sa_path, "sa tour")

    plt.figure(figsize=(6, 4))
    plt.bar(["hc", "sa"], [hc_avg, sa_avg], color=["lightblue", "lightcoral"])
    plt.ylabel("avg cost")
    plt.title("tsp avg cost comp")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

