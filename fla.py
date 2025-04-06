import time
import heapq
import matplotlib.pyplot as plt
import numpy as np
import gym
import imageio

def man_dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def pos(idx, cols):
    return (idx // cols, idx % cols)

class BnB:
    def __init__(self, env):
        self.env = env
        desc = env.unwrapped.desc
        self.rows, self.cols = desc.shape
        goal = np.where(desc == b'G')
        self.gx, self.gy = goal[0][0], goal[1][0]

    def h(self, s):
        return man_dist(pos(s, self.cols), (self.gx, self.gy))

    def search(self, timeout=600):
        t0 = time.time()
        seen = set()
        pq = []
        s0 = self.env.reset()
        if isinstance(s0, tuple): s0 = s0[0]
        s0 = int(np.argmax(s0))
        heapq.heappush(pq, (self.h(s0), 0, s0, []))

        while pq and time.time() - t0 < timeout:
            _, cost, s, path = heapq.heappop(pq)
            if s in seen: continue
            seen.add(s)

            if self.env.unwrapped.desc.flatten()[s] == b'G':
                return {"reward": 1, "steps": len(path), "path": path + [s]}

            for a in range(self.env.action_space.n):
                ns = self.env.P[s][a][0][1]
                if ns in seen: continue
                heapq.heappush(pq, (cost + 1 + self.h(ns), cost + 1, ns, path + [s]))

        return {"reward": 0, "steps": -1}

class IDA:
    def __init__(self, env):
        self.env = env
        desc = env.unwrapped.desc
        self.rows, self.cols = desc.shape
        goal = np.where(desc == b'G')
        self.gx, self.gy = goal[0][0], goal[1][0]

    def h(self, s):
        return man_dist(pos(s, self.cols), (self.gx, self.gy))

    def search(self, timeout=600):
        s0 = self.env.reset()
        s0 = int(np.argmax(s0[0]))
        bound = self.h(s0)
        t0 = time.time()

        def dfs(s, g, path, seen):
            if time.time() - t0 > timeout: return float('inf'), None
            f = g + self.h(s)
            if f > bound: return f, None
            if self.env.unwrapped.desc.flatten()[s] == b'G': return -1, path

            minf = float('inf')
            seen.add(s)
            for a in range(self.env.action_space.n):
                ns = self.env.P[s][a][0][1]
                if ns in seen: continue
                t, res = dfs(ns, g+1, path + [s], seen)
                if res is not None: return -1, res
                minf = min(minf, t)
            seen.remove(s)
            return minf, None

        while time.time() - t0 < timeout:
            t, res = dfs(s0, 0, [], set())
            if res is not None: return {"reward": 1, "steps": len(res), "path": res}
            if t == float('inf'): break
            bound = t
        return {"reward": 0, "steps": -1}

def test(agent, env, runs=5, timeout=600):
    times, rewards, steps = [], [], []
    for _ in range(runs):
        a = agent(env)
        t1 = time.time()
        res = a.search(timeout)
        times.append(time.time() - t1)
        rewards.append(res['reward'])
        steps.append(res['steps'])
    return np.mean(times), np.mean(rewards), np.mean([s for s in steps if s != -1])

def states_to_actions(path, ncol=4):
    acts = []
    for i in range(len(path)-1):
        d = path[i+1] - path[i]
        if d == 1:
            acts.append(2)  # RIGHT
        elif d == -1:
            acts.append(0)  # LEFT
        elif d == ncol:
            acts.append(1)  # DOWN
        elif d == -ncol:
            acts.append(3)  # UP
        else:
            raise ValueError(f"Invalid move: {path[i]} -> {path[i+1]}")
    return acts

#gif
def render_frozen_lake_path(state_path, name="frozenlake.gif"):
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
    env.reset()
    frames = []

    for s in state_path:
        env.unwrapped.s = s  
        frame = env.render()
        frames.append(frame)

    imageio.mimsave(name, frames, duration=1.5)
    print(f"Saved {name}")

def main():
    env = gym.make("FrozenLake-v1", is_slippery=False)
    t1, r1, s1 = test(BnB, env)
    t2, r2, s2 = test(IDA, env)

    plt.bar(['BnB', 'IDA*'], [t1, t2], color=['blue', 'green'])
    plt.ylabel("Avg Time (s)")
    plt.title("Avg Time to Reach Goal")
    plt.show()

    print("BnB -> Time:", t1, "Reward:", r1, "Steps:", s1)
    print("IDA* -> Time:", t2, "Reward:", r2, "Steps:", s2)

    bnb = BnB(env)
    res1 = bnb.search()
    if res1["reward"] == 1:
        act1 = res1["path"]
        render_frozen_lake_path(act1, "bnb.gif")

    ida = IDA(env)
    res2 = ida.search()
    if res2["reward"] == 1:
        act2 = res2["path"]
        act2.append(15)
        #print(act2)
        render_frozen_lake_path(act2, "ida.gif")

if __name__ == '__main__':
    main()