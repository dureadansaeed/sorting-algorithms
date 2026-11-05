"""
The Midnight Queue - interactive web app (Masha & the Bear edition).   24i-3037, section C

    python midnight_app.py          ->  opens http://localhost:8000 in your browser

Needs: creditcard.csv next to this file, and  pip install numpy pandas
Everything (the hand-written sorts, the top-K methods, the live-feed strategies, the timing server and the
web page) is in this one file. The big multi-hour benchmark is embedded as a replay (results.csv is used instead if present).
"""
# ============================ common: roll-number params, keys, input conditions ============================
"""Shared stuff: roll-number params, loading the csv, building keys and the six input conditions."""
import os
import random

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "creditcard.csv")
CACHE = os.path.join(HERE, "cache.npz")

ROLL = 243037                 # 24i-3037, digits only
SEED = ROLL
K_MINE = 25 + (ROLL % 226)    # = 112
SIZES = [1000, 10000, 100000, 284807]
CONDS = ["C1", "C2", "C3", "C4", "C5", "C6"]
COND_NAMES = {"C1": "Arrival", "C2": "Shuffled", "C3": "Pre-ranked",
              "C4": "Reverse-ranked", "C5": "Nearly ranked", "C6": "Band-only"}

# key layout (smaller key = comes earlier in triage order), 63 bits in total:
# [ 9-band : 4 ][ CMAX - cents : 22 ][ time rank : 18 ][ txn_id : 19 ]
CMAX = (1 << 22) - 1


def build_cache(force=False):
    """Read the csv once, compute risk/band/key and stash everything in cache.npz."""
    if os.path.exists(CACHE) and not force:
        return
    import pandas as pd
    df = pd.read_csv(CSV)
    V = df[[f"V{i}" for i in range(1, 29)]].to_numpy(dtype=np.float64)
    # population std (ddof=0) since the "full file" is the whole population we care about
    z = (V - V.mean(axis=0)) / V.std(axis=0)
    risk = (z ** 2).sum(axis=1)
    band = np.clip(np.floor(np.log2(risk)), 0, 9).astype(np.int64)
    time = df["Time"].to_numpy(dtype=np.float64)
    amount = df["Amount"].to_numpy(dtype=np.float64)
    cents = np.rint(amount * 100).astype(np.int64)
    txn = np.arange(1, len(df) + 1, dtype=np.int64)
    trank = np.unique(time, return_inverse=True)[1].astype(np.int64)
    key = ((((9 - band) << 22) | (CMAX - cents)) << 18 | trank) << 19 | txn
    np.savez(CACHE, risk=risk, band=band, time=time, amount=amount, cents=cents,
             txn=txn, key=key, cls=df["Class"].to_numpy(dtype=np.int64))


def load_cache():
    build_cache()
    d = np.load(CACHE)
    return {k: d[k] for k in d.files}


def reference_order(c):
    """Triage order with plain numpy lexsort (verification only): returns row indices."""
    return np.lexsort((c["txn"], c["time"], -c["amount"], -c["band"]))


def subsample_idx(N, n):
    if n >= N:
        return np.arange(N)
    # file is already in Time order so sorting the indices restores Time order
    return np.sort(np.random.default_rng(SEED).choice(N, n, replace=False))


def near_ranked(c3, seed=SEED):
    """C3 with 1% of the records moved to random positions."""
    n = len(c3)
    m = max(1, round(0.01 * n))
    rng = np.random.default_rng(seed)
    arr = np.array(c3, dtype=np.int64)
    moved = rng.choice(n, m, replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[moved] = True
    targets = rng.choice(n, m, replace=False)
    tmask = np.zeros(n, dtype=bool)
    tmask[targets] = True
    out = np.empty(n, dtype=np.int64)
    out[targets] = arr[moved]
    out[~tmask] = arr[~mask]
    return out.tolist()


def make_conditions(c, n):
    """Returns dict C1..C6 -> list of int keys (all built from the same n records)."""
    idx = subsample_idx(len(c["key"]), n)
    c1 = c["key"][idx].tolist()
    c2 = c1[:]
    random.Random(SEED).shuffle(c2)
    c3 = sorted(c1)               # prep only, not one of the contenders
    c4 = c3[::-1]
    c5 = near_ranked(c3)
    c6 = (9 - c["band"][idx]).tolist()
    return {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C6": c6}, idx

# ============================ the contenders (all written by hand, no built-in sort) ============================

import random

SEED = 243037


# ---------------------------------------------------------------- insertion sort
def insertion_sort(a):
    cmp = mv = 0
    for i in range(1, len(a)):
        x = a[i]
        j = i - 1
        while j >= 0:
            cmp += 1
            if x < a[j]:
                a[j + 1] = a[j]
                mv += 1
                j -= 1
            else:
                break
        if j + 1 != i:          # only write if x actually moved
            a[j + 1] = x
            mv += 1
    return cmp, mv


# ---------------------------------------------------------------- merge sort (top-down, ping-pong buffers)
def merge_sort(a):
    n = len(a)
    if n < 2:
        return 0, 0
    aux = a[:]                  # n moves to set up the second buffer
    cmp, mv = _msort(aux, a, 0, n)
    return cmp, mv + n


def _msort(src, dst, lo, hi):
    # both src and dst hold the same unsorted values on [lo, hi) when we get here
    if hi - lo < 2:
        return 0, 0
    mid = (lo + hi) // 2
    c1, m1 = _msort(dst, src, lo, mid)   # roles swapped: sorted halves end up in src
    c2, m2 = _msort(dst, src, mid, hi)
    i, j, k = lo, mid, lo
    cmp = 0
    while i < mid and j < hi:
        cmp += 1
        if src[j] < src[i]:     # strict, so ties take the left one -> stable
            dst[k] = src[j]
            j += 1
        else:
            dst[k] = src[i]
            i += 1
        k += 1
    mv = k - lo
    if i < mid:
        dst[k:hi] = src[i:mid]
        mv += mid - i
    elif j < hi:
        dst[k:hi] = src[j:hi]
        mv += hi - j
    return c1 + c2 + cmp, m1 + m2 + mv


# ---------------------------------------------------------------- quick sort, naive
def quick_naive(a):
    """Last element as pivot, Lomuto partition. Explicit stack instead of recursion,
    otherwise sorted input (depth n) blows Python's recursion limit."""
    cmp = mv = 0
    stack = [(0, len(a) - 1)]
    while stack:
        lo, hi = stack.pop()
        if lo >= hi:
            continue
        p = a[hi]
        i = lo
        for j in range(lo, hi):
            if a[j] < p:
                a[i], a[j] = a[j], a[i]
                mv += 2
                i += 1
        cmp += hi - lo          # exactly one comparison per loop iteration
        a[i], a[hi] = a[hi], a[i]
        mv += 2
        stack.append((lo, i - 1))
        stack.append((i + 1, hi))
    return cmp, mv


# ---------------------------------------------------------------- quick sort, hardened
def quick_hardened(a, seed=SEED):
    """Random pivot + three-way (Dijkstra) partition, loop on the smaller side and
    push the bigger one, so the stack stays O(log n). Not stable."""
    cmp = mv = 0
    rnd = random.Random(seed).randrange
    stack = [(0, len(a) - 1)]
    while stack:
        lo, hi = stack.pop()
        while lo < hi:
            p = a[rnd(lo, hi + 1)]
            lt, i, gt = lo, lo, hi
            while i <= gt:
                x = a[i]
                cmp += 1
                if x < p:
                    a[i] = a[lt]
                    a[lt] = x
                    mv += 2
                    lt += 1
                    i += 1
                else:
                    cmp += 1
                    if p < x:
                        a[i] = a[gt]
                        a[gt] = x
                        mv += 2
                        gt -= 1
                    else:
                        i += 1
            # [lo, lt) < p   [lt, gt] == p   (gt, hi] > p
            if lt - lo < hi - gt:
                stack.append((gt + 1, hi))
                hi = lt - 1
            else:
                stack.append((lo, lt - 1))
                lo = gt + 1
    return cmp, mv


# ---------------------------------------------------------------- heap sort
def _sift_max(a, i, x, n, force):
    """Sift x down from slot i in a max-heap of size n (hole style: children move up)."""
    cmp = mv = 0
    start = i
    while True:
        c = 2 * i + 1
        if c >= n:
            break
        if c + 1 < n:
            cmp += 1
            if a[c] < a[c + 1]:
                c += 1
        cmp += 1
        if x < a[c]:
            a[i] = a[c]
            mv += 1
            i = c
        else:
            break
    if i != start or force:
        a[i] = x
        mv += 1
    return cmp, mv


def heap_sort(a):
    n = len(a)
    cmp = mv = 0
    for s in range(n // 2 - 1, -1, -1):       # Floyd bottom-up build
        c, m = _sift_max(a, s, a[s], n, False)
        cmp += c
        mv += m
    for end in range(n - 1, 0, -1):
        x = a[end]
        a[end] = a[0]                          # biggest goes to its final slot
        mv += 1
        c, m = _sift_max(a, 0, x, end, True)
        cmp += c
        mv += m
    return cmp, mv


# ---------------------------------------------------------------- top-K
def _sift_min(h, i, x, n, force):
    cmp = mv = 0
    start = i
    while True:
        c = 2 * i + 1
        if c >= n:
            break
        if c + 1 < n:
            cmp += 1
            if h[c + 1] < h[c]:
                c += 1
        cmp += 1
        if h[c] < x:
            h[i] = h[c]
            mv += 1
            i = c
        else:
            break
    if i != start or force:
        h[i] = x
        mv += 1
    return cmp, mv


def topk_heap(a, K):
    """Size-K min-heap on negated keys, so the root is the weakest of the K best
    records seen so far. Gives back (top K best-first, comparisons, moves)."""
    n = len(a)
    K = min(K, n)
    if K <= 0:
        return [], 0, 0
    h = [-x for x in a[:K]]
    cmp, mv = 0, K
    for s in range(K // 2 - 1, -1, -1):
        c, m = _sift_min(h, s, h[s], K, False)
        cmp += c
        mv += m
    for idx in range(K, n):
        nk = -a[idx]
        cmp += 1
        if h[0] < nk:           # new record is better than the weakest kept one
            c, m = _sift_min(h, 0, nk, K, True)
            cmp += c
            mv += m
    out, c, m = _drain(h)
    return out, cmp + c, mv + m


def _drain(h):
    """Pop the whole min-heap (destroys h). Weakest record comes out first, so the
    result is filled from the back and ends up best-first."""
    size = len(h)
    out = [0] * size
    cmp = mv = 0
    for pos in range(size - 1, -1, -1):
        out[pos] = -h[0]
        mv += 1
        size -= 1
        if size:
            c, m = _sift_min(h, 0, h[size], size, True)
            cmp += c
            mv += m
    return out, cmp, mv


def topk_buffer(a, K):
    """Bounded insertion buffer: sorted array of at most K records, best first."""
    n = len(a)
    K = min(K, n)
    if K <= 0:
        return [], 0, 0
    buf = [0] * K
    cnt = 0
    cmp = mv = 0
    for x in a:
        if cnt == K:
            cmp += 1
            if not (x < buf[K - 1]):
                continue
            j = K - 1                   # last one falls off the end
        else:
            j = cnt
            cnt += 1
        while j > 0:
            cmp += 1
            if x < buf[j - 1]:
                buf[j] = buf[j - 1]
                mv += 1
                j -= 1
            else:
                break
        buf[j] = x
        mv += 1
    return buf, cmp, mv


# ---------------------------------------------------------------- live feed helpers
class LiveHeap:
    """A size-K min-heap (negated keys) that stays alive between windows."""

    def __init__(self, K):
        self.K = K
        self.h = []

    def feed(self, items):
        h, K = self.h, self.K
        cmp = mv = 0
        for x in items:
            nk = -x
            if len(h) < K:
                i = len(h)
                h.append(nk)
                mv += 1
                while i > 0:                      # sift up
                    par = (i - 1) // 2
                    cmp += 1
                    if nk < h[par]:
                        h[i] = h[par]
                        mv += 1
                        i = par
                    else:
                        break
                if i != len(h) - 1:
                    h[i] = nk
                    mv += 1
            else:
                cmp += 1
                if h[0] < nk:
                    c, m = _sift_min(h, 0, nk, K, True)
                    cmp += c
                    mv += m
        return cmp, mv

    def publish(self):
        """Ranked top-K without wrecking the live heap (works on a copy)."""
        tmp = self.h[:]
        out, cmp, mv = _drain(tmp)
        return out, cmp, mv + len(tmp)


def merge_top(cur, new, K):
    """Merge two sorted lists but stop once K records are out."""
    out = []
    i = j = 0
    cmp = 0
    ni, nj = len(cur), len(new)
    while len(out) < K and i < ni and j < nj:
        cmp += 1
        if new[j] < cur[i]:
            out.append(new[j])
            j += 1
        else:
            out.append(cur[i])
            i += 1
    if len(out) < K:
        rest = cur[i:] if i < ni else new[j:]
        out.extend(rest[:K - len(out)])
    return out, cmp, len(out)


# ---------------------------------------------------------------- registry
SORTS = {
    "Insertion Sort": insertion_sort,
    "Merge Sort": merge_sort,
    "Naive Quick Sort": quick_naive,
    "Hardened Quick Sort": quick_hardened,
    "Heap Sort": heap_sort,
}

# ============================ live-feed strategies ============================
import time as _t
time = _t
def window_slices(c, wsec):
    t = c["time"]
    nw = int(t.max() // wsec) + 1
    edges = np.searchsorted(t, np.arange(nw + 1) * wsec, side="left")
    edges[-1] = len(t)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(nw)]


def live_pass(strategy, keys, slices, K, report=None):
    """One full pass over all windows. Returns list of cumulative seconds after each window."""
    cum, total = [], 0.0
    cmp = mv = 0
    if strategy == "a":
        seen = []
    elif strategy == "b":
        lh = LiveHeap(K)
    else:
        cur = []
    top = None
    for w, (lo, hi) in enumerate(slices):
        new = keys[lo:hi]
        t0 = time.perf_counter()
        if strategy == "a":
            seen.extend(new)
            tmp = seen[:]
            c1, m1 = merge_sort(tmp)
            top = tmp[:K]
            cmp += c1
            mv += m1
        elif strategy == "b":
            c1, m1 = lh.feed(new)
            top, c2, m2 = lh.publish()
            cmp += c1 + c2
            mv += m1 + m2
        else:
            tmp = new[:]
            c1, m1 = merge_sort(tmp)
            top, c2, m2 = merge_top(cur, tmp, K)
            cur = top
            cmp += c1 + c2
            mv += m1 + m2
        total += time.perf_counter() - t0
        cum.append(total)
        if report:
            report(w, total)
    return cum, top, cmp, mv




# ============================ embedded benchmark replay (results.csv, zlib+base64) ============================
_RESULTS_B64 = (
    "eNqlfVuPZLmN5vsC+x/qfcIJ3S8I7IvdY8zAmFkMvO9GuTvbU3C7yqju9uzOr9+PpOKWRSoigwV0nxORIeociaR4+UT9/euXP//0+rfDx5/+8u"
    "Xw/ZfPPxw+H/5w+K9Pn3/48l+Hr69///nwt9cfPn38/Kef8ZOf/vT1188/4/bnXz7+8uvP+P3f/v7x66efv3zGz7784/Xnwz9ev3768dMrqHz5"
    "5fV//o/fHf7188+vX3/59OXzhz9++frL4XfxEEMIh8OhHsJLqC3HSTcjpyMuZdajfM2XLp/yPPzvPxxSrWlMufTD//n66+uBOvi3169/eVWJhz"
    "RCWjfHq0uOx5svifhojRrP3i6U//3jp3+8fviPXz99/1edfpylCP1wlM/qhejHkHLA/3Mvlw7+5ePXH14/v/6w66O0Ju9Q+nFz4T5GTfWQ6N9V"
    "H68f/25QxsDKjYxHSXKRdzl9GZhyG32idclXw/N2ZtMN+Z5z7XJTiBLGly+50aUP7mzWxDNbMClNLlGf2VviAbSb3LSjcunrsmZ23p/ZN/RTC3"
    "PdHM1LXjPbWz7EHGu6M7Nv+qgpRrlhLsdsyCXdXriPPvESKc0b7rnM7BvKJfU1s0LydOk3n/Ka2cIz24M9s/mWfCAx5BuhZFyI/Jz4pT6j+e2M"
    "ZnnmvF5cpCfN60fPlYiWmdP9Gb2hH2MKZfANk6LPcqHeMIVdLknozxoCN59z3JnUN69RehPGLONmzG8+1TX0BTcpjDbUSX1DuZXWhV0WEWHCLs"
    "Ml8hRaFXapA63xjt2e1HIzPj2WmuSGuBGjNPgySHhTavRlCrFcjQ/+w50+ueWNOPUmHJPD8Vt9fKOIayjl/uTe0p+tp7lueB6Fa+RTmKnIpd8+"
    "/P25ffMWNcc1A/lGYMvNp7wENpIqjrOpc/uGcol1Kfl0o4pvFXNkynVikifUjT219c0ylUJbN8fNhajniRUTynXo01rfrq/rsZeUniZyLmG9nt"
    "Ye2/1pvaUfR5QlJA4eW3yWS7q5yLBAGw+MKkY8hDvTWr/Rw8KcdagK+GZaE+vhmHU9XN8qgy4avlVFC5xVg1BueAqIbE72vLYb8qUXMT9ww2aS"
    "KIbSeZSK9FnGWkBGxWqDS6mGPm5vrYNebqyDm0sWZs9MfOSQ78/tLf0YSR7l5mhekjx8iwVsWUa4Zzu9eQfoJ3mHIONvXJjxExZAqDtdXN/QhR"
    "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"
)

PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>The Midnight Queue - Masha &amp; the Bear</title>
<style>
:root{--pink:#F06292;--pinkl:#FFD6E3;--pinkd:#C2185B;--brown:#8B5A2B;--browndk:#4A2E14;--brownl:#EBD3B4;--honey:#F5B83D;--green:#2E7D4F;--greenl:#D8EFD9;--cream:#FFF6E5;--paper:#FFFBF2}
*{box-sizing:border-box}
body{margin:0;font-family:"Trebuchet MS","Segoe UI",system-ui,sans-serif;color:var(--browndk);line-height:1.5;
  background:radial-gradient(circle,rgba(240,98,146,.10) 3px,transparent 4px) 0 0/34px 34px,var(--cream)}
header.hero{position:relative;overflow:hidden;padding:20px 28px 58px;color:#fff;border-bottom:5px solid #fff;
  background:radial-gradient(circle,rgba(255,255,255,.30) 4px,transparent 5px) 0 0/26px 26px,linear-gradient(135deg,#F06292 0%,#FF8FAB 55%,#F5B83D 135%);
  box-shadow:0 8px 24px rgba(139,90,43,.3)}
.hero-in{max-width:1180px;margin:0 auto;display:flex;flex-wrap:wrap;align-items:center;gap:22px}
.hero h1{margin:2px 0;font-size:40px;line-height:1.1;font-weight:800;text-shadow:2px 3px 0 rgba(139,40,80,.45)}
.kick{font-size:12.5px;letter-spacing:.14em;text-transform:uppercase;font-weight:700}
.sub{font-size:16px;margin:4px 0 8px}
.chip{display:inline-block;margin:3px 6px 3px 0;padding:2px 12px;border-radius:999px;background:rgba(255,255,255,.93);color:#8B2252;font-weight:700;font-size:12.5px}
.trees{position:absolute;left:0;right:0;bottom:0;padding:6px 18px 2px;background:linear-gradient(#3E8E5A,#2E7D4F);font-size:24px;letter-spacing:6px;white-space:nowrap;overflow:hidden}
.bounce{animation:bob 2.6s ease-in-out infinite}.bounce.b2{animation-delay:.9s}
@keyframes bob{0%,100%{transform:translateY(0) rotate(-2deg)}50%{transform:translateY(-7px) rotate(2deg)}}
nav{position:sticky;top:0;z-index:20;display:flex;flex-wrap:wrap;gap:8px;justify-content:center;padding:10px 12px;background:rgba(255,246,229,.95);backdrop-filter:blur(4px);border-bottom:3px dashed var(--brownl)}
nav a{cursor:pointer;text-decoration:none;padding:7px 16px;border-radius:999px;font-weight:800;font-size:14.5px;color:var(--browndk);background:var(--paper);border:2px solid var(--brownl);transition:.15s}
nav a:hover{transform:translateY(-2px);border-color:var(--pink)}
nav a.on{background:var(--pink);border-color:var(--pinkd);color:#fff;box-shadow:0 3px 0 var(--pinkd)}
main{max-width:1180px;margin:0 auto;padding:12px 20px 110px}
.sec{display:flex;align-items:center;gap:16px;border-radius:20px;padding:12px 22px;margin:12px 0 8px;border:3px solid}
.sec.masha{background:linear-gradient(90deg,#FFD6E3,#FFF0F5);border-color:var(--pink)}
.sec.bear{background:linear-gradient(90deg,#EBD3B4,#FBF1E3);border-color:var(--brown)}
.sec h2{margin:0;font-size:25px}.sec p{margin:2px 0 0;color:#6B4423;font-size:14.5px}
.tag{display:inline-block;padding:1px 12px;margin-bottom:3px;border-radius:999px;font-size:11.5px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#fff;background:var(--pink)}
.sec.bear .tag{background:var(--brown)}
.say{display:flex;align-items:flex-start;gap:12px;margin:10px 0}
.bub{max-width:960px;padding:10px 16px;border-radius:18px;font-size:15px;border:2px solid;animation:pop .35s ease}
@keyframes pop{from{transform:scale(.92);opacity:0}to{transform:scale(1);opacity:1}}
.bub.masha{background:#FFE4EE;border-color:var(--pink);color:#5A1331}.bub.bear{background:#F3E2CB;border-color:var(--brown);color:#3E2410}
.bub .who{display:block;font-size:11px;letter-spacing:.1em;text-transform:uppercase;font-weight:800}
.card{background:var(--paper);border:2px solid #E8D3B0;border-radius:16px;padding:14px 18px;margin:10px 0}
.card h3{margin:0 0 8px;color:var(--pinkd)}
.note{border-left:7px solid;border-radius:10px;padding:8px 14px;margin:8px 0;font-size:14.5px}
.note.info{background:#DFF0FA;border-color:#4A9AC9;color:#173A52}.note.ok{background:var(--greenl);border-color:var(--green);color:#173D26}
.note.warn{background:#FFF1CC;border-color:var(--honey);color:#5A3F00}.note.bad{background:#FFD9D2;border-color:#D9534F;color:#5C1710}
.tiles{display:flex;flex-wrap:wrap;gap:12px;margin:10px 0}
.tile{flex:1 1 150px;border-radius:16px;padding:10px 14px;color:#fff;box-shadow:0 3px 8px rgba(74,46,20,.2)}
.tile b{display:block;font-size:24px;line-height:1.25}.tile span{font-size:11.5px;text-transform:uppercase;letter-spacing:.07em;font-weight:700}
.t0{background:var(--pink)}.t1{background:var(--brown)}.t2{background:var(--green)}.t3{background:var(--honey);color:var(--browndk)}
.row{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-end;margin:8px 0}
.row label{display:flex;flex-direction:column;font-weight:800;font-size:13px;gap:3px}
select,input[type=number]{padding:7px 10px;border-radius:10px;border:2px solid var(--brown);background:var(--paper);color:var(--browndk);font-size:14.5px;font-family:inherit}
input[type=range]{accent-color:var(--pink);width:190px}
.chk{display:inline-flex;align-items:center;gap:5px;margin:2px 10px 2px 0;font-weight:700;font-size:14px}
.chk input{accent-color:var(--pink);width:17px;height:17px}
.btn{cursor:pointer;border:0;border-radius:999px;padding:10px 24px;font-weight:800;font-size:15px;color:#fff;background:var(--pink);box-shadow:0 4px 0 var(--pinkd);font-family:inherit;transition:.12s}
.btn:hover{transform:translateY(-2px)}.btn:active{transform:translateY(3px);box-shadow:0 1px 0 var(--pinkd)}
.btn.bear{background:var(--brown);box-shadow:0 4px 0 #5D3A1A}.btn.green{background:var(--green);box-shadow:0 4px 0 #1d5a38}
.btn[disabled]{opacity:.5;cursor:wait}
table.tbl{border-collapse:separate;border-spacing:0;margin:8px 0;font-size:13.5px;background:var(--paper);border-radius:12px;overflow:hidden;border:2px solid #E8D3B0;width:100%}
.tbl th{background:var(--brown);color:var(--cream);padding:6px 10px;font-weight:700;position:sticky;top:0}
.tbl td{padding:5px 10px;text-align:center;border-top:1px solid #EFDDBF}
.tbl tr:nth-child(even) td{background:#FFF3DC}.tbl td:first-child{text-align:left;font-weight:700}
.tbl td.win{background:#FFE29A!important}.tbl td.dnf{background:#FFD9D2!important;color:#8a1d12;font-weight:700}
.scroll{max-height:420px;overflow:auto;border-radius:12px}
.bar{height:12px;border-radius:6px;background:var(--pink)}
.cast{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:10px 0}
.cc{cursor:pointer;border-radius:16px;padding:10px 14px;background:var(--paper);border:2px solid #E8D3B0;font-size:13.5px;transition:.15s}
.cc:hover{transform:translateY(-4px) rotate(-1deg);border-color:var(--pink);box-shadow:0 6px 14px rgba(240,98,146,.25)}
.cc h4{margin:0 0 4px;font-size:16px;color:var(--pinkd)}.cc small{display:block;color:#7A5A3A;margin-top:4px}
.track{position:relative;height:40px;border-radius:20px;background:var(--greenl);border:3px solid var(--green);margin:6px 0;overflow:hidden}
.fill{position:absolute;left:0;top:0;bottom:0;width:0;background:repeating-linear-gradient(45deg,#F5B83D,#F5B83D 10px,#FFD36B 10px,#FFD36B 20px);transition:width .2s}
.run{position:absolute;top:2px;font-size:26px;line-height:30px;transition:left .2s}.home{position:absolute;right:8px;top:2px;font-size:26px;line-height:30px}
.trk-txt{font-size:13px;margin:4px 0}
.vizgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:14px}
.viz{background:var(--paper);border:3px solid #E8D3B0;border-radius:18px;padding:10px 14px}
.viz h3{margin:0 0 6px;font-size:17px;color:var(--pinkd)}
.bars{display:flex;align-items:flex-end;gap:2px;height:230px;padding:4px;border-radius:12px;background:linear-gradient(#FFF6E5,#FBE9C9)}
.bars div{flex:1;border-radius:4px 4px 0 0;background:linear-gradient(#FF8FAB,#F06292);transition:height .06s}
.bars div.c{background:linear-gradient(#8b2252,#C2185B)}.bars div.m{background:linear-gradient(#FFD36B,#F5B83D)}.bars div.d{background:linear-gradient(#6fc28b,#2E7D4F)}
.stat{display:flex;gap:16px;font-weight:700;font-size:13.5px;margin-top:6px;flex-wrap:wrap}
.chartbox{background:var(--paper);border:2px solid #E8D3B0;border-radius:16px;padding:8px;margin:8px 0;overflow-x:auto}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12.5px;font-weight:700;margin:2px 8px 6px}
.legend i{display:inline-block;width:14px;height:4px;border-radius:2px;margin-right:5px;vertical-align:middle}
.chip2{display:inline-block;padding:0 9px;border-radius:999px;color:#fff;font-weight:800;font-size:12px}
#mascot{position:fixed;right:14px;bottom:12px;z-index:30;display:flex;align-items:flex-end;gap:8px;max-width:360px;pointer-events:none}
#mascot .bub{pointer-events:auto;font-size:13.5px;padding:8px 12px;box-shadow:0 4px 12px rgba(74,46,20,.25)}
#mascot .av{pointer-events:auto;cursor:pointer}
footer{text-align:center;padding:20px;font-size:13px;color:#7A5A3A}
@media(max-width:640px){.hero h1{font-size:28px}.vizgrid{grid-template-columns:1fr}}
</style></head>
<body>
<header class="hero"><div class="hero-in">
  <span id="hAv1" class="bounce"></span><span id="hAv2" class="bounce b2"></span>
  <div><div class="kick">DAA &middot; Assignment 2 &middot; 24i-3037 &middot; Section C</div>
  <h1>The Midnight Queue</h1>
  <div class="sub">Ranking suspicious transactions at scale &mdash; play with Masha &amp; the Bear &#127855;</div>
  <span class="chip">K = 25 + (243037 mod 226) = 112</span><span class="chip">seed 243037</span><span class="chip">284,807 transactions</span><span class="chip">hand-written sorts</span></div>
</div><div class="trees">&#127794;&#127795;&#127794;&#127794;&#127812;&#127794;&#127795;&#127794;&#127794;&#127812;&#127794;&#127795;&#127794;&#127794;&#127812;&#127794;&#127795;&#127794;&#127794;&#127812;&#127794;&#127795;&#127794;&#127794;&#127812;&#127794;&#127795;&#127794;&#127794;&#127812;&#127794;</div></header>
<nav id="tabs"></nav>
<main id="view"></main>
<div id="mascot"></div>
<footer>Made with &#127855; for DAA Assignment 2 &mdash; every sort here is the student's own code; the server only does the timing.</footer>

<script>
const $=(s,r=document)=>r.querySelector(s), $$=(s,r=document)=>[...r.querySelectorAll(s)];
const fmtT=s=>s==null||isNaN(s)?'-':(s<10?(s*1000).toFixed(1)+' ms':s.toFixed(1)+' s');
const fmtN=x=>Number(x).toLocaleString();
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function api(path,params={}){
  try{const r=await fetch('/api/'+path+'?'+new URLSearchParams(params));return await r.json();}
  catch(e){return {error:'Could not reach the server: '+e};}
}

/* ---------- characters ---------- */
const AV={
 masha:s=>`<svg width="${s}" height="${s}" viewBox="0 0 100 100" style="flex:none;display:block"><circle cx="50" cy="56" r="36" fill="#FFDDBE" stroke="#D9A074" stroke-width="2"/><path d="M12 58 C2 70 8 86 19 88 C14 77 15 66 19 58Z" fill="#E9A23B"/><path d="M88 58 C98 70 92 86 81 88 C86 77 85 66 81 58Z" fill="#E9A23B"/><path d="M10 56 C10 12 90 12 90 56 C78 40 22 40 10 56Z" fill="#F06292" stroke="#C2185B" stroke-width="2"/><g fill="#fff"><circle cx="25" cy="38" r="3.2"/><circle cx="41" cy="27" r="3.2"/><circle cx="60" cy="26" r="3.2"/><circle cx="76" cy="38" r="3.2"/><circle cx="50" cy="40" r="2.6"/></g><ellipse cx="37" cy="62" rx="6.5" ry="7.5" fill="#fff" stroke="#8a5a3a"/><ellipse cx="63" cy="62" rx="6.5" ry="7.5" fill="#fff" stroke="#8a5a3a"/><circle cx="38" cy="63" r="3.6" fill="#2E8B57"/><circle cx="62" cy="63" r="3.6" fill="#2E8B57"/><circle cx="38" cy="63" r="1.8" fill="#10301f"/><circle cx="62" cy="63" r="1.8" fill="#10301f"/><circle cx="39" cy="61.5" r="1.1" fill="#fff"/><circle cx="63" cy="61.5" r="1.1" fill="#fff"/><circle cx="26" cy="75" r="5" fill="#FF8FAB" opacity=".75"/><circle cx="74" cy="75" r="5" fill="#FF8FAB" opacity=".75"/><path d="M39 77 Q50 91 61 77Z" fill="#B71C4C"/><path d="M44 83 Q50 87 56 83Z" fill="#FF8FAB"/></svg>`,
 bear:s=>`<svg width="${s}" height="${s}" viewBox="0 0 100 100" style="flex:none;display:block"><circle cx="20" cy="26" r="13" fill="#7A4B22"/><circle cx="20" cy="26" r="7" fill="#C9915A"/><circle cx="80" cy="26" r="13" fill="#7A4B22"/><circle cx="80" cy="26" r="7" fill="#C9915A"/><circle cx="50" cy="54" r="38" fill="#8B5A2B" stroke="#5D3A1A" stroke-width="2"/><ellipse cx="50" cy="69" rx="19" ry="15" fill="#E8C08A"/><ellipse cx="50" cy="60" rx="7" ry="5" fill="#2B1608"/><circle cx="35" cy="46" r="4.6" fill="#2B1608"/><circle cx="65" cy="46" r="4.6" fill="#2B1608"/><circle cx="36.5" cy="44.5" r="1.5" fill="#fff"/><circle cx="66.5" cy="44.5" r="1.5" fill="#fff"/><path d="M27 36 L42 40 M73 36 L58 40" stroke="#2B1608" stroke-width="3" stroke-linecap="round"/><path d="M50 65 L50 71 M42 75 Q50 81 58 75" stroke="#2B1608" stroke-width="2.4" fill="none" stroke-linecap="round"/></svg>`};
$('#hAv1').innerHTML=AV.masha(120);$('#hAv2').innerHTML=AV.bear(120);

const say=(w,t)=>`<div class="say">${AV[w](50)}<div class="bub ${w}"><span class="who">${w=='masha'?'Masha':'Bear'}</span>${t}</div></div>`;
const section=(tag,title,sub,w='masha')=>`<div class="sec ${w}">${AV[w](56)}<div><span class="tag">${tag}</span><h2>${title}</h2>${sub?`<p>${sub}</p>`:''}</div></div>`;
const note=(t,k='info')=>`<div class="note ${k}">${t}</div>`;
const tiles=a=>`<div class="tiles">${a.map(([l,v],i)=>`<div class="tile t${i%4}"><span>${l}</span><b>${v}</b></div>`).join('')}</div>`;
function chat(w,t){$('#mascot').innerHTML=`<div class="bub ${w}"><span class="who">${w=='masha'?'Masha':'Bear'}</span>${t}</div><div class="av bounce" id="mav">${AV[w](64)}</div>`;
  $('#mav').onclick=()=>{const f=FACTS[Math.floor(Math.random()*FACTS.length)];chat(f[0],f[1]);};}
const FACTS=[['masha','Did you know? Insertion Sort on already-sorted data needs only n-1 comparisons!'],['bear','Quick Sort with the last element as pivot is a disaster on sorted data. Hrrm.'],
 ['masha','Merge Sort never has a bad day: always n log n!'],['bear','A size-K heap throws away most records after ONE comparison.'],
 ['masha','The three-way partition makes Hardened Quick Sort fly when many keys are equal.'],['bear','Frauds hide in the top risk bands. Only 0.3% of rows are in band 9.']];
function track(el){
  el.innerHTML=`<div class="trk-txt"></div><div class="track"><div class="fill"></div><div class="run">&#127939;&#8205;&#9792;&#65039;</div><div class="home">&#127969;&#128059;</div></div>`;
  return (f,t)=>{f=Math.min(1,Math.max(0,f));$('.fill',el).style.width=`calc(${f*100}% - 4px)`;$('.run',el).style.left=`calc(${f*100}% - ${f*34}px)`;$('.trk-txt',el).innerHTML=t||'';};
}

/* ---------- charts (log/linear SVG) ---------- */
function tickFmt(v){if(v>=1e6)return (v/1e6)+'M';if(v>=1e3)return (v/1e3)+'k';if(v<1&&v>0)return String(+v.toPrecision(2));return String(v);}
function lineChart(series,o={}){
  const W=o.w||680,H=o.h||330,m={l:62,r:16,t:14,b:46};
  const ok=p=>(!o.logx||p[0]>0)&&(!o.logy||p[1]>0);
  const all=series.flatMap(s=>s.pts.filter(ok).concat((s.xs||[]).filter(ok))).concat((o.hlines||[]).map(h=>[series[0].pts.length?series[0].pts[0][0]:1,h.y]).filter(ok)); if(!all.length)return '<div class="note warn">no data</div>';
  const fx=o.logx?Math.log10:x=>x, fy=o.logy?Math.log10:y=>y;
  let x0=Math.min(...all.map(p=>fx(p[0]))),x1=Math.max(...all.map(p=>fx(p[0]))),y0=Math.min(...all.map(p=>fy(p[1]))),y1=Math.max(...all.map(p=>fy(p[1])));
  if(o.logx){x0=Math.floor(x0);x1=Math.ceil(x1);} if(o.logy){y0=Math.floor(y0);y1=Math.ceil(y1);} else {y0=Math.min(y0,0);}
  if(x1==x0)x1=x0+1; if(y1==y0)y1=y0+1;
  const sx=x=>m.l+(fx(x)-x0)/(x1-x0)*(W-m.l-m.r), sy=y=>H-m.b-(fy(y)-y0)/(y1-y0)*(H-m.t-m.b);
  let g='';
  const yt=[];if(o.logy){for(let k=y0;k<=y1;k++)yt.push(Math.pow(10,k));}else{for(let i=0;i<=4;i++)yt.push(y0+(y1-y0)*i/4);}
  yt.forEach(v=>{g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${sy(v)}" y2="${sy(v)}" stroke="#E6D3B3" stroke-dasharray="3 3"/><text x="${m.l-6}" y="${sy(v)+4}" font-size="11" text-anchor="end" fill="#6B4423">${tickFmt(+v.toPrecision(3))}</text>`;});
  const xt=[];if(o.logx){for(let k=x0;k<=x1;k++)xt.push(Math.pow(10,k));}else{for(let i=0;i<=5;i++)xt.push(x0+(x1-x0)*i/5);}
  xt.forEach(v=>{g+=`<line y1="${m.t}" y2="${H-m.b}" x1="${sx(v)}" x2="${sx(v)}" stroke="#EFE0C4" stroke-dasharray="3 3"/><text x="${sx(v)}" y="${H-m.b+16}" font-size="11" text-anchor="middle" fill="#6B4423">${tickFmt(Math.round(v))}</text>`;});
  (o.hlines||[]).forEach(h=>{g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${sy(h.y)}" y2="${sy(h.y)}" stroke="#999" stroke-dasharray="6 4"/><text x="${W-m.r-4}" y="${sy(h.y)-5}" font-size="11" text-anchor="end" fill="#777">${h.t}</text>`;});
  series.forEach(s=>{
    const pts=s.pts.filter(ok); if(!pts.length)return;
    if(!s.nolines)g+=`<polyline fill="none" stroke="${s.color}" stroke-width="2.4" ${s.dash?'stroke-dasharray="6 5"':''} points="${pts.map(p=>sx(p[0])+','+sy(p[1])).join(' ')}"/>`;
    if(s.markers!==false&&pts.length<60)pts.forEach(p=>{g+=`<circle cx="${sx(p[0])}" cy="${sy(p[1])}" r="4" fill="${p[2]=='dnf'?'#fff':s.color}" stroke="${s.color}" stroke-width="2"><title>${s.name}: x=${fmtN(Math.round(p[0]*100)/100)}, y=${p[1]<10?p[1].toFixed(4):p[1].toFixed(2)}</title></circle>`;});
    (s.xs||[]).forEach(p=>{g+=`<text x="${sx(p[0])}" y="${sy(p[1])+5}" font-size="16" text-anchor="middle" fill="${s.color}" font-weight="800">&#10006;</text>`;});
  });
  g+=`<text x="${(W+m.l)/2}" y="${H-6}" font-size="12" text-anchor="middle" fill="#4A2E14" font-weight="700">${o.xl||''}</text><text transform="translate(14 ${H/2}) rotate(-90)" font-size="12" text-anchor="middle" fill="#4A2E14" font-weight="700">${o.yl||''}</text>`;
  return `<div class="chartbox"><svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px">${g}</svg><div class="legend">${series.map(s=>`<span><i style="background:${s.color}"></i>${s.name}</span>`).join('')}</div></div>`;
}
const COL={'Insertion Sort':'#D55E00','Merge Sort':'#0072B2','Naive Quick Sort':'#009E73','Hardened Quick Sort':'#CC79A7','Heap Sort':'#E69F00'};
const SORTS=Object.keys(COL);
const CONDS={C1:'Arrival',C2:'Shuffled',C3:'Pre-ranked',C4:'Reverse-ranked',C5:'Nearly ranked',C6:'Band-only'};
const SLOW=['Insertion Sort','Naive Quick Sort'];
const BANDCOL=['#9aa5b1','#9aa5b1','#4A9AC9','#2E7D4F','#7ab648','#F5B83D','#F28C28','#E5532D','#C2185B','#7B1FA2'];

/* ---------- sorting animation recorders ---------- */
function recordSort(algo,arr){
  const a=arr.slice(),n=a.length,fr=[];let cmp=0,mv=0;
  fr.push({a:a.slice(),c:[],m:[],cmp:0,mv:0});
  const C=(...i)=>{cmp++;fr.push({a:a.slice(),c:i,m:[],cmp,mv});};
  const M=(k,...i)=>{mv+=k;fr.push({a:a.slice(),c:[],m:i,cmp,mv});};
  const sw=(x,y)=>{const t=a[x];a[x]=a[y];a[y]=t;M(2,x,y);};
  if(algo=='Insertion Sort'){
    for(let i=1;i<n;i++){const x=a[i];let j=i-1;
      while(j>=0){C(j,i);if(x<a[j]){a[j+1]=a[j];M(1,j+1);j--;}else break;}
      if(j+1!==i){a[j+1]=x;M(1,j+1);}}
  }else if(algo=='Merge Sort'){
    const ms=(lo,hi)=>{if(hi-lo<2)return;const mid=(lo+hi)>>1;ms(lo,mid);ms(mid,hi);
      const L=a.slice(lo,mid),R=a.slice(mid,hi);let i=0,j=0,k=lo;
      while(i<L.length&&j<R.length){C(lo+i,mid+j);if(R[j]<L[i])a[k]=R[j++];else a[k]=L[i++];M(1,k);k++;}
      while(i<L.length){a[k]=L[i++];M(1,k++);}while(j<R.length){a[k]=R[j++];M(1,k++);}};
    ms(0,n);
  }else if(algo=='Naive Quick Sort'){
    const qs=(lo,hi)=>{if(lo>=hi)return;const p=a[hi];let i=lo;
      for(let j=lo;j<hi;j++){C(j,hi);if(a[j]<p){sw(i,j);i++;}}
      sw(i,hi);qs(lo,i-1);qs(i+1,hi);};
    qs(0,n-1);
  }else if(algo=='Hardened Quick Sort'){
    const qh=(lo,hi)=>{while(lo<hi){const p=a[lo+Math.floor(Math.random()*(hi-lo+1))];let lt=lo,i=lo,gt=hi;
      while(i<=gt){const x=a[i];C(i,lt);if(x<p){sw(i,lt);lt++;i++;}else{C(i,gt);if(p<x){sw(i,gt);gt--;}else i++;}}
      if(lt-lo<hi-gt){qh(lo,lt-1);lo=gt+1;}else{qh(gt+1,hi);hi=lt-1;}}};
    qh(0,n-1);
  }else{
    const sift=(i,end)=>{while(true){let c=2*i+1;if(c>=end)break;if(c+1<end){C(c,c+1);if(a[c]<a[c+1])c++;}C(i,c);if(a[i]<a[c]){sw(i,c);i=c;}else break;}};
    for(let s=(n>>1)-1;s>=0;s--)sift(s,n);
    for(let e=n-1;e>0;e--){sw(0,e);sift(0,e);}
  }
  fr.push({a:a.slice(),c:[],m:[],cmp,mv,done:true});
  return fr;
}
function makeInput(kind,n){
  let a=Array.from({length:n},(_,i)=>i+1);
  if(kind=='random'){for(let i=n-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[a[i],a[j]]=[a[j],a[i]];}}
  else if(kind=='reversed')a.reverse();
  else if(kind=='nearly'){for(let k=0;k<Math.max(1,Math.round(n*.08));k++){const i=Math.floor(Math.random()*n),j=Math.floor(Math.random()*n);[a[i],a[j]]=[a[j],a[i]];}}
  else if(kind=='fewunique'){a=a.map(v=>Math.ceil(v/(n/4)));for(let i=n-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[a[i],a[j]]=[a[j],a[i]];}}
  return a;
}
const SHOWTXT={
 'Insertion Sort':['bear','I pick up one jar at a time and slide it left into the sorted shelf. Great when the shelf is almost sorted, painful when it is reversed.'],
 'Merge Sort':['masha','I split the jars in halves, tidy each half, then merge two sorted shelves. Always n log n!'],
 'Naive Quick Sort':['masha','I grab the LAST jar as my pivot and throw smaller ones to the left. On sorted shelves I run in circles (n&sup2;)...'],
 'Hardened Quick Sort':['masha','I close my eyes for a random pivot and use a three-way split, so equal jars do not slow me down.'],
 'Heap Sort':['bear','I build a pyramid of honey pots (a max-heap) and keep taking the biggest pot off the top.']};

class Viz{
  constructor(el,algo,arr){this.el=el;this.algo=algo;this.fr=recordSort(algo,arr);this.i=0;this.max=Math.max(...arr);
    el.innerHTML=`<div class="viz"><h3>${algo}</h3><div class="bars"></div><div class="stat"><span>&#128269; comparisons: <b class="vc">0</b></span><span>&#9998; moves: <b class="vm">0</b></span><span>step <b class="vs">0</b>/${this.fr.length-1}</span><span class="vd"></span></div></div>`;
    this.bars=$('.bars',el);this.n=arr.length;this.bars.innerHTML=arr.map(()=>'<div></div>').join('');this.draw();}
  get done(){return this.i>=this.fr.length-1;}
  draw(){const f=this.fr[this.i],ch=this.bars.children;
    for(let k=0;k<this.n;k++){const d=ch[k];d.style.height=(f.a[k]/this.max*100)+'%';d.className=f.done?'d':f.c.includes(k)?'c':f.m.includes(k)?'m':'';}
    $('.vc',this.el).textContent=fmtN(f.cmp);$('.vm',this.el).textContent=fmtN(f.mv);$('.vs',this.el).textContent=this.i;
    if(f.done)$('.vd',this.el).innerHTML='&#9989; sorted!';}
  step(k=1){this.i=Math.min(this.fr.length-1,this.i+k);this.draw();}
  reset(){this.i=0;$('.vd',this.el).innerHTML='';this.draw();}
}

/* ---------- tabs ---------- */
const TABS=[['home','&#127968;','Home'],['data','&#128202;','Data'],['show','&#127916;','Sort Show'],['arena','&#127937;','Arena'],['console','&#127855;','Analyst Console'],['live','&#128236;','Live Feed'],['results','&#128200;','Big Results'],['verdict','&#127942;','Verdict']];
const VIEWS={};let INFO=null,timers=[];
function clearTimers(){timers.forEach(clearInterval);timers=[];}
async function route(){
  clearTimers();const t=(location.hash||'#home').slice(1);const tab=VIEWS[t]?t:'home';
  $('#tabs').innerHTML=TABS.map(([k,e,l])=>`<a class="${k==tab?'on':''}" href="#${k}">${e} ${l}</a>`).join('');
  const v=$('#view');v.innerHTML='';window.scrollTo({top:0});
  if(!INFO){INFO=await api('info');}
  VIEWS[tab](v);
}
window.addEventListener('hashchange',route);

/* HOME */
VIEWS.home=v=>{
  v.innerHTML=section('Welcome','Masha &amp; the Bear sort the midnight mail','pick a room in the house using the tabs above')+
   say('masha','Hi! I am Masha! We have <b>284,807</b> card transactions to rank from <i>most suspicious</i> to <i>least</i>. Bear says we are not allowed to use the built-in <code>sort()</code>, so we wrote every algorithm ourselves!')+
   say('bear','Hrrm. Click a contender below and watch it work. Then race them in the Arena, build a real shift queue, or watch the live feed.')+
   `<div class="cast">`+[['&#127855;','Insertion Sort','Bear files one jar at a time.','best &Theta;(n) &middot; worst &Theta;(n&sup2;) &middot; stable'],['&#129309;','Merge Sort','Merge two tidy shelves.','&Theta;(n log n) &middot; O(n) space &middot; stable'],
    ['&#127939;&#8205;&#9792;&#65039;','Naive Quick Sort','Last jar is the pivot!','avg n log n &middot; worst n&sup2;'],['&#127922;','Hardened Quick Sort','Random pivot + 3-way split.','expected n log n &middot; O(log n) stack'],['&#127802;','Heap Sort','Pyramid of honey pots.','&Theta;(n log n) &middot; O(1) space']].map(([e,n,s,f])=>`<div class="cc" data-a="${n}"><h4>${e} ${n}</h4>${s}<small>${f}</small></div>`).join('')+`</div>`+
   tiles([['Transactions',fmtN(INFO.N)],['Frauds',INFO.frauds],['K (my queue size)',INFO.K],['Half the frauds need K =',fmtN(INFO.k_half)]]);
  $$('.cc',v).forEach(c=>c.onclick=()=>{window.__showAlgo=c.dataset.a;location.hash='#show';});
  chat('masha','Click a card to watch it sort!');
};

/* DATA */
VIEWS.data=v=>{
  const mx=Math.max(...INFO.bands.map(b=>b.count));
  v.innerHTML=section('Problem A','Know your data','look before touching any algorithm','bear')+
   tiles([['Transactions',fmtN(INFO.N)],['Frauds (Class=1)',INFO.frauds],['Distinct Amounts',fmtN(INFO.distinct_amounts)],['Top-10 amounts cover',(INFO.top10_share*100).toFixed(1)+'%']])+
   `<div class="card"><h3>Risk bands</h3><table class="tbl"><tr><th>band</th><th>count</th><th>share</th><th>relative size</th><th>frauds</th><th>fraud rate</th></tr>`+
   INFO.bands.filter(b=>b.count).map(b=>`<tr><td><span class="chip2" style="background:${BANDCOL[b.band]}">band ${b.band}</span></td><td>${fmtN(b.count)}</td><td>${(b.share*100).toFixed(1)}%</td><td style="width:34%"><div class="bar" style="width:${Math.max(2,b.count/mx*100)}%;background:${BANDCOL[b.band]}"></div></td><td>${b.frauds}</td><td>${(b.fraud_rate*100).toFixed(2)}%</td></tr>`).join('')+`</table></div>`+
   say('bear','Only 8 of 10 bands are used. Bands 3 and 4 hold most of the file; band 9 has just a few hundred rows but a huge fraud rate.')+
   `<div class="card"><h3>&#128269; How many frauds does a queue of length K catch?</h3><div class="row"><label>K = <b id="kv">${INFO.K}</b><input type="range" id="ks" min="0" max="1000" value="${Math.round(Math.log10(INFO.K)/Math.log10(INFO.N)*1000)}"></label><button class="btn bear" id="k112">my K = 112</button><button class="btn green" id="khalf">50% of frauds</button></div><div id="kout"></div><div id="kchart"></div></div>`;
  const curve=INFO.curve;
  $('#kchart',v).innerHTML=lineChart([{name:'Frauds caught (triage order)',color:'#C2185B',pts:curve.map(p=>[p[0],p[1]]),markers:false},{name:'Random order (baseline)',color:'#999',dash:true,markers:false,pts:curve.map(p=>[p[0],p[0]/INFO.N*INFO.frauds])}],{logx:true,xl:'K (queue length)',yl:'frauds caught',w:720,h:300});
  let tmo;const upd=async()=>{const K=Math.max(1,Math.round(Math.pow(INFO.N,$('#ks',v).value/1000)));$('#kv',v).textContent=fmtN(K);
    clearTimeout(tmo);tmo=setTimeout(async()=>{const r=await api('caught',{K});$('#kout',v).innerHTML=tiles([['Queue length K',fmtN(K)],['Frauds caught',r.caught+' of '+INFO.frauds],['Share of all frauds',(r.caught/INFO.frauds*100).toFixed(1)+'%'],['Random order would catch',(K/INFO.N*INFO.frauds).toFixed(1)]]);},120);};
  $('#ks',v).oninput=upd;
  $('#k112',v).onclick=()=>{$('#ks',v).value=Math.round(Math.log10(112)/Math.log10(INFO.N)*1000);upd();chat('masha','K = 112 catches almost nothing: Band 9 is ordered by Amount, so the first ranks are the biggest purchases, not the riskiest!');};
  $('#khalf',v).onclick=()=>{$('#ks',v).value=Math.round(Math.log10(INFO.k_half)/Math.log10(INFO.N)*1000);upd();chat('bear','Half of the frauds need K = '+fmtN(INFO.k_half)+'. The scoring rule limits analysts, not the algorithm.');};
  upd();chat('bear','Slide K and see how many frauds the queue catches.');
};

/* SORT SHOW */
VIEWS.show=v=>{
  const first=window.__showAlgo||'Insertion Sort';
  v.innerHTML=section('Problem B','The Sort Show','watch the real algorithms work, step by step - or race two of them','masha')+
   `<div class="card"><div class="row">
    <label>Contender A<select id="sa">${SORTS.map(s=>`<option ${s==first?'selected':''}>${s}</option>`).join('')}</select></label>
    <label>Race against (optional)<select id="sb"><option value="">- nobody -</option>${SORTS.map(s=>`<option ${s=='Merge Sort'&&first!='Merge Sort'?'selected':''}>${s}</option>`).join('')}</select></label>
    <label>Jars (n)<select id="sn">${[16,24,32,48,64].map(x=>`<option ${x==32?'selected':''}>${x}</option>`).join('')}</select></label>
    <label>Shelf order<select id="sk"><option value="random">Shuffled</option><option value="reversed">Reversed</option><option value="nearly">Nearly sorted</option><option value="sorted">Already sorted</option><option value="fewunique">Few distinct values</option></select></label>
    <label>Speed<input type="range" id="sp" min="1" max="10" value="6"></label></div>
    <div class="row"><button class="btn" id="play">&#9654; Play</button><button class="btn bear" id="stp">&#9197; Step</button><button class="btn bear" id="rst">&#8634; Reset</button><button class="btn green" id="new">&#127922; New shelf</button></div></div>
   <div class="vizgrid" id="vg"></div><div id="sw"></div>`;
  let arr,vz=[],playing=false;
  const build=()=>{arr=makeInput($('#sk',v).value,+$('#sn',v).value);setup();};
  const setup=()=>{playing=false;$('#play',v).innerHTML='&#9654; Play';vz=[];$('#vg',v).innerHTML='';
    [$('#sa',v).value,$('#sb',v).value].filter(Boolean).forEach(a=>{const d=document.createElement('div');$('#vg',v).appendChild(d);vz.push(new Viz(d,a,arr));});
    const t=SHOWTXT[$('#sa',v).value];$('#sw',v).innerHTML=say(t[0],t[1]);chat(t[0],t[1]);};
  const tick=()=>{if(!playing)return;const k=Math.max(1,Math.round(Math.pow(1.5,+$('#sp',v).value-4)));vz.forEach(z=>z.step(k));
    if(vz.every(z=>z.done)){playing=false;$('#play',v).innerHTML='&#9654; Play again';report();}};
  const report=()=>{const r=vz.map(z=>`<b>${z.algo}</b>: ${fmtN(z.fr[z.fr.length-1].cmp)} comparisons, ${fmtN(z.fr[z.fr.length-1].mv)} moves`);
    let msg=r.join('<br>');if(vz.length==2){const a=vz[0].fr.length,b=vz[1].fr.length;msg+=`<br>${a==b?'A tie!':'Fewer steps: <b>'+(a<b?vz[0].algo:vz[1].algo)+'</b>'}`;}
    $('#sw',v).innerHTML=say('bear','All sorted! '+msg);chat('masha','Try the same race on a reversed or already-sorted shelf!');};
  timers.push(setInterval(tick,45));
  $('#play',v).onclick=()=>{if(vz.every(z=>z.done))vz.forEach(z=>z.reset());playing=!playing;$('#play',v).innerHTML=playing?'&#9208; Pause':'&#9654; Play';};
  $('#stp',v).onclick=()=>{playing=false;$('#play',v).innerHTML='&#9654; Play';vz.forEach(z=>z.step(1));};
  $('#rst',v).onclick=()=>{playing=false;$('#play',v).innerHTML='&#9654; Play';vz.forEach(z=>z.reset());};
  $('#new',v).onclick=build;['sa','sb','sn','sk'].forEach(id=>$('#'+id,v).onchange=build);
  window.__showAlgo=null;build();
};

/* ARENA */
VIEWS.arena=v=>{
  v.innerHTML=section('Problem C','The Arena','the real tournament, live: your n, your contenders, your input shapes (timed by the server)','bear')+
   say('masha','Pick the contenders and input shapes, press <b>Start</b>, and I will run to the Bear\'s house while the server times every cell. Each result is checked against Python\'s <code>sorted()</code>.')+
   `<div class="card"><div class="row"><label>Records (n)<select id="an">${[1000,3000,5000,10000].map(x=>`<option ${x==3000?'selected':''}>${x}</option>`).join('')}</select></label></div>
    <div><b>Contenders: </b>${SORTS.map(s=>`<label class="chk"><input type="checkbox" class="aa" value="${s}" checked>${s}</label>`).join('')}</div>
    <div><b>Input shapes: </b>${Object.entries(CONDS).map(([k,l])=>`<label class="chk"><input type="checkbox" class="ac" value="${k}" checked>${k} ${l}</label>`).join('')}</div>
    <div class="row"><button class="btn" id="go">&#127937; Start the race!</button></div>
    <div id="an-note"></div></div><div id="atrk"></div><div id="aboard"></div><div id="asay"></div>`;
  const upd=()=>{$('#an-note',v).innerHTML=+$('#an',v).value>=10000?note('At n = 10,000 Insertion and Naive Quick Sort can take a minute or more on the quadratic shapes (C3/C4/C6). Be patient &mdash; or untick them.','warn'):'';};
  $('#an',v).onchange=upd;upd();
  $('#go',v).onclick=async()=>{
    const btn=$('#go',v);btn.disabled=true;
    const n=+$('#an',v).value,algos=$$('.aa',v).filter(x=>x.checked).map(x=>x.value),conds=$$('.ac',v).filter(x=>x.checked).map(x=>x.value);
    if(!algos.length||!conds.length){btn.disabled=false;return;}
    const tk=track($('#atrk',v));const res={};const total=algos.length*conds.length;let done=0;
    const draw=()=>{
      const win={};conds.forEach(c=>{let best=null;algos.forEach(a=>{const r=res[a+'|'+c];if(r&&r.seconds!=null&&(best==null||r.seconds<res[best+'|'+c].seconds))best=a;});win[c]=best;});
      $('#aboard',v).innerHTML=`<table class="tbl"><tr><th>n = ${fmtN(n)}</th>${conds.map(c=>`<th>${c}<br><small>${CONDS[c]}</small></th>`).join('')}</tr>`+
       algos.map(a=>`<tr><td>${a}</td>${conds.map(c=>{const r=res[a+'|'+c];if(!r)return '<td>&middot;</td>';if(r.error)return '<td class="dnf">skipped</td>';
         return `<td class="${win[c]==a?'win':''}">${win[c]==a?'&#127942; ':''}${fmtT(r.seconds)}${r.verified?'':' &#10060;'}<br><small>${fmtN(r.comparisons)} cmp</small></td>`;}).join('')}</tr>`).join('')+`</table>`;};
    for(const c of conds)for(const a of algos){
      tk(done/total,`running <b>${a}</b> on <b>${c} ${CONDS[c]}</b>, n=${fmtN(n)}`);
      res[a+'|'+c]=await api('sort',{algo:a,n,cond:c,rows:0});done++;draw();
    }
    tk(1,'race finished!');btn.disabled=false;
    const w3=conds.includes('C3')?(()=>{let b=null;algos.forEach(a=>{const r=res[a+'|C3'];if(r&&r.seconds!=null&&(b==null||r.seconds<res[b+'|C3'].seconds))b=a;});return b;})():null;
    $('#asay',v).innerHTML=say('bear','Finished! '+(w3?`On pre-ranked data (C3) the winner is <b>${w3}</b>. `:'')+'Look at the red cells and trophies: no algorithm wins every shape.')+say('masha','Now try n = 10,000 on the Big Results tab to see where the quadratic ones stop finishing (DNF)!');
    chat('masha','Whoever wins, remember: all runs were verified against sorted()!');
  };
  chat('bear','Choose contenders, then press Start.');
};

/* CONSOLE */
VIEWS.console=v=>{
  const ALG=SORTS.concat(['Size-K Heap (top-K)','Bounded Insertion Buffer (top-K)']);
  v.innerHTML=section('Bonus','Analyst Console','build the shift queue with any algorithm and see the actual cases','masha')+
   say('bear','Choose the algorithm that builds the queue, how many cases K the analysts can handle, how many records n, and the input order. The table shows each case with band, risk, Amount and Time. &#128681; marks real frauds (used only for evaluation).')+
   `<div class="card"><div class="row"><label>Algorithm<select id="ca">${ALG.map(s=>`<option ${s=='Size-K Heap (top-K)'?'selected':''}>${s}</option>`).join('')}</select></label>
    <label>K (queue size)<input type="number" id="ck" value="${INFO.K}" min="1" style="width:110px"></label>
    <label>n (records)<select id="cn">${[1000,10000,100000,284807].map(x=>`<option ${x==100000?'selected':''}>${x}</option>`).join('')}</select></label>
    <label>Input order<select id="cc"><option value="C1">C1 Arrival</option><option value="C2">C2 Shuffled</option><option value="C3">C3 Pre-ranked</option><option value="C4">C4 Reverse-ranked</option><option value="C5">C5 Nearly ranked</option></select></label>
    <button class="btn" id="cg">&#127855; Build the queue!</button></div></div><div id="cout"></div>`;
  $('#cg',v).onclick=async()=>{
    const btn=$('#cg',v);btn.disabled=true;const out=$('#cout',v);out.innerHTML=note('Masha is running to the Bear\'s house ...&#127939;&#8205;&#9792;&#65039;');
    const r=await api('sort',{algo:$('#ca',v).value,n:$('#cn',v).value,cond:$('#cc',v).value,K:$('#ck',v).value,rows:1000});btn.disabled=false;
    if(r.error){out.innerHTML=note(esc(r.error),'warn');chat('bear','Hrrm. '+esc(r.error));return;}
    const fr=r.rows.filter(x=>x.fraud).length;
    out.innerHTML=tiles([['Algorithm',r.algo],['Time',fmtT(r.seconds)],['Comparisons',fmtN(r.comparisons)],['Moves',fmtN(r.moves)]])+
     (r.verified?note('&#9989; Output verified against Python\'s sorted()','ok'):note('&#10060; Output differs from sorted()!','bad'))+
     `<div class="card"><h3>Top ${fmtN(Math.min(r.K,r.rows.length))} cases ${r.K>r.rows.length?'(first '+r.rows.length+' shown)':''} &mdash; &#128681; ${fr} real frauds in view</h3><div class="scroll"><table class="tbl"><tr><th>rank</th><th>txn_id</th><th>band</th><th>risk</th><th>Amount</th><th>Time (s)</th><th></th></tr>`+
     r.rows.map((x,i)=>`<tr><td>${i+1}</td><td>${x.txn}</td><td><span class="chip2" style="background:${BANDCOL[x.band]}">${x.band}</span></td><td>${x.risk}</td><td>${x.amount.toFixed(2)}</td><td>${fmtN(x.time)}</td><td>${x.fraud?'&#128681;':''}</td></tr>`).join('')+`</table></div></div>`;
    chat('masha',`${r.algo} took ${fmtT(r.seconds)} with ${fmtN(r.comparisons)} comparisons.`);
  };
  chat('masha','Press the big pink button!');
};

/* LIVE FEED */
VIEWS.live=v=>{
  v.innerHTML=section('Problem E','The Live Feed','records arrive in windows - the Top-K must refresh every time','bear')+
   say('masha','New letters keep arriving! After each window we must publish the current top-K (K = 112). Three strategies: <b>(a)</b> re-rank everything seen so far, <b>(b)</b> keep a size-K heap fed only by the new window, <b>(c)</b> sort the window and merge into the current top-K.')+
   `<div class="card"><div class="row"><label>Window size<select id="lw"><option value="3600">1 hour (48 windows)</option><option value="600">10 minutes (288 windows)</option></select></label>
    <label>(a) re-rank: only first N windows<input type="number" id="lm" value="8" min="1" max="48" style="width:90px"></label></div>
    <div><label class="chk"><input type="checkbox" class="ls" value="a" checked>(a) re-rank everything</label><label class="chk"><input type="checkbox" class="ls" value="b" checked>(b) size-K heap</label><label class="chk"><input type="checkbox" class="ls" value="c" checked>(c) merge window into top-K</label></div>
    <div class="row"><button class="btn" id="lg">&#128236; Run the feed!</button></div></div><div id="ltrk"></div><div id="lout"></div>`;
  $('#lg',v).onclick=async()=>{
    const btn=$('#lg',v);btn.disabled=true;const w=+$('#lw',v).value;const ss=$$('.ls',v).filter(x=>x.checked).map(x=>x.value);
    const tk=track($('#ltrk',v));const series=[],rows=[];const names={a:'(a) re-rank everything',b:'(b) size-K heap',c:'(c) merge window + merge into top-K'},cols={a:'#D55E00',b:'#E69F00',c:'#0072B2'};
    for(let i=0;i<ss.length;i++){const s=ss[i];tk(i/ss.length,`running strategy <b>${names[s]}</b>`);
      const r=await api('live',{strategy:s,window:w,maxw:$('#lm',v).value});
      if(r.error){$('#lout',v).innerHTML=note(esc(r.error),'bad');btn.disabled=false;return;}
      series.push({name:names[s]+(r.partial?` (first ${r.cum.length} windows)`:''),color:cols[s],pts:r.cum.map((t,k)=>[k+1,t]),markers:false});
      rows.push([names[s],r.partial?`first ${r.cum.length} windows`:'whole day',fmtT(r.cum[r.cum.length-1]),fmtN(r.comparisons),r.correct==null?'(partial)':r.correct?'&#9989;':'&#10060;']);}
    tk(1,'feed finished!');btn.disabled=false;
    $('#lout',v).innerHTML=lineChart(series,{logy:true,xl:'window number',yl:'cumulative time (s)',w:760,h:320})+
      `<table class="tbl"><tr><th>strategy</th><th>covered</th><th>time</th><th>comparisons</th><th>final top-K correct</th></tr>${rows.map(r=>'<tr>'+r.map(c=>`<td>${c}</td>`).join('')+'</tr>').join('')}</table>`+
      say('bear','Strategy (a) gets slower and slower every window: it re-sorts everything seen so far. The heap barely notices new windows.');
    chat('masha','The heap wins because most records are rejected after ONE comparison!');
  };
  chat('bear','Run the feed to see which strategy keeps up.');
};

/* BIG RESULTS */
VIEWS.results=async v=>{
  v.innerHTML=section('Problems C-E','Big Results','the full multi-hour benchmark (n up to 284,807, DNF after 5 minutes), replayed from results.csv','masha')+'<div id="rb">'+note('loading ...')+'</div>';
  const R=await api('results');if(R.error){$('#rb',v).innerHTML=note(esc(R.error),'bad');return;}
  const sizes=[1000,10000,100000,284807];
  const board=n=>{const sub=R.C.filter(r=>r.n==n);const win={};Object.keys(CONDS).forEach(c=>{const ok=sub.filter(r=>r.cond==c&&r.status=='OK');ok.sort((a,b)=>a.s-b.s);win[c]=ok.length?ok[0].algo:null;});
    return `<table class="tbl"><tr><th>n = ${fmtN(n)}</th>${Object.entries(CONDS).map(([c,l])=>`<th>${c}<br><small>${l}</small></th>`).join('')}</tr>`+SORTS.map(a=>`<tr><td>${a}</td>${Object.keys(CONDS).map(c=>{const r=sub.find(x=>x.algo==a&&x.cond==c);if(!r)return '<td>-</td>';
      if(r.status=='DNF')return '<td class="dnf">DNF &#10006;</td>';return `<td class="${win[c]==a?'win':''}">${win[c]==a?'&#127942; ':''}${fmtT(r.s)}</td>`;}).join('')}</tr>`).join('')+'</table>';};
  const growth=cond=>{const ser=SORTS.map(a=>{const rows=R.C.filter(r=>r.algo==a&&r.cond==cond).sort((x,y)=>x.n-y.n);
      return {name:a,color:COL[a],pts:rows.filter(r=>r.status=='OK').map(r=>[r.n,r.s]),xs:rows.filter(r=>r.status=='DNF').map(r=>[r.n,300])};});
    return lineChart(ser,{logx:true,logy:true,xl:'n (records)',yl:'median time (s)',hlines:[{y:300,t:'5 min DNF cap'}],w:760,h:340});};
  const D=R.D,ks=[...new Set(D.map(r=>r.K))].sort((a,b)=>a-b),dm=['Full Sort + Slice','Size-K Heap','Bounded Insertion Buffer'];
  const live=(w)=>{const e=R.E[w];return ['a','b','c'].map((s,i)=>({name:{a:'(a) re-rank everything',b:'(b) size-K heap',c:'(c) merge window'}[s],color:{a:'#D55E00',b:'#E69F00',c:'#0072B2'}[s],pts:(e[s]||[]).map((t,k)=>[k+1,t]),markers:false,xs:(R.Edn[w]||[]).includes(s)&&e[s]&&e[s].length?[[e[s].length,e[s][e[s].length-1]]]:[]}));};
  $('#rb',v).innerHTML=`<div class="card"><h3>&#127937; Tournament board</h3><div class="row"><label>Records n<select id="rn">${sizes.map(s=>`<option ${s==284807?'selected':''}>${s}</option>`).join('')}</select></label></div><div id="rbd"></div>${note('&#127942; fastest in that column &middot; red = DNF (a single run over 5 minutes, process killed). Medians of 5 runs (3 or 1 when the first run was slow).','info')}</div>
   <div class="card"><h3>&#128200; Growth with n</h3><div class="row"><label>Input shape<select id="rc">${Object.entries(CONDS).map(([c,l])=>`<option value="${c}" ${c=='C2'?'selected':''}>${c} ${l}</option>`).join('')}</select></label></div><div id="rg"></div>
    ${say('bear','Quadratic cells climb like a rocket and cross the 5-minute line (&#10006; = DNF). The n log n ones stay low. Merge Sort wins on random data from n = 100,000 up.')}</div>
   <div class="card"><h3>&#127855; Top-K (n = 284,807, C2)</h3><table class="tbl"><tr><th>method</th>${ks.map(k=>`<th>K = ${fmtN(k)}</th>`).join('')}</tr>${dm.map(m=>`<tr><td>${m}</td>${ks.map(k=>{const r=D.find(x=>x.algo==m&&x.K==k);return !r?'<td>-</td>':r.status=='DNF'?'<td class="dnf">DNF &#10006;</td>':`<td>${fmtT(r.s)}</td>`;}).join('')}</tr>`).join('')}</table>
    ${say('masha','The heap wins from K &asymp; 100 and stays fast; the buffer degrades quadratically.')}</div>
   <div class="card"><h3>&#128236; Live feed, whole day</h3><div class="row"><label>Windows<select id="rw"><option value="3600">1 hour</option><option value="600">10 minutes</option></select></label></div><div id="rl"></div></div>`;
  const dB=()=>$('#rbd',v).innerHTML=board(+$('#rn',v).value),dG=()=>$('#rg',v).innerHTML=growth($('#rc',v).value),dL=()=>$('#rl',v).innerHTML=lineChart(live($('#rw',v).value),{logy:true,xl:'window number',yl:'cumulative time (s)',w:760,h:300});
  $('#rn',v).onchange=dB;$('#rc',v).onchange=dG;$('#rw',v).onchange=dL;dB();dG();dL();
  chat('bear','Use the drop-downs to explore every size and shape.');
};

/* VERDICT */
VIEWS.verdict=v=>{
  v.innerHTML=section('Problem H','The Verdict','what I would tell the platform lead','bear')+
   `<div class="card"><h3>1. Overnight batch (~285k transactions): ship <u>Merge Sort</u></h3>Fewest comparisons (4.8M), stable, no bad case on any of the six input shapes (never worse than 1.9 s at n = 284,807 on the benchmark laptop). Insertion and Naive Quick Sort are DNF on several shapes.<br><b>Wrong if:</b> keys have very few distinct values (band-only): Hardened Quick Sort is ~10&times; faster; or memory is so tight that Heap Sort's O(1) space matters.</div>
    <div class="card"><h3>2. Shift queue of K = 112: ship the <u>size-K min-heap</u></h3>~91 ms versus ~2.4 s for a full sort at n = 284,807 (about 27&times;). It beats the insertion buffer for every K &ge; 100.<br><b>Wrong if:</b> K is a large fraction of n (sort once instead), or records arrive in strictly improving order.</div>
    <div class="card"><h3>3. Live refresh each window: a <u>size-K heap fed only by the new window</u></h3>Whole-day cost a fraction of a second, versus seconds for merge-window and a DNF for re-ranking everything with 10-minute windows; kilobytes of memory instead of megabytes.<br><b>Wrong if:</b> old records can be re-scored or withdrawn, or windows get so tiny that publishing K log K per window dominates.</div>`+
   say('masha','And the surprise: K = 112 catches almost no frauds. The scoring rule, not the algorithm, limits what the analysts see. Open the Data tab and slide K!')+
   say('bear','Now... can I please go back to sleep? &#128564;');
  chat('bear','Good night, little queue.');
};

route();
</script>
</body></html>
'''

# =====================================================================================
#  SERVER: tiny JSON API (timing + sorting happen here) + the single-page Masha & Bear UI
# =====================================================================================
import base64
import csv
import json
import threading
import time
import webbrowser
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

MASK = (1 << 19) - 1
SORT_NAMES = ["Insertion Sort", "Merge Sort", "Naive Quick Sort", "Hardened Quick Sort", "Heap Sort"]
SLOW_SORTS = ("Insertion Sort", "Naive Quick Sort")
TOPK_NAMES = ("Size-K Heap (top-K)", "Bounded Insertion Buffer (top-K)")
C = None
_lock = threading.Lock()                 # one timing run at a time keeps the measurements clean
_cond_cache = {}


def conds_for(n):
    if n not in _cond_cache:
        if len(_cond_cache) >= 2:
            _cond_cache.pop(next(iter(_cond_cache)))
        _cond_cache[n] = make_conditions(C, n)[0]
    return _cond_cache[n]


def row_of(key, rank):
    i = (key & MASK) - 1
    return {"rank": rank, "txn": int(i + 1), "band": int(C["band"][i]), "amount": float(C["amount"][i]),
            "time": int(C["time"][i]), "risk": round(float(C["risk"][i]), 2), "fraud": int(C["cls"][i])}


def api_info(q):
    band, cls, amount = C["band"], C["cls"], C["amount"]
    N = len(band)
    bc = np.bincount(band, minlength=10)
    fr = np.bincount(band[cls == 1], minlength=10)
    vals, cnt = np.unique(amount, return_counts=True)
    top = np.argsort(-cnt, kind="stable")[:10]
    order = np.argsort(C["key"], kind="stable")
    caught = np.cumsum(cls[order])
    half = int(np.searchsorted(caught, int(np.ceil(cls.sum() / 2)), side="left") + 1)
    ks = np.unique(np.round(np.logspace(0, np.log10(N), 160)).astype(int))
    return {"N": N, "K": K_MINE, "frauds": int(cls.sum()), "k_half": half,
            "distinct_amounts": int(len(vals)), "top10_share": float(cnt[top].sum() / N),
            "bands": [{"band": b, "count": int(bc[b]), "share": float(bc[b] / N), "frauds": int(fr[b]),
                       "fraud_rate": float(fr[b] / bc[b]) if bc[b] else 0.0} for b in range(10)],
            "curve": [[int(k), int(caught[k - 1])] for k in ks]}


def api_caught(q):
    K = max(1, min(int(q["K"]), len(C["key"])))
    return {"K": K, "caught": int(_caught[K - 1])}


def api_sort(q):
    algo, n, cond = q["algo"], int(q["n"]), q["cond"]
    K = max(1, int(q.get("K", 1)))
    nrows = int(q.get("rows", 0))
    if algo in SLOW_SORTS and n > 10000:
        return {"error": "That one is quadratic in the worst case - keep n <= 10,000 here (at big n it is a DNF in the real benchmark)."}
    if algo == "Bounded Insertion Buffer (top-K)" and K > 2000 and n > 10000:
        return {"error": "The bounded buffer costs about K^2 work - with K > 2,000 at this n it would run for minutes (DNF in the real benchmark). Try the heap!"}
    if cond not in COND_NAMES or n not in SIZES + [3000, 5000]:
        return {"error": "unknown n / input order"}
    data = conds_for(n)[cond]
    K = min(K, n)
    with _lock:
        if algo == "Size-K Heap (top-K)":
            t0 = time.perf_counter(); out, cmp, mv = topk_heap(data, K); sec = time.perf_counter() - t0
        elif algo == "Bounded Insertion Buffer (top-K)":
            t0 = time.perf_counter(); out, cmp, mv = topk_buffer(data, K); sec = time.perf_counter() - t0
        else:
            a = data[:]
            t0 = time.perf_counter(); cmp, mv = SORTS[algo](a); sec = time.perf_counter() - t0
            out = a if nrows == 0 else a[:K]
    ref = sorted(data)
    ok = (out == ref[:len(out)]) if algo in TOPK_NAMES else (out == ref or out == ref[:len(out)])
    res = {"algo": algo, "seconds": sec, "comparisons": int(cmp), "moves": int(mv), "verified": bool(ok), "K": K, "n": n}
    if nrows:
        res["rows"] = [row_of(k, i + 1) for i, k in enumerate(out[:min(nrows, K)])]
    return res


def api_live(q):
    strat, wsec, maxw = q["strategy"], int(q["window"]), int(q.get("maxw", 8))
    keys = C["key"].tolist()
    sl = window_slices(C, wsec)
    partial = False
    if strat == "a":
        lim = maxw if wsec == 3600 else min(maxw * 6, len(sl))
        partial = lim < len(sl)
        sl = sl[:lim]
    with _lock:
        cum, top, cmp, mv = live_pass(strat, keys, sl, K_MINE)
    correct = None if partial else bool(top == sorted(keys)[:K_MINE])
    return {"cum": cum, "comparisons": int(cmp), "moves": int(mv), "correct": correct, "partial": partial}


def results_rows():
    p = os.path.join(HERE, "results.csv")
    text = open(p, newline="").read() if os.path.exists(p) else zlib.decompress(base64.b64decode("".join(_RESULTS_B64))).decode()
    return list(csv.DictReader(text.splitlines()))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def api_results(q):
    rows = results_rows()
    Cr = [{"algo": r["algo"], "cond": r["cond"], "n": int(r["n"]), "s": num(r["median_s"]), "status": r["status"]}
          for r in rows if r["problem"] == "C"]
    D = [{"algo": r["algo"], "K": int(float(r["K"])), "s": num(r["median_s"]), "status": r["status"]}
         for r in rows if r["problem"] == "D" and r["n"] == "284807" and r["cond"] == "C2"]
    E, Edn = {"3600": {}, "600": {}}, {"3600": [], "600": []}
    for r in rows:
        if r["problem"] == "E":
            w = str(int(float(r["window"])))
            if r["note"].startswith("window_index=") and r["median_s"]:
                E[w].setdefault(r["algo"], []).append((int(r["note"].split("=")[1]), float(r["median_s"])))
            if r["status"] == "DNF":
                Edn[w].append(r["algo"])
    for w in E:
        for a in E[w]:
            E[w][a] = [t for _, t in sorted(E[w][a])]
    return {"C": Cr, "D": D, "E": E, "Edn": Edn}


API = {"info": api_info, "caught": api_caught, "sort": api_sort, "live": api_live, "results": api_results}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            body, ctype = PAGE.encode("utf-8"), "text/html; charset=utf-8"
        elif u.path.startswith("/api/") and u.path[5:] in API:
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                res = API[u.path[5:]](q)
            except Exception as e:                       # never leave the browser hanging
                res = {"error": f"{type(e).__name__}: {e}"}
            body, ctype = json.dumps(res).encode(), "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def main(port=8000):
    global C, _caught
    if not os.path.exists(CSV) and not os.path.exists(CACHE):
        raise SystemExit("creditcard.csv not found - put it next to midnight_app.py (Kaggle: mlg-ulb/creditcardfraud)")
    print("Masha is loading the data (the first run builds cache.npz, about 10-20 s) ...", flush=True)
    C = load_cache()
    _caught = np.cumsum(C["cls"][np.argsort(C["key"], kind="stable")])
    while True:
        try:
            srv = ThreadingHTTPServer(("localhost", port), Handler)
            break
        except OSError:
            port += 1
    url = f"http://localhost:{port}"
    print(f"\n  The Midnight Queue is open at  {url}\n  (press Ctrl+C to stop)\n", flush=True)
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("Good night!")


if __name__ == "__main__":
    main()

