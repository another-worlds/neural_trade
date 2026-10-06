"""Paired comparison of a screen spec against hc_baseline on the same (slice, seed) pairs.
usage: python hc_compare.py <candidate_name> ; reads runs/tactical/screens/<name>/results*.jsonl
Reports the per-slice mean diff, the mean over slices (the unit of inference = slice) and a t-interval over slices;
also the pooled per-trial diff SE. Mean AUC = mean of h0-h2 val AUC."""
import glob, json, sys, statistics as st, math, collections
def load(name):
    d = {}
    for f in glob.glob(f'runs/tactical/screens/{name}/results*.jsonl'):
        for l in open(f):
            r = json.loads(l); a = r['direction_auc']
            if all(a.get(h) and a[h].get('auc') is not None for h in ('h0', 'h1', 'h2')):
                d[(r['data_end'], r['seed'])] = st.mean(a[h]['auc'] for h in ('h0', 'h1', 'h2'))
    return d
b, c = load('hc_baseline'), load(sys.argv[1])
keys = sorted(set(b) & set(c)); by = collections.defaultdict(list)
for k in keys: by[k[0]].append(c[k] - b[k])
print('paired pairs', len(keys), '| candidate mean %.4f baseline mean %.4f' % (st.mean(c[k] for k in keys), st.mean(b[k] for k in keys)))
sl = []
for s, v in sorted(by.items()):
    print(s[:10], 'n=%d diff %+.4f (SE %.4f)' % (len(v), st.mean(v), st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else float('nan'))); sl.append(st.mean(v))
if len(sl) > 1:
    m, se = st.mean(sl), st.stdev(sl) / math.sqrt(len(sl)); t = {2:12.7,3:4.30,4:3.18,5:2.78,6:2.57}.get(len(sl), 2.0)
    print('mean diff over %d slices %+.4f, 95%% CI [%+.4f, %+.4f] (t over slices)' % (len(sl), m, m - t * se, m + t * se))
