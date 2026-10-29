# One hundred valid fixed-graph generations

Two independent disposable copies of one new legal G1 were run serially without
CI or another benchmark. Both keep 123 nodes / 123 edges and perform **zero Growth
steps**. Each actually computes and commits 99 seven-row neural Reevaluation
proposals through the existing format-3 checkpoint, closed scientific export and
exact receipt commit path. This measures metadata/history cost; it is not evidence
of 100 generations of search or Training. Every raw proposal remains archived.

The baseline keeps all 100 complete boundaries. The rolling arm requests explicit
retirement every five generations, using the existing analyzer and ownership/pins.
At G100 it keeps exactly G98/G99/G100 complete, with all 99 receipts, all sealed
descriptors/configs and actual frozen models retained. Paired complete checkpoint,
selector, search RNG and rollout RNG hashes match at G2/5/10/25/50/75/100.

| G100 observation | Full history | Rolling |
| --- | ---: | ---: |
| Complete search/scientific boundaries | 100 | 3 |
| Allocated regular-file bytes | 9,068,544 | 4,308,992 |
| Allocation including directories | 14,491,648 | 6,553,600 |
| Distinct entry inodes | 3,044 | 1,298 |
| File names | 2,118 | 1,148 |
| Current authority bytes | 370,481 | 380,181 |
| Metadata owned Python heap bytes | 518,243 | 528,487 |
| Aggregate authority JSON bytes written | 18,767,263 | 23,214,888 |
| Whole workload wall seconds, excluding interpreter startup | 204.51 | 237.55 |

97 payloads, 970 files and 776 directories were physically removed only in the
rolling copy. Their file blocks total 4,767,744 bytes; the authority's additional
retirement markers cost 8,192 allocated bytes at G100. Total allocation including
directories is lower by 7,938,048 bytes. This is actual observed reclamation, not
the analyzer's conditional estimate. The original G1 and baseline arm remain intact.

Uninstrumented cold-runtime recovery at G100 took about 0.260 seconds in each arm;
old exact receipt ACK took 0.049 / 0.042 seconds. These are warmed-process/filesystem
observations, not cold process startup or confidence intervals. Separate argument
traces show zero historical full-payload validations in either current recovery
or exact ACK. An explicit whole-archive audit validates 99 historical full payloads
in the baseline and two in the rolling arm, checking retired descriptors/models
instead for the other 97. No actual required-byte hashing is replaced by a cache.
The fixed active model means its original bundle is also pinned in the newest
capsule; this does not measure worst-case old-model-era descriptor scan length.

Retirement costs additional scans, fsyncs and authority replacements: it reduces
storage rather than making this small workload faster. Cumulative publication and
receipt metadata remains O(G+R), with O(G²) aggregate rewriting. The current G100
authority is below 0.4 MB and recovery remains below 0.3 seconds for this small graph.
No broader authority rewrite or second journal is justified at this measured scale.
Longer histories still need a separately proved compact active-root/evidence layout;
exact UUID/digest evidence cannot be replaced with watermarks.

The first pilot stopped before authorization because the newly committed head had
not yet been adopted by a reader, so its stable pin was absent. The driver now
performs normal `generation_read_pin` adoption after publication. The analyzer
continues refusing missing pins; no ownership evidence is fabricated by retirement.
A fresh ten-generation pilot then passed before the two complete 100-generation
arms. All failed-pilot artifacts are preserved.

The full [measurements and trace evidence](evidence/metadata-scaling.json) identify
the exact B package and tool, actual budgets and resources. Source/wheel
identity is independent of package version. No model was activated or trained in
this fixed-model study; genuine multi-era learning is validated separately.
