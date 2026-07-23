# What makes a BM25-vs-dense comparison fair? — IR evaluation literature review

> Purpose: ground truth for judging whether our 3-stack eval (BM25-over-wiki vs
> FAISS-dense-over-docs vs hybrid; n=260 queries; hit@3; small clean corpus) is a
> fair test, before pre-registering kill criteria on it.
>
> Method: primary sources only, verified via arXiv/ACM/publisher pages on
> 2026-07-23. Each claim carries its source. Claims are tagged **[documented]**
> (stated in the source) or **[inferred]** (our extrapolation from documented
> claims to our setup). Conflicting sources are shown side by side.

## Sources index

| Key | Work | ID |
|---|---|---|
| BEIR | Thakur et al. 2021, *BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models*, NeurIPS Datasets & Benchmarks | [arXiv:2104.08663](https://arxiv.org/abs/2104.08663) |
| RZ09 | Robertson & Zaragoza 2009, *The Probabilistic Relevance Framework: BM25 and Beyond*, FnTIR 3(4):333–389 | DOI [10.1561/1500000019](https://doi.org/10.1561/1500000019) ([PDF](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)) |
| DPR | Karpukhin et al. 2020, *Dense Passage Retrieval for Open-Domain QA*, EMNLP | [arXiv:2004.04906](https://arxiv.org/abs/2004.04906) |
| RG21 | Reimers & Gurevych 2021, *The Curse of Dense Low-Dimensional Information Retrieval for Large Index Sizes*, ACL (short) | [arXiv:2012.14210](https://arxiv.org/abs/2012.14210), [ACL 2021.acl-short.77](https://aclanthology.org/2021.acl-short.77/) |
| Luan21 | Luan, Eisenstein, Toutanova, Collins 2021, *Sparse, Dense, and Attentional Representations for Text Retrieval*, TACL | [arXiv:2005.00181](https://arxiv.org/abs/2005.00181) |
| SAC07 | Smucker, Allan, Carterette 2007, *A comparison of statistical significance tests for information retrieval evaluation*, CIKM | DOI [10.1145/1321440.1321528](https://dl.acm.org/doi/10.1145/1321440.1321528) |
| ULH19 | Urbano, Lima, Hanjalic 2019, *Statistical Significance Testing in IR: An Empirical Analysis of Type I, Type II and Type III Errors*, SIGIR | DOI [10.1145/3331184.3331259](https://dl.acm.org/doi/10.1145/3331184.3331259), [arXiv:1905.11096](https://arxiv.org/abs/1905.11096) |
| WMZ08 | Webber, Moffat, Zobel 2008, *Statistical power in retrieval experimentation*, CIKM | DOI [10.1145/1458082.1458158](https://dl.acm.org/doi/10.1145/1458082.1458158) |
| VB02 | Voorhees & Buckley 2002, *The effect of topic set size on retrieval experiment error*, SIGIR | DOI [10.1145/564376.564432](https://dl.acm.org/doi/10.1145/564376.564432) |
| BV00 | Buckley & Voorhees 2000, *Evaluating evaluation measure stability*, SIGIR | DOI [10.1145/345508.345543](https://dl.acm.org/doi/10.1145/345508.345543) |
| Sakai | Sakai 2018, *Laboratory Experiments in Information Retrieval: Sample Sizes, Effect Sizes, and Statistical Power*, Springer (book; topic-set-size design) | DOI [10.1007/978-981-13-1199-4](https://link.springer.com/book/10.1007/978-981-13-1199-4) |
| Fuhr17 | Fuhr 2017, *Some Common Mistakes In IR Evaluation, and How They Can Be Avoided*, SIGIR Forum 51(3):32–41 | DOI [10.1145/3190580.3190586](https://dl.acm.org/doi/10.1145/3190580.3190586) |
| Sakai20 | Sakai 2020, *On Fuhr's Guideline for IR Evaluation*, SIGIR Forum 54(1) | DOI [10.1145/3451964.3451976](https://dl.acm.org/doi/10.1145/3451964.3451976) ([PDF](http://www.sigir.org/wp-content/uploads/2020/06/p14.pdf)) |
| Ren22 | Ren et al. 2022, *A Thorough Examination on Zero-shot Dense Retrieval* | [arXiv:2204.12755](https://arxiv.org/abs/2204.12755) |
| Sciav21 | Sciavolino et al. 2021, *Simple Entity-Centric Questions Challenge Dense Retrievers*, EMNLP | [arXiv:2109.08535](https://arxiv.org/abs/2109.08535) |
| MARCO | MS MARCO passage ranking task (official metric page) | [arXiv:1611.09268](https://arxiv.org/abs/1611.09268), [leaderboard datasets page](https://microsoft.github.io/msmarco/Datasets.html) |
| Prompt | Dai et al. 2022, *Promptagator: Few-shot Dense Retrieval From 8 Examples* | [arXiv:2209.11755](https://arxiv.org/abs/2209.11755) |

Scope note on our setup, before the axes: the literature's fairness premise is
that both retrievers run **over the same corpus with the same judgments** (BEIR
evaluates every model on identical corpora per dataset; BEIR §3). Our 3-stack
eval compares BM25-over-*wiki* against dense-over-*docs* — different corpora per
arm. That makes it a fair test of **stacks/pipelines**, but NOT a fair test of
**retrieval algorithms**: any observed difference confounds retriever family
with corpus content/granularity. **[inferred** from BEIR's same-corpus design**]**
The kill criteria must be phrased about stacks, not about "BM25 vs dense".

---

## 1. Corpus size and document/passage length

### Documented claims

- **BM25 is a robust zero-shot baseline; dense retrievers' advantage is largely
  in-domain.** On BEIR's 18 datasets, BM25 "generally outperforms" many more
  complex approaches zero-shot, while dense models that beat BM25 by 7–18 points
  in-domain (MS MARCO) "perform significantly worse than BM25" on many other
  datasets; "in-domain performance is not a good indicator for out-of-domain
  generalization" (BEIR §5, Findings 1 & 3, Table 2). **[documented]**
- **Dense retrieval degrades faster than BM25 as index size grows.** Reimers &
  Gurevych show "theoretically and empirically that the performance for dense
  representations decreases quicker than sparse representations for increasing
  index sizes," down to a tipping point where sparse outperforms dense; lower
  embedding dimensionality increases false-positive chance (RG21, abstract).
  **[documented]**
- **Fixed-length dense encodings lose precision on long documents.** Luan et al.
  establish "connections between the encoding dimension, the margin between gold
  and lower-ranked documents, and the document length, suggesting limitations in
  the capacity of fixed-length encodings to support precise retrieval of long
  documents" (Luan21, abstract/§ theory). **[documented]**
- **BM25 handles length via tunable soft normalization.** RZ09 §3.4.5: length
  variation is a mix of a *verbosity* hypothesis (normalize tf by length) and a
  *scope* hypothesis (don't), motivating soft normalization
  `B = (1−b) + b·dl/avdl`, `0 ≤ b ≤ 1`; "setting b = 1 will perform full
  document-length normalisation, while b = 0 will switch normalisation off"
  (RZ09 eqs. 3.12–3.15). BM25's length handling is thus explicit and tunable,
  not a hard cutoff. **[documented]**
- **Dense retrievers have documented length preferences.** BEIR Finding 6 /
  Appendix H: dense/neural retrievers show a "preference for shorter or longer
  documents" depending on their training regime, which materially shifts
  results across datasets with different length distributions (BEIR corpora
  span 3,633 docs / avg 11 words to 15M docs / avg 635 words, Table 1).
  **[documented]**
- DPR sidesteps document length entirely by splitting Wikipedia into ~100-word
  passages (DPR §3.1) — the standard dense workaround is chunking/truncation,
  not normalization. **[documented** in DPR's setup; the generalization that
  "truncation is the standard dense workaround" is **inferred]**

### What would make OUR setup unfair on this axis

- Our corpus (tens of docs / ~10² wiki sections) is 1–2 orders of magnitude
  below the smallest BEIR corpus (NFCorpus, 3,633 docs). RG21's curse runs in
  the direction of *large* indexes hurting dense; by the same mechanism a tiny
  index has almost no false-positive neighbors, so **a tiny corpus flatters
  dense retrieval relative to how it would behave at production scale**.
  **[inferred** — RG21 documents the large-index direction only**]**
- Also on tiny corpora, IDF statistics are estimated from very few documents,
  so BM25's term weighting is noisy relative to web-scale collections; we found
  no primary source quantifying BM25 IDF degradation at N≈10²
  (**not verified — gap**). Treat any BM25-vs-dense gap here as
  non-transferable to larger corpora in either direction. **[inferred]**
- If wiki sections (BM25 arm) and doc chunks (dense arm) have systematically
  different lengths, BEIR Finding 6 + Luan21 say length distribution alone can
  move the result. Check and report the length distributions of both corpora
  alongside the scores. **[inferred from documented length effects]**

## 2. Query set size and statistical power

### Documented claims

- **TREC convention is 50 topics per track**, and the reliability literature is
  built on that: VB02 split TREC's 50-topic sets into disjoint 25-topic halves
  and found that with 25 topics an absolute MAP difference of ~8–9% was needed
  before the two halves agreed on the run ordering with <5% error (VB02).
  **[documented]**
- **Rules of thumb validated:** "the number of queries needed for a good
  experiment is at least 25 and 50 is better" (BV00, abstract). **[documented]**
- **Even 50 topics is often underpowered.** WMZ08 frame power as the number of
  topics needed to reliably detect a given true superiority, and show that
  estimating the required topic count from prior or trial data "leaves wide
  margins of error" (WMZ08). Sakai's topic-set-size design gives explicit
  formulas/tools (paired t-test, ANOVA, CI based) to compute the topic count
  for a target effect size and power (Sakai). **[documented]**
- **Which paired test:** SAC07 compared t-test, Wilcoxon, sign, bootstrap, and
  permutation on TREC runs and found "little practical difference between the
  randomization, bootstrap, and t tests," while Wilcoxon and sign tests can
  disagree and are discouraged (SAC07). ULH19, on >500M simulated p-values from
  TREC data, again finds the paired t-test well-behaved on Type I error across
  topic-set sizes and recommends it by default, with power rising substantially
  from 25 to 50 to 100 topics (ULH19). **[documented;** the ULH19 details were
  extracted from the paper PDF via automated summarization — spot-check §5–6
  before quoting verbatim**]**

### What would make OUR setup unfair (or underpowered) on this axis

- n=260 queries is comfortably above TREC's 50-topic convention — size itself
  is not the problem. **[documented baseline, inferred application]**
- But hit@3 is a **binary per-query outcome**, so the paired analysis is a
  McNemar/sign-type test on discordant pairs, and power depends on the
  *discordant* count, not on n. Back-of-envelope (standard paired-proportions
  power, our calculation, **[inferred]**): minimal detectable difference at
  α=.05, power .8 is ≈ 2.8·√(ψ/n) where ψ = discordant fraction. With n=260 and
  ψ=0.15 (≈39 queries where the stacks disagree), that is ≈ **6–7 points of
  hit@3** — differences smaller than that are noise. Pre-registered kill
  thresholds must clear this bar, or the eval cannot kill anything.
- Tension to resolve explicitly: SAC07 discourage the *sign* test for IR score
  comparisons, yet with a binary metric the McNemar/exact binomial on
  discordant pairs IS the natural test (SAC07's objection concerned discarding
  magnitude information from continuous metrics like AP; with hit@3 there is no
  magnitude to discard). Alternative that stays inside the literature's
  recommendation: bootstrap over queries on the hit@3 difference (SAC07;
  ULH19). **[inferred reconciliation of documented positions]**

## 3. Choice of evaluation cutoff k

### Documented claims

- **k should fit the judgment structure and the task.** BEIR chose nDCG@10
  because Precision/Recall are rank-unaware and MRR/MAP "fail to evaluate tasks
  with graded relevance judgements" (BEIR §3.3). MS MARCO passage ranking, with
  ~1 relevant passage per query, uses **MRR@10** as its official metric (MARCO
  leaderboard page). DPR evaluates top-k retrieval accuracy at k=20/100 —
  chosen because a downstream reader consumes that many passages (DPR §4).
  **[documented]**
- So **success@k / hit@k with exactly one gold item is a recognized setup**
  (it is the binarized cousin of MS MARCO's MRR@10; DPR's "top-k accuracy" is
  exactly hit@k). **[documented setups; the equivalence framing is inferred]**
- **Shallow cutoffs are less stable.** BV00 show shallow precision cutoffs
  carry roughly double the error rate of MAP at the same topic count
  ("Precision at 30 documents has about twice the average error rate as Average
  Precision"), i.e. shallow metrics need MORE queries for the same confidence
  (BV00). **[documented]**
- **Conflict on record:** Fuhr17 lists averaging reciprocal rank (MRR) as a
  mistake (not interval-scaled); Sakai20 explicitly rejects that argument and
  defends averaging RR (Fuhr17; Sakai20). Both positions are respectable; we
  note the dispute and use hit@3 (a mean of a 0/1 variable), which is an
  ordinary proportion and untouched by the interval-scale objection.
  **[documented conflict; last clause inferred]**
- We found **no primary source on choosing k relative to corpus size** for tiny
  corpora — TREC/BEIR corpora are large enough that k≪N always holds
  (**not verified — gap**).

### What would make OUR setup unfair on this axis

- **Ceiling/chance effects.** With N retrievable items and one gold, a random
  ranker gets hit@3 = 3/N. At N≈50–120 sections that is ~2.5–6% chance hit, and
  a merely mediocre retriever saturates quickly; when both stacks sit near the
  ceiling the discordant fraction shrinks, which (axis 2) destroys power and
  compresses real quality differences. Report hit@1 and MRR alongside hit@3 to
  see separation the ceiling hides. **[inferred; ceiling mechanism consistent
  with BV00's stability results]**
- k=3 is defensible **iff** the downstream answerer consumes top-3 (the DPR
  precedent: pick k to match the consumer). If the bot's context window takes a
  different number of chunks, align k with it. **[documented precedent, inferred
  application]**

## 4. Query provenance (how queries were generated)

### Documented claims

- **Queries written while looking at the document favor BM25.** DPR on SQuAD:
  "the annotators wrote questions after seeing the passage. As a result, there
  is a high lexical overlap between passages and questions, which gives BM25 a
  clear advantage" — one of two reasons DPR excluded SQuAD from multi-dataset
  training (DPR §5.1/§6.1). **[documented, exact quote]**
- **Lexical overlap predicts which family wins.** Ren22 §3.4 (Fig. 3): "BM25
  overall performs better on the target dataset with a larger overlap
  coefficient," while dense retrieval beats BM25 at low overlap; and the bias
  often originates in dataset construction, since "sparse models are often used
  to create the annotation data for dataset construction, resulting in the
  lexical bias on target datasets" (Ren22 §3.4). **[documented]**
- **Benchmark judgments themselves can be lexically biased.** BEIR §6: many
  datasets used TF-IDF/BM25 to retrieve annotation candidates, which
  "disfavours approaches that don't rely on lexical matching, like dense
  retrieval methods, as retrieved hits without lexical overlap are
  automatically assumed to be irrelevant"; after manually judging unjudged
  "holes" on TREC-COVID, dense ANCE gained +6.7 nDCG points while docT5query
  moved +0.1 (BEIR §6, Table 4). **[documented]**
- **Query *type* matters independently of overlap.** Simple entity-centric
  questions ("Where was Arve Furset born?") make dense retrievers "drastically
  underperform" BM25; dense models generalize only to common entities or
  question patterns seen in training (Sciav21). **[documented]**
- **Synthetic-query pipelines filter through a retriever.** Promptagator/InPars
  generate LLM queries from documents, then keep only queries whose source
  document is retrieved top-K by a retriever trained on the same synthetic data
  (round-trip consistency filtering) (Prompt §3; InPars,
  [arXiv:2202.05144](https://arxiv.org/abs/2202.05144)). That filtering
  mechanism selects for queries answerable by the filtering retriever's family.
  **[mechanism documented; the bias consequence for *evaluation* use is
  inferred — these papers use synthetic queries for training, not eval]**

### What would make OUR setup unfair on this axis

- Our eval queries are **LLM paraphrases generated from the corpus documents**
  (eval/paraphrase_comparison). Two documented forces pull in opposite
  directions: generation-from-the-document pushes overlap up (DPR's SQuAD
  effect → favors BM25); paraphrasing pushes overlap down (Ren22 → favors
  dense). The net direction is **not knowable a priori** — it depends on how
  aggressive the paraphrases are. **[inferred from documented mechanisms]**
- Mitigation the literature supports: compute a query–gold lexical overlap
  coefficient (Ren22's diagnostic) for all 260 queries, report the eval result
  **stratified by overlap tercile**, and check that neither stack's win rate is
  driven by a single stratum. If the overlap distribution of our synthetic
  queries differs wildly from real user questions, the whole comparison is
  measuring the paraphraser, not the retrievers. **[inferred; diagnostic is
  documented in Ren22]**
- Gold-label provenance: if the "gold section" for each query was picked by the
  same process that wrote the query (single-gold, no pooling), we inherit
  BEIR §6's hole problem in miniature — an answer-bearing section the dense arm
  finds that isn't the designated gold counts as a miss. On a small corpus,
  manually auditing a sample of dense-arm "misses" for false negatives is
  cheap and directly mirrors BEIR's Table 4 exercise. **[documented problem,
  inferred mitigation]**

---

## Bottom line for the 3-stack eval

Fair enough to proceed **if and only if** the kill criteria are written to
respect four constraints: (1) claims are about *stacks over their own corpora*,
never "BM25 vs dense" in general — and nothing measured at N≈10² transfers to
larger corpora (RG21 direction); (2) the kill threshold on hit@3 exceeds the
minimal detectable difference implied by the discordant-pair count (≈6–7 pts if
~15% of queries are discordant; compute the real ψ before registering); (3)
report hit@1/MRR beside hit@3 to expose ceiling compression, and justify k=3 by
the downstream consumer; (4) stratify results by query–gold lexical overlap and
audit a sample of losing-arm misses for gold-label false negatives before
declaring a winner.
