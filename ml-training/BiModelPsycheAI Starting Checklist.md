# BiModelPsycheAI — Starting Checklist (PHQ_Score focus)

## Phase 0: Environment

- [ ] Create Python venv and activate it
- [ ] Install `pandas`, `librosa`, `soundfile`, `scikit-learn`, `xgboost`, `transformers`
- [ ] Install PyTorch with CUDA support (matching RTX 3050 driver)
- [ ] Verify `torch.cuda.is_available()` returns `True`
- [ ] Set `ROOT` in `config.py` to the unzipped E-DAIC folder
- [ ] Run `01_explore.py` and confirm folders, labels, one transcript, one wav load correctly
- [ ] Inspect one `*egemaps*` and one `*mfcc*` feature CSV — check if it's already a single summary vector or a per-frame sequence

## Phase 1: Manifest

- [ ] Run `02_build_manifest.py`
- [ ] Confirm `manifest.csv` has: pid, split, PHQ_Score, PHQ_Binary, wav path, transcript path, eGeMAPS/MFCC paths
- [ ] Check "ok" column — resolve any participants missing wav or transcript

## Phase 2: Lock targets and metrics

- [ ] Primary target: `PHQ_Score` (regression)
- [ ] Primary metrics: CCC and RMSE
- [ ] Secondary metric: threshold predicted score at PHQ-8 ≥ 10 → report F1/precision/recall
- [ ] Decide this before writing model code (fixes loss function + eval code)

## Phase 3: Tier 1 — Classical Regression Baselines (CPU only, do this first)

- [ ] Build feature sets: eGeMAPS Mean, eGeMAPS Mean+Std, MFCC Mean, MFCC Mean+Std
- [ ] Add a dummy/majority-mean baseline as a floor
- [ ] Train LinearSVR, RandomForestRegressor, XGBRegressor on each feature set (12 combos)
- [ ] Use k-fold cross-validation (train+dev) for model selection — dataset is small
- [ ] Report mean ± std CCC/RMSE per combo → Table 1 of results

## Phase 4: Tier 3 — Modern Foundation Model Embeddings (GPU-light, do this second)

- [ ] Extract frozen text embeddings: Sentence-BERT / BGE / E5
- [ ] Extract frozen audio embeddings: Wav2Vec2 / HuBERT (chunk long audio, mean-pool windows)
- [ ] Cache all embeddings as `.npy` files
- [ ] Train small MLP regression head on cached text embeddings alone
- [ ] Train small MLP regression head on cached audio embeddings alone
- [ ] Concatenate text + audio embeddings → MLP → PHQ_Score (fusion)
- [ ] Report CCC/RMSE → Table 3 of results

## Phase 5: Tier 2 — Deep Learning Reproduction (most finicky, do this last)

- [ ] Text: Transcript → BERT (or DistilBERT) → CNN → PHQ_Score
- [ ] Audio: MFCC sequence → CNN → PHQ_Score (reproduction-specific; separate from any raw log-mel CNN)
- [ ] Fusion: Audio CNN + Text CNN → fusion layer → PHQ_Score
- [ ] Label clearly as "reproduction of Makiuchi et al. (2019)" in the report
- [ ] Report CCC/RMSE → Table 2 of results

## Phase 6: Compare against literature

- [ ] Gather CCC/RMSE from AVEC 2019 baseline (RMSE 6.37)
- [ ] Gather CCC/RMSE from Makiuchi et al. 2019 (CCC 0.497 text, 0.608 audio)
- [ ] Build final summary table: classical vs. reproduction vs. foundation models vs. literature
- [ ] Discuss any gaps vs. published numbers in the report/paper