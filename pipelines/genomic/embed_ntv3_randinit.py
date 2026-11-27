#!/usr/bin/env python
"""
============================================================
NTv3 embeddings with RANDOM WEIGHTS
"Architecture-only" control -- same scheme as the BrainHarmonix control
(corrected 2026-09)
============================================================
Same pipeline as embed_ntv3.py (one embedding per transcript per subject,
attention-mask-weighted mean pooling over the last hidden layer,
VCF_COUNT=0 reference-reuse shortcut, automatic resume), but the model is
built from its configuration only: NO pre-trained checkpoint is loaded.
Comparing the geometry of these embeddings with that of the trained model
isolates what comes from the architecture alone.

--init_scheme {native, pytorch_default}, "native" being the default,
aligned with the corrected version of the BrainHarmonix script:

  * native (default, recommended): the model is built from its config only
    (AutoConfig + explicit resolution of the model class, see
    load_random_model below); weights are drawn by the model code's NATIVE
    scheme (post_init() / _init_weights if any); NO reset is applied
    afterwards. This was already the behaviour of the original script --
    there is therefore NO change of default behaviour, only the explicit
    flag and the diagnostic were added, for consistency with BrainHarmonix.

  * pytorch_default: explicit reset of every sub-module exposing
    reset_parameters() (Linear, LayerNorm, Conv*, Embedding...) after
    construction -- GENERIC PyTorch scheme (kaiming_uniform for Linear...),
    NOT the authors' scheme. SENSITIVITY variant for the init scheme, to be
    compared explicitly with the "native" run, never to be mixed with it in
    the same GW analysis (outputs in a separate sub-folder:
    seed_<N>_pytorch_default/).

Diagnostic _log_weight_std (shape, std and max |bias| of the first
nn.Linear) to check by hand which scheme is actually applied -- useful in
particular to confirm whether post_init() is called by the remote NTv3 code.

The reference cache is seed-dependent (_reference_cache_seed<N>.npz) and
the output folder gets one sub-folder per seed (and per init_scheme if
pytorch_default): embeddings from different random models must never be
mixed.

------------------------------------------------------------
Usage
------------------------------------------------------------
    python embed_ntv3_randinit.py \\
        --fasta_dir /path/to/cohort/fasta \\
        --out_dir /path/to/random_embeddings \\
        --model_cache_dir /path/to/model_cache_ntv3 \\
        --seeds 0 1 2

The model files must be in --model_cache_dir (Hugging Face cache layout, or
--local_model_dir / NTV3_LOCAL_DIR pointing to a snapshot): the script runs
offline (HF_HUB_OFFLINE=1), as on compute nodes without internet access.

Inputs: --fasta_dir contains one multi-sequence FASTA per subject
(sub-*.fasta) and, unless --ref_fasta is given, the reference genome
REF_GRCH38.fasta (same header format as embed_ntv3.py).

Options also readable from environment variables (command line wins):
    SEEDS            (default "0" -- a single seed for testing; comma-separated
                      list for several, e.g. "0,1,2")
    INIT_SCHEME      (default "native"; "pytorch_default" for the sensitivity
                      variant -- see above)
    BATCH_SIZE       (default 24)
    MAX_LENGTH       (default: no truncation)
    OFFSET           (default 0)
    N_LIMIT          (default: no limit)
    USE_BF16         (default 1)
    FORCE_RECOMPUTE  (default 0)
    HF_TOKEN / HUGGINGFACE_HUB_TOKEN (only if the model is not cached yet)
============================================================
"""
import argparse
import glob
import os
import re

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

MODEL_NAME = "InstaDeepAI/NTv3_650M_pre"
NTV3_PAD_MULTIPLE = 2 ** 7  # = 128 (7 downsampling stages, config.json: num_downsamples=7)
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

# set in main()
MODEL_SOURCE = MODEL_NAME
MODEL_CACHE_DIR = None
tokenizer = None
BATCH_SIZE, MAX_LENGTH, FORCE_RECOMPUTE, USE_BF16 = 24, None, False, True


def resolve_local_model_dir(model_cache_dir, local_dir=None):
    """Path of a local snapshot of the model (explicit folder, NTV3_LOCAL_DIR, or the snapshot found in the
    Hugging Face cache); falls back to the repository id. Needed offline: from_config() with
    trust_remote_code does not forward cache_dir when it resolves the model code."""
    env_dir = local_dir or os.environ.get("NTV3_LOCAL_DIR")
    if env_dir:
        print(f"  local model folder (NTV3_LOCAL_DIR / --local_model_dir): {env_dir}")
        return env_dir

    repo_slug = "models--" + MODEL_NAME.replace("/", "--")
    snapshots_root = os.path.join(model_cache_dir, repo_slug, "snapshots")
    if os.path.isdir(snapshots_root):
        candidates = [
            os.path.join(snapshots_root, d) for d in sorted(os.listdir(snapshots_root))
        ]
        candidates = [c for c in candidates if os.path.isdir(c)]
        for c in candidates:
            if os.path.exists(os.path.join(c, "config.json")):
                print(f"  local model folder auto-detected: {c}")
                return c
        if candidates:
            print(f"  WARNING: snapshot(s) found in {snapshots_root} but without "
                  f"config.json. Falling back to the repository id.")

    print(f"  no usable local snapshot in {model_cache_dir}, "
          f"falling back to the repository id {MODEL_NAME}")
    return MODEL_NAME


# ── Multi-sequence FASTA parsing ───────────────────────────────────────────
HEADER_RE = re.compile(
    r"^>(?P<record_id>\S+)\s+LOC=(?P<loc>[^|]+)\|STRAND=(?P<strand>[^|]+)\|"
    r"LEN=(?P<len>\d+)\|GENE=(?P<gene>[^|]+)\|TRANSCRIPT=(?P<transcript>[^|]+)\|"
    r"SOURCE=(?P<source>[^|]+)\|"
    r"(?:STATE=VCF\|VCF_COUNT=(?P<vcf_count>\d+)|(?:STATE=)?REFERENCE)\s*$"
)


def parse_fasta(path):
    records = []
    header = None
    seq_chunks = []

    def flush():
        if header is not None:
            m = HEADER_RE.match(header)
            if not m:
                raise ValueError(f"Unrecognised FASTA header in {path}: {header!r}")
            vcf_count_str = m.group("vcf_count")
            vcf_count = int(vcf_count_str) if vcf_count_str is not None else 0
            records.append({
                "transcript": m.group("transcript"),
                "gene": m.group("gene"),
                "vcf_count": vcf_count,
                "sequence": "".join(seq_chunks),
            })

    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
        flush()

    return records


def _reinit_module_pytorch_default(module: torch.nn.Module) -> int:
    """
    Generic PyTorch reset (kaiming_uniform for Linear, etc.) via
    reset_parameters() -- NOT the model's native scheme if the model applies
    a custom _init_weights through post_init(). Sensitivity variant only,
    see the docstring at the top of the file.
    """
    n_reset = 0
    for m in module.modules():
        reset_fn = getattr(m, "reset_parameters", None)
        if callable(reset_fn):
            reset_fn()
            n_reset += 1
    return n_reset


def _log_weight_std(module: torch.nn.Module, label: str, seed: int):
    """
    Diagnostic: std and max |bias| of the first nn.Linear found. A non-zero
    bias is a strong sign that the code applies the generic PyTorch init
    (reset_parameters, or no custom init) rather than a xavier/trunc_normal
    scheme with zero bias (typical of ViT/MAE, and possibly of a custom
    post_init() on the NTv3 side).
    """
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            std = m.weight.detach().float().std().item()
            bias_max = m.bias.detach().float().abs().max().item() if m.bias is not None else float("nan")
            print(f"  [seed={seed}] {label}: first nn.Linear shape={tuple(m.weight.shape)} "
                  f"std={std:.5f} bias_abs_max={bias_max:.5f} "
                  f"(indicative values -- kaiming_uniform(a=sqrt(5)) ~0.021, xavier_uniform ~0.026-0.036, "
                  f"trunc_normal(0.02) ~0.02; bias ~0 => custom init likely, non-zero bias => "
                  f"generic PyTorch init likely)")
            return
    print(f"  [seed={seed}] {label}: no nn.Linear found for the diagnostic")


def load_random_model(seed: int, init_scheme: str):
    """
    Builds the NTv3 architecture from its config ONLY (no checkpoint
    loaded). With 'native' (default), no reset afterwards -- the
    instantiation alone (the model's post_init()) defines the weights, same
    principle as the corrected BrainHarmonix script. With 'pytorch_default',
    explicit reset to the generic PyTorch init, for the sensitivity test.
    """
    torch.manual_seed(seed)
    config = AutoConfig.from_pretrained(
        MODEL_SOURCE, cache_dir=MODEL_CACHE_DIR, trust_remote_code=True, token=HF_TOKEN
    )

    # The model code lives in a separate repository (config.auto_map); resolve the class explicitly with
    # the right cache_dir, because AutoModelForMaskedLM.from_config() does not forward it (offline failure).
    auto_map = getattr(config, "auto_map", None) or {}
    class_ref = auto_map.get("AutoModelForMaskedLM")

    if class_ref is not None:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        if "--" in class_ref:
            code_repo_id, class_ref = class_ref.split("--", 1)
        else:
            code_repo_id = MODEL_SOURCE

        print(f"  [seed={seed}] explicit class resolution: "
              f"code_repo={code_repo_id} | class_ref={class_ref} | cache_dir={MODEL_CACHE_DIR}")
        model_class = get_class_from_dynamic_module(
            class_ref, code_repo_id, cache_dir=MODEL_CACHE_DIR, token=HF_TOKEN
        )
        model = model_class(config)
    else:
        print(f"  [seed={seed}] no auto_map in the config, standard from_config")
        model = AutoModelForMaskedLM.from_config(config)

    if init_scheme == "pytorch_default":
        n_reset = _reinit_module_pytorch_default(model)
        print(f"  [seed={seed}] init_scheme=pytorch_default: {n_reset} sub-module(s) reset "
              f"to the generic PyTorch init -- sensitivity variant, NOT the native scheme.")
    else:
        print(f"  [seed={seed}] init_scheme=native: no reset after construction -- "
              f"weights as produced by the model code (post_init() if any).")

    model = model.cuda().eval()
    _log_weight_std(model, "ntv3_model", seed)
    print(f"  [seed={seed}] NTv3 model built from the config ONLY "
          f"(no pre-trained checkpoint loaded) -- device: {next(model.parameters()).device}")
    return model


def embed_sequences(model, seqs, batch_size=None, max_length=None):
    batch_size = batch_size or BATCH_SIZE
    max_length = max_length if max_length is not None else MAX_LENGTH
    embeddings = []
    n_batches = (len(seqs) + batch_size - 1) // batch_size
    pbar = tqdm(range(0, len(seqs), batch_size), total=n_batches, desc="  batches", leave=False)
    for i in pbar:
        batch = seqs[i:i + batch_size]
        batch_char_len = max(len(s) for s in batch)

        if max_length is None:
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                truncation=False,
                padding=True,
                pad_to_multiple_of=NTV3_PAD_MULTIPLE,
                return_attention_mask=True,
            ).to("cuda")
        else:
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_attention_mask=True,
            ).to("cuda")

        n_tokens = tokens["input_ids"].shape[1]

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                hidden = model(**tokens, output_hidden_states=True).hidden_states[-1]
        attn = tokens["attention_mask"].unsqueeze(-1).float()
        emb = (hidden.float() * attn).sum(1) / attn.sum(1).clamp(min=1)
        embeddings.append(emb.cpu().numpy())

        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        pbar.set_postfix(
            seq_nt=f"{batch_char_len:,}",
            tokens=f"{n_tokens:,}",
            gpu_alloc=f"{mem_alloc:.1f}GB",
            gpu_reserved=f"{mem_reserved:.1f}GB",
        )

        del tokens, hidden, attn, emb
        torch.cuda.empty_cache()
    return np.concatenate(embeddings, axis=0)


def build_or_load_reference_cache(model, ref_cache_path, ref_fasta_path):
    if os.path.exists(ref_cache_path) and not FORCE_RECOMPUTE:
        print(f"Reference cache found: {ref_cache_path}")
        data = np.load(ref_cache_path, allow_pickle=True)
        return dict(zip(data["transcript_ids"], data["embeddings"]))

    print(f"No reference cache, computing it from {ref_fasta_path}...")
    if not os.path.exists(ref_fasta_path):
        raise FileNotFoundError(
            f"Reference genome not found: {ref_fasta_path}\n"
            "Check the path (--ref_fasta), e.g. if REF_GRCH38.fasta is not "
            "directly inside --fasta_dir."
        )
    ref_records = parse_fasta(ref_fasta_path)
    print(f"  {len(ref_records)} reference transcript(s) found -- computing (once per seed).")

    ref_seqs = [r["sequence"] for r in ref_records]
    ref_transcripts = [r["transcript"] for r in ref_records]
    ref_embeddings = embed_sequences(model, ref_seqs)

    np.savez(
        ref_cache_path,
        transcript_ids=np.array(ref_transcripts),
        embeddings=ref_embeddings,
    )
    print(f"  Reference cache saved -> {ref_cache_path}")
    print(f"  Reference embeddings shape: {ref_embeddings.shape} "
          f"({ref_embeddings.shape[0]} transcripts, dim={ref_embeddings.shape[1]})")
    return dict(zip(ref_transcripts, ref_embeddings))


def run_one_seed(seed: int, init_scheme: str, fasta_dir, out_root, ref_fasta_path, offset=0, n_limit=None):
    suffix = "" if init_scheme == "native" else f"_{init_scheme}"
    out_dir = os.path.join(out_root, f"seed_{seed}{suffix}")
    os.makedirs(out_dir, exist_ok=True)
    ref_cache_path = os.path.join(out_dir, f"_reference_cache_seed{seed}.npz")

    model = load_random_model(seed, init_scheme)
    ref_cache = build_or_load_reference_cache(model, ref_cache_path, ref_fasta_path)

    subject_fastas = sorted(glob.glob(os.path.join(fasta_dir, "sub-*.fasta")))
    n_total_subjects = len(subject_fastas)
    if n_limit is not None:
        subject_fastas = subject_fastas[offset:offset + n_limit]
    else:
        subject_fastas = subject_fastas[offset:]

    print(f"[seed={seed}] {n_total_subjects} subject(s) in total in {fasta_dir}")
    print(f"[seed={seed}] Slice processed: ranks [{offset} - {offset + len(subject_fastas) - 1}] "
          f"({len(subject_fastas)} subject(s))")

    n_done = 0
    n_skipped = 0
    n_failed = 0

    pbar = tqdm(subject_fastas, desc=f"[seed={seed}] Subjects", unit="subject")
    for fasta_path in pbar:
        subject_id = os.path.splitext(os.path.basename(fasta_path))[0]
        out_path = os.path.join(out_dir, f"{subject_id}_randinit_seed{seed}.npz")
        pbar.set_postfix(done=n_done, skipped=n_skipped, failed=n_failed)

        if os.path.exists(out_path):
            if FORCE_RECOMPUTE:
                os.remove(out_path)
            else:
                n_skipped += 1
                continue

        try:
            records = parse_fasta(fasta_path)
        except Exception as e:
            tqdm.write(f"[seed={seed}] X {subject_id} -- parsing error: {e}")
            n_failed += 1
            continue

        transcript_ids = []
        gene_ids = []
        vcf_counts = []
        to_compute_idx = []
        to_compute_seqs = []
        final_embeddings = [None] * len(records)

        for idx, rec in enumerate(records):
            transcript_ids.append(rec["transcript"])
            gene_ids.append(rec["gene"])
            vcf_counts.append(rec["vcf_count"])

            if rec["vcf_count"] == 0 and rec["transcript"] in ref_cache:
                final_embeddings[idx] = ref_cache[rec["transcript"]]
            else:
                to_compute_idx.append(idx)
                to_compute_seqs.append(rec["sequence"])

        if to_compute_seqs:
            computed = embed_sequences(model, to_compute_seqs)
            for local_i, global_idx in enumerate(to_compute_idx):
                final_embeddings[global_idx] = computed[local_i]

        embeddings_arr = np.stack(final_embeddings, axis=0)

        np.savez(
            out_path,
            transcript_ids=np.array(transcript_ids),
            gene_ids=np.array(gene_ids),
            vcf_counts=np.array(vcf_counts),
            embeddings=embeddings_arr,
            n_from_reference=len(records) - len(to_compute_seqs),
            n_computed=len(to_compute_seqs),
            seed=seed,
            init_scheme=init_scheme,
        )

        n_done += 1
        tqdm.write(
            f"[seed={seed}] [{n_done + n_skipped + n_failed}/{len(subject_fastas)}] {subject_id} "
            f"-- {len(records)} transcript(s), "
            f"{len(records) - len(to_compute_seqs)} from the reference, "
            f"{len(to_compute_seqs)} computed, "
            f"embeddings shape={embeddings_arr.shape} (dim={embeddings_arr.shape[1]}) "
            f"-> {out_path}"
        )

    print(f"[seed={seed}] Done: {n_done} subject(s) processed, {n_skipped} already present (skipped), "
          f"{n_failed} failure(s).")
    print(f"[seed={seed}] Peak reserved GPU memory: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")

    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main():
    global MODEL_SOURCE, MODEL_CACHE_DIR, tokenizer, BATCH_SIZE, MAX_LENGTH, FORCE_RECOMPUTE, USE_BF16
    env = os.environ.get
    ap = argparse.ArgumentParser(description="NTv3 embeddings with random weights (architecture-only control)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--fasta_dir", required=True, help="folder with one sub-*.fasta per subject")
    ap.add_argument("--out_dir", required=True, help="output root; one sub-folder seed_<N>[_pytorch_default]/ per seed")
    ap.add_argument("--model_cache_dir", required=True, help="Hugging Face cache containing NTv3 (and its code repo)")
    ap.add_argument("--local_model_dir", default=None, help="explicit local snapshot folder (else NTV3_LOCAL_DIR or auto-detection)")
    ap.add_argument("--ref_fasta", default=None, help="reference genome FASTA (default: <fasta_dir>/REF_GRCH38.fasta)")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[int(s) for s in env("SEEDS", "0").split(",") if s.strip() != ""])
    ap.add_argument("--init_scheme", choices=["native", "pytorch_default"], default=env("INIT_SCHEME", "native"))
    ap.add_argument("--batch_size", type=int, default=int(env("BATCH_SIZE", "24")))
    ap.add_argument("--max_length", type=int, default=int(env("MAX_LENGTH", "0")), help="0: no truncation")
    ap.add_argument("--offset", type=int, default=int(env("OFFSET", "0")))
    ap.add_argument("--n_limit", type=int, default=int(env("N_LIMIT", "0")), help="0: no limit")
    ap.add_argument("--no_bf16", action="store_true", default=env("USE_BF16", "1") != "1")
    ap.add_argument("--force_recompute", action="store_true", default=env("FORCE_RECOMPUTE", "0") == "1")
    a = ap.parse_args()

    MODEL_CACHE_DIR = a.model_cache_dir
    os.environ.setdefault("HF_HUB_CACHE", MODEL_CACHE_DIR)
    BATCH_SIZE, MAX_LENGTH = a.batch_size, (a.max_length or None)
    FORCE_RECOMPUTE, USE_BF16 = a.force_recompute, not a.no_bf16
    MODEL_SOURCE = resolve_local_model_dir(MODEL_CACHE_DIR, a.local_model_dir)
    ref_fasta = a.ref_fasta or os.path.join(a.fasta_dir, "REF_GRCH38.fasta")

    print("Loading the NTv3 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SOURCE, cache_dir=MODEL_CACHE_DIR, trust_remote_code=True, token=HF_TOKEN
    )

    print(f"Seeds to process: {a.seeds} | init_scheme={a.init_scheme}")
    print(f"fasta_dir = {a.fasta_dir}")
    print(f"out_dir   = {a.out_dir}")
    for seed in a.seeds:
        print(f"\n{'=' * 60}\nSEED {seed} (init_scheme={a.init_scheme})\n{'=' * 60}")
        run_one_seed(seed, a.init_scheme, a.fasta_dir, a.out_dir, ref_fasta, a.offset, a.n_limit or None)
    print("\nAll seeds done.")


if __name__ == "__main__":
    main()
