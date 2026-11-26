#!/usr/bin/env python
"""
============================================================
NTv3 (Nucleotide Transformer v3) embeddings
============================================================
Computes one embedding per transcript for each subject, from
multi-sequence FASTA files (one file per subject, one sequence per
transcript), with headers formatted as:

    >TRANSCRIPT_ID LOC=...|STRAND=...|LEN=...|GENE=...|TRANSCRIPT=...|
     SOURCE=...|STATE=VCF|VCF_COUNT=N
    (or |REFERENCE / |STATE=REFERENCE for the reference genome)

Mean-pooling (weighted by the attention mask) over the last hidden
layer of the model.

"Reference" shortcut: if a transcript has VCF_COUNT=0 (no mutation
relative to the reference), the embedding already computed for the
same transcript on the reference genome is reused instead of running
the model again. This reference cache is computed once and stored on
disk (<embeddings_dir>/_reference_cache.npz).

Automatic resume: a subject is only (re)processed if its output .npz
is missing (or if --force_recompute is passed). An interrupted subject
never writes a partial output.

------------------------------------------------------------
Usage
------------------------------------------------------------
    python embed_ntv3.py \\
        --fasta_dir /path/to/cohort/fasta \\
        --embeddings_dir /path/to/cohort/embeddings \\
        --model_cache_dir /path/to/model_cache_ntv3

Useful options:
    --offset N / --n_limit N   slice over the sorted subject list
                                (for parallelizing across several jobs)
    --batch_size N               (default 24)
    --max_length N                enables truncation to N tokens
                                   (default: no truncation, dynamic padding)
    --no_bf16                      disables torch.autocast(bfloat16)
    --force_recompute
============================================================
"""
import argparse
import os
import re
import glob

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

# NTv3 is a U-Net architecture (7 downsampling stages) -> token length
# must be a multiple of 2**7 = 128, otherwise the residual
# skip-connections no longer match in size and the forward pass crashes.
NTV3_PAD_MULTIPLE = 2 ** 7

HEADER_RE = re.compile(
    r"^>(?P<record_id>\S+)\s+LOC=(?P<loc>[^|]+)\|STRAND=(?P<strand>[^|]+)\|"
    r"LEN=(?P<len>\d+)\|GENE=(?P<gene>[^|]+)\|TRANSCRIPT=(?P<transcript>[^|]+)\|"
    r"SOURCE=(?P<source>[^|]+)\|"
    r"(?:STATE=VCF\|VCF_COUNT=(?P<vcf_count>\d+)|(?:STATE=)?REFERENCE)\s*$"
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-transcript NTv3 embeddings for a cohort of multi-sequence FASTA files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fasta_dir", default=os.environ.get("FASTA_DIR"),
                    help="Directory containing sub-*.fasta and REF_GRCH38.fasta (or --ref_fasta_path).")
    p.add_argument("--embeddings_dir", default=os.environ.get("EMBEDDINGS_DIR"),
                    help="Output directory for the embedding .npz files.")
    p.add_argument("--model_cache_dir", default=os.environ.get("MODEL_CACHE_DIR"),
                    help="HuggingFace cache directory where the model was pre-downloaded.")
    p.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "InstaDeepAI/NTv3_650M_pre"))
    p.add_argument("--ref_fasta_path", default=os.environ.get("REF_FASTA_PATH"),
                    help="Path to the reference FASTA (default: <fasta_dir>/REF_GRCH38.fasta).")
    p.add_argument("--offset", type=int, default=int(os.environ.get("OFFSET", "0")))
    p.add_argument("--n_limit", type=int, default=int(os.environ.get("N_LIMIT", "0")) or None,
                    help="Number of subjects to process starting at --offset (default: all).")
    p.add_argument("--batch_size", type=int, default=int(os.environ.get("BATCH_SIZE", "24")))
    p.add_argument("--max_length", type=int, default=int(os.environ.get("MAX_LENGTH", "0")) or None)
    p.add_argument("--no_bf16", action="store_true",
                    default=not (os.environ.get("USE_BF16", "1") == "1"))
    p.add_argument("--force_recompute", action="store_true",
                    default=os.environ.get("FORCE_RECOMPUTE", "0") == "1")
    args = p.parse_args()

    missing = [name for name in ("fasta_dir", "embeddings_dir", "model_cache_dir")
               if getattr(args, name) is None]
    if missing:
        p.error(f"Missing required arguments: {', '.join('--' + m for m in missing)} "
                f"(or the equivalent environment variables).")

    if args.ref_fasta_path is None:
        args.ref_fasta_path = os.path.join(args.fasta_dir, "REF_GRCH38.fasta")

    return args


# ── GPU monitoring (memory + power), optional ───────────────────────────────
_NVML_OK = False
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _NVML_OK = True
except Exception as _e:
    print(f"pynvml unavailable ({_e}) — GPU power monitoring disabled "
          f"(GPU memory is still reported via torch).")


def gpu_power_watts():
    if not _NVML_OK:
        return None
    try:
        return pynvml.nvmlDeviceGetPowerUsage(_NVML_HANDLE) / 1000.0
    except Exception:
        return None


def gpu_util_percent():
    if not _NVML_OK:
        return None
    try:
        return pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE).gpu
    except Exception:
        return None


def parse_fasta(path):
    """
    Parses a multi-sequence FASTA. Returns a list of dicts:
    transcript, gene, vcf_count, sequence.
    """
    records = []
    header = None
    seq_chunks = []

    def flush():
        if header is not None:
            m = HEADER_RE.match(header)
            if not m:
                raise ValueError(f"Unrecognized FASTA header in {path}: {header!r}")
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


def main():
    args = parse_args()
    os.makedirs(args.embeddings_dir, exist_ok=True)
    ref_cache_path = os.path.join(args.embeddings_dir, "_reference_cache.npz")
    use_bf16 = not args.no_bf16

    print("Loading NTv3...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, cache_dir=args.model_cache_dir, trust_remote_code=True
    )
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name, cache_dir=args.model_cache_dir, trust_remote_code=True
    )
    model = model.cuda().eval()
    print(f"Model loaded — device: {next(model.parameters()).device}")
    if use_bf16:
        print("bf16 enabled: forward pass under torch.autocast(dtype=bfloat16).")

    def embed_sequences(seqs, batch_size, max_length):
        embeddings = []
        n_batches = (len(seqs) + batch_size - 1) // batch_size
        pbar = tqdm(range(0, len(seqs), batch_size), total=n_batches, desc="  batches", leave=False)
        for i in pbar:
            batch = seqs[i:i + batch_size]
            batch_char_len = max(len(s) for s in batch)

            if max_length is None:
                tokens = tokenizer(
                    batch, return_tensors="pt", truncation=False, padding=True,
                    pad_to_multiple_of=NTV3_PAD_MULTIPLE, return_attention_mask=True,
                ).to("cuda")
            else:
                tokens = tokenizer(
                    batch, return_tensors="pt", truncation=True, padding="max_length",
                    max_length=max_length, return_attention_mask=True,
                ).to("cuda")

            n_tokens = tokens["input_ids"].shape[1]

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    hidden = model(**tokens, output_hidden_states=True).hidden_states[-1]
            attn = tokens["attention_mask"].unsqueeze(-1).float()
            emb = (hidden.float() * attn).sum(1) / attn.sum(1).clamp(min=1)
            embeddings.append(emb.cpu().numpy())

            mem_alloc = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            power = gpu_power_watts()
            util = gpu_util_percent()
            postfix = {
                "seq_nt": f"{batch_char_len:,}", "tokens": f"{n_tokens:,}",
                "gpu_alloc": f"{mem_alloc:.1f}GB", "gpu_reserved": f"{mem_reserved:.1f}GB",
            }
            if power is not None:
                postfix["gpu_power"] = f"{power:.0f}W"
            if util is not None:
                postfix["gpu_util"] = f"{util}%"
            pbar.set_postfix(**postfix)

            del tokens, hidden, attn, emb
            torch.cuda.empty_cache()
        return np.concatenate(embeddings, axis=0)

    def build_or_load_reference_cache():
        if os.path.exists(ref_cache_path) and not args.force_recompute:
            print(f"Reference cache found: {ref_cache_path}")
            data = np.load(ref_cache_path, allow_pickle=True)
            return dict(zip(data["transcript_ids"], data["embeddings"]))

        print(f"Reference cache missing, computing it from {args.ref_fasta_path}...")
        if not os.path.exists(args.ref_fasta_path):
            raise FileNotFoundError(
                f"Reference genome not found: {args.ref_fasta_path}\n"
                "Check --ref_fasta_path or the presence of REF_GRCH38.fasta in --fasta_dir."
            )
        ref_records = parse_fasta(args.ref_fasta_path)
        print(f"  {len(ref_records)} reference transcript(s) found — computing (one-time cost).")

        ref_seqs = [r["sequence"] for r in ref_records]
        ref_transcripts = [r["transcript"] for r in ref_records]
        ref_embeddings = embed_sequences(ref_seqs, args.batch_size, args.max_length)

        np.savez(ref_cache_path, transcript_ids=np.array(ref_transcripts), embeddings=ref_embeddings)
        print(f"  Reference cache saved → {ref_cache_path} (shape={ref_embeddings.shape})")
        return dict(zip(ref_transcripts, ref_embeddings))

    ref_cache = build_or_load_reference_cache()

    subject_fastas = sorted(glob.glob(os.path.join(args.fasta_dir, "sub-*.fasta")))
    n_total_subjects = len(subject_fastas)
    if args.n_limit is not None:
        subject_fastas = subject_fastas[args.offset:args.offset + args.n_limit]
    else:
        subject_fastas = subject_fastas[args.offset:]

    print(f"{n_total_subjects} subject(s) total in {args.fasta_dir}")
    print(f"Slice being processed: ranks [{args.offset} - {args.offset + len(subject_fastas) - 1}] "
          f"({len(subject_fastas)} subject(s))")

    n_done = n_skipped = n_failed = 0
    pbar = tqdm(subject_fastas, desc="Subjects", unit="subject")
    for fasta_path in pbar:
        subject_id = os.path.splitext(os.path.basename(fasta_path))[0]
        out_path = os.path.join(args.embeddings_dir, f"{subject_id}.npz")
        pbar.set_postfix(done=n_done, skipped=n_skipped, failed=n_failed)

        if os.path.exists(out_path):
            if args.force_recompute:
                os.remove(out_path)
            else:
                n_skipped += 1
                continue

        try:
            records = parse_fasta(fasta_path)
        except Exception as e:
            tqdm.write(f"✗ {subject_id} — parsing error: {e}")
            n_failed += 1
            continue

        transcript_ids, gene_ids, vcf_counts = [], [], []
        to_compute_idx, to_compute_seqs = [], []
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
            computed = embed_sequences(to_compute_seqs, args.batch_size, args.max_length)
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
        )

        n_done += 1
        tqdm.write(
            f"[{n_done + n_skipped + n_failed}/{len(subject_fastas)}] {subject_id} "
            f"— {len(records)} transcript(s), {len(records) - len(to_compute_seqs)} from reference, "
            f"{len(to_compute_seqs)} computed, shape={embeddings_arr.shape} → {out_path}"
        )

    print("")
    print(f" Done: {n_done} subject(s) processed, {n_skipped} already present (skipped), {n_failed} failure(s).")
    print(f"Peak reserved GPU memory: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
