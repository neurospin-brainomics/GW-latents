####
#The idea of that code is to generate "plausible" biological DNA sequences.
#To do that, we pick genes using pyensembl, and we choose using gnomAD "common" variants.

# Dependencies
# ------------
# - pyensembl (https://github.com/openvax/pyensembl)
#
# - Ensembl GRCh38 primary assembly FASTA (downloaded separately)
#     Stored at: ~/.cache/pyensembl/GRCh38/ensembl109/
#                Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
#     Download:  wget https://ftp.ensembl.org/pub/release-109/fasta/
#                       homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
#     The file is read line-by-line with Python's gzip module (no pyfaidx,
#     which requires BGZF format).
#
# - gnomAD GraphQL API (https://gnomad.broadinstitute.org/api)





from __future__ import annotations

import gzip
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import requests
from pyensembl import EnsemblRelease

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

ENSEMBL_RELEASE = 109
SPECIES         = "human"
GNOMAD_API      = "https://gnomad.broadinstitute.org/api"
RANDOM_SEED     = 42

# Minimal frequency to include a variant

AF_THRESHOLD    = 1e-4

MAX_VARIANTS    = 500


# ── Data structure ─────────────────────────────────────────────────────

@dataclass
class Variant:
    pos : int    # 1-based genomic position
    ref : str    # reference
    alt : str    # alternative
    af  : float  # allele frequency gnomAD
    consequence: str = "unknown"


@dataclass
class SyntheticPatient:
    patient_id      : str
    gene_names      : list[str]
    sequence        : str
    applied_variants: dict[str, list[Variant]] = field(default_factory=dict)

    def n_mutations(self) -> int:
        return sum(len(v) for v in self.applied_variants.values())


# ── Chargement Ensembl ────────────────────────────────────────────────────────

def load_ensembl_db(release: int = ENSEMBL_RELEASE) -> EnsemblRelease:
    log.info(f"Loading Ensembl release {release}...")
    db = EnsemblRelease(release, species=SPECIES)
    try:
        db.genes_by_name("TP53")
    except Exception:
        raise RuntimeError(
            f"Ensembl base not found. Try :\n"
            f"  pyensembl install --release {release} --species human"
        )
    return db


# ── Genomic sequences ────────────────────────────────────────────────────────

def find_genomic_fasta(release: int = ENSEMBL_RELEASE) -> Optional[str]:
    cache_root = os.environ.get(
        "PYENSEMBL_CACHE_DIR",
        os.path.expanduser("~/.cache/pyensembl")
    )
    cache_dir = os.path.join(cache_root, "pyensembl", "GRCh38", f"ensembl{release}")
    if not os.path.isdir(cache_dir):
        return None
    for fname in os.listdir(cache_dir):
        if "dna.primary_assembly" in fname and fname.endswith(".fa.gz"):
            return os.path.join(cache_dir, fname)
    return None

def get_genomic_sequence(gene_name: str, db: EnsemblRelease) -> Optional[tuple]:
    """
    Extract the full genomic sequence of a gene (introns + exons) from the
    Ensembl primary assembly FASTA.
    """
    genes = db.genes_by_name(gene_name)
    if not genes:
        log.warning(f" Can't find '{gene_name}'.")
        return None

    gene  = genes[0]
    chrom = str(gene.contig)
    start = gene.start  # 1-based
    end   = gene.end

    fasta_path = find_genomic_fasta()
    if fasta_path is None:
        log.error("  Can't find genomic FASTA.")
        return None

    try:
        target_seq = []
        in_target  = False
        collected  = 0

        with gzip.open(fasta_path, "rt") as f:
            for line in f:
                line = line.rstrip()
                if line.startswith(">"):
                    header_chrom = line.split()[0].lstrip(">").lstrip("chr")
                    in_target = (header_chrom == chrom)
                    if target_seq:
                        break
                    continue

                if not in_target:
                    continue

                chunk_start = collected
                chunk_end   = collected + len(line)

                ov_start = max(chunk_start, start - 1)
                ov_end   = min(chunk_end,   end)

                if ov_start < ov_end:
                    target_seq.append(line[ov_start - chunk_start:ov_end - chunk_start])

                collected += len(line)
                if collected >= end:
                    break

        seq = "".join(target_seq)
        if not seq:
            log.warning(f"  Empty sequence'{gene_name}'.")
            return None

    except Exception as e:
        log.warning(f"  FASTA failed '{gene_name}' : {e}")
        return None

    log.info(f"  '{gene_name}' : chr{chrom}:{start}-{end} ({len(seq)} bp)")
    return gene, seq


# ── Variants gnomAD via GraphQL ───────────────────────────────────────────────

def fetch_variants(gene_name: str, af_threshold: float = AF_THRESHOLD) -> list[Variant]:
    """
    Fetch all SNVs for a gene from the gnomAD r3 dataset via the public
    GraphQL API (https://gnomad.broadinstitute.org/api).
    """
    query = """
    {
      gene(gene_symbol: "%s", reference_genome: GRCh38) {
        variants(dataset: gnomad_r3) {
          variant_id
          ref
          alt
          consequence
          genome {
            af
          }
        }
      }
    }
    """ % gene_name

    log.info(f"  Request gnomAD GraphQL : {gene_name}")
    try:
        r = requests.post(
            GNOMAD_API,
            json={"query": query},
            timeout=60,  
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning(f"  Requête gnomAD échouée pour '{gene_name}' : {e}")
        return []

    raw_variants = (
        data.get("data", {})
            .get("gene", {})
            .get("variants", [])
    )

    variants = []
    for v in raw_variants:
        ref = v.get("ref", "")
        alt = v.get("alt", "")
        af  = v.get("genome", {})
        af  = af.get("af", 0.0) if af else 0.0

        # SNPs only
        if len(ref) != 1 or len(alt) != 1 or ref == alt:
            continue
        if af is None or float(af) < af_threshold:
            continue

        # variant_id format : "17-43045615-G-A"
        try:
            pos = int(v["variant_id"].split("-")[1])
        except (KeyError, ValueError, IndexError):
            continue

        variants.append(Variant(
            pos=pos,
            ref=ref,
            alt=alt,
            af=float(af),
            consequence=v.get("consequence") or "unknown",
        ))

        if len(variants) >= MAX_VARIANTS:
            break

    variants.sort(key=lambda x: x.pos)
    log.info(
        f"  → {len(variants)} variants kept "
        f"(AF ≥ {af_threshold}, {len(raw_variants)} raw variants)"
    )
    return variants


# ── Variants applications ──────────────────────────────────────────────────

def apply_variants(seq: str, variants: list[Variant], gene_start: int) -> str:
    """
    offset = pos_génomique - gene_start.
    """
    seq_list = list(seq)

    for v in sorted(variants, key=lambda x: x.pos, reverse=True):
        offset = v.pos - gene_start
        if offset < 0 or offset >= len(seq_list):
            continue
        if seq_list[offset].upper() != v.ref.upper():
            log.debug(
                f"  Mismatch ref : attendu {v.ref}, "
                f"trouvé {seq_list[offset]} à offset {offset}"
            )
            continue
        seq_list[offset] = v.alt

    return "".join(seq_list)


# ── Generation ──────────────────────────────────────────────────────

def generate_synthetic_patients(
    gene_names  : list[str],
    n_patients  : int,
    af_threshold: float = AF_THRESHOLD,
    seed        : int   = RANDOM_SEED,
    ensembl_release: int = ENSEMBL_RELEASE,
    gene_separator: str = "N" * 50,  # séparateur entre gènes concaténés
) -> list[SyntheticPatient]:
    """
    Génère n patients, chacun ayant une séquence unique qui concatène
    TOUS les gènes de gene_names (avec leurs variants respectifs).
    """
    rng = random.Random(seed)
    db  = load_ensembl_db(ensembl_release)

    # Précharger séquence de référence + pool de variants pour chaque gène
    gene_data = {}
    for gene_name in gene_names:
        log.info(f"\n{'─'*60}\nGène : {gene_name}\n{'─'*60}")
        result = get_genomic_sequence(gene_name, db)
        if result is None:
            continue
        gene_obj, seq_ref = result
        variant_pool = fetch_variants(gene_name, af_threshold)
        gene_data[gene_name] = (gene_obj, seq_ref, variant_pool)

    all_patients: list[SyntheticPatient] = []

    for idx in range(n_patients):
        seq_parts = []
        variants_by_gene = {}

        for gene_name, (gene_obj, seq_ref, variant_pool) in gene_data.items():
            selected = [v for v in variant_pool if rng.random() < v.af]
            seq_synthetic = apply_variants(seq_ref, selected, gene_obj.start)
            seq_parts.append(seq_synthetic)
            variants_by_gene[gene_name] = selected

        full_sequence = gene_separator.join(seq_parts)

        all_patients.append(SyntheticPatient(
            patient_id=f"patient_{idx:05d}",
            gene_names=list(gene_data.keys()),
            sequence=full_sequence,
            applied_variants=variants_by_gene,
        ))

    n_muts = [p.n_mutations() for p in all_patients]
    log.info(
        f"  ✓ {n_patients} patients multi-gènes ({len(gene_data)} gènes chacun) — "
        f"mutations/patient : min={min(n_muts)}, max={max(n_muts)}, moy={sum(n_muts)/len(n_muts):.1f}"
    )

    return all_patients


# ── Example ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    patients = generate_synthetic_patients(
        gene_names=["BRCA1", "TP53", "EGFR"],
        n_patients=1500,
        af_threshold=AF_THRESHOLD,
    )

    print(f"\n{'='*60}")
    print(f"{len(patients)} patients générés")
    print(f"{'='*60}")
    for p in patients[:6]:
        print(
            f"  {p.patient_id:<16} | {p.n_mutations():>3} variants "
            f"| {p.sequence[:50]}..."
        )
    # Sauvegarde des séquences
    import numpy as np
    sequences = np.array([p.sequence for p in patients])
    np.save(os.path.join(os.environ["WORK"], "sequences.npy"), sequences)
    print(f"Séquences sauvegardées — {len(sequences)} séquences")
