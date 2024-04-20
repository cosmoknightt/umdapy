from mol2vec import features
from pathlib import Path as pt
from umdalib.utils import logger


def gen_corpus(
    smi_file: str,
    radius: int = 1,
    sentence_type: str = "alt",
):
    smi_in_file: pt = pt(smi_file)

    corpus_out_file = (
        smi_in_file.parent
        / f"m2v_corpus_radius_{radius}_sentence_type_{sentence_type}_smi_{smi_in_file.stem}.dat"
    )

    features.generate_corpus(
        in_file=str(smi_in_file.resolve()),
        out_file=str(corpus_out_file.resolve()),
        r=radius,
        sentence_type=sentence_type,
        n_jobs=n_jobs,
    )
    return corpus_out_file


def gen_model(
    corpus_file_in: str | pt,
    vector_size: int = 300,
    min_count: int = 1,
):
    if isinstance(corpus_file_in, str):
        corpus_file_in = pt(corpus_file_in)

    pkl_out = (
        corpus_file_in.parent / f"m2v_model_{vector_size}-dim_{min_count}-min-count.pkl"
    )

    model = features.train_word2vec_model(
        infile_name=str(corpus_file_in.resolve()),
        outfile_name=str(pkl_out.resolve()),
        vector_size=vector_size,
        min_count=min_count,
        n_jobs=n_jobs,
    )

    return model, pkl_out


n_jobs = 32  # Number of cpu cores used for calculation


class Args:
    smi_file: str
    corpus_file: str
    radius: int
    vector_size: int
    sentence_type: str
    n_jobs: int
    min_count: int


def main(args: Args):
    global n_jobs

    if args.n_jobs:
        n_jobs = args.n_jobs

    if not pt(args.smi_file).exists():
        raise FileNotFoundError(f"SMILES file {args.smi_file} does not exist.")

    m2v_model = None
    pkl_file: str = None
    if args.corpus_file:
        m2v_model, pkl_file = gen_model(
            args.corpus_file, args.vector_size, args.min_count
        )
    else:
        smi_file = pt(args.smi_file)
        corpus_file = gen_corpus(smi_file, args.radius, args.sentence_type)
        m2v_model, pkl_file = gen_model(corpus_file, args.vector_size, args.min_count)

    if not pt(pkl_file).exists():
        raise FileNotFoundError(f"Model file {pkl_file} does not exist.")

    if not m2v_model:
        raise ValueError("Model not generated.")

    logger.info(
        f"Model save to {pkl_file} with vector size {m2v_model.vector_size} dimensions."
    )
