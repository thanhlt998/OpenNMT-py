#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data / features files and build vocabulary
"""
import codecs
import glob
import sys
import gc
import torch
from functools import partial

from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import split_corpus
import onmt.inputters as inputters
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
import configargparse


def check_existing_pt_files(opt):
    """ Check if there are existing .pt files to avoid overwriting them """
    pattern = opt.save_data + '.{}*.pt'
    for t in ['train', 'valid', 'vocab']:
        path = pattern.format(t)
        if glob.glob(path):
            sys.stderr.write("Please backup existing pt files: %s, "
                             "to avoid overwriting them!\n" % path)
            sys.exit(1)


def build_save_dataset(corpus_type, fields, src_reader, tgt_reader, opt):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        src = opt.train_src
        tgt = opt.train_tgt
    else:
        src = opt.valid_src
        tgt = opt.valid_tgt

    logger.info("Reading source and target files: %s %s." % (src, tgt))

    src_shards = split_corpus(src, opt.shard_size)
    tgt_shards = split_corpus(tgt, opt.shard_size)
    shard_pairs = zip(src_shards, tgt_shards)
    dataset_paths = []
    if (corpus_type == "train" or opt.filter_valid) and tgt is not None:
        filter_pred = partial(
            inputters.filter_example, use_src_len=opt.data_type == "text",
            max_src_len=opt.src_seq_length, max_tgt_len=opt.tgt_seq_length)
    else:
        filter_pred = None
    for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
        assert len(src_shard) == len(tgt_shard)
        logger.info("Building shard %d." % i)
        dataset = inputters.Dataset(
            fields,
            readers=[src_reader, tgt_reader] if tgt_reader else [src_reader],
            data=([("src", src_shard), ("tgt", tgt_shard)]
                  if tgt_reader else [("src", src_shard)]),
            # dirs=[opt.src_dir, None] if tgt_reader else [opt.src_dir],
            sort_key=inputters.str2sortkey[opt.data_type],
            filter_pred=filter_pred
        )

        data_path = "{:s}.{:s}.{:d}.pt".format(opt.save_data, corpus_type, i)
        dataset_paths.append(data_path)

        logger.info(" * saving %sth %s data shard to %s."
                    % (i, corpus_type, data_path))

        dataset.save(data_path)

        del dataset.examples
        gc.collect()
        del dataset
        gc.collect()

    return dataset_paths


def build_save_vocab(train_dataset, fields, opt):
    fields = inputters.build_vocab(
        train_dataset, fields, opt.data_type, opt.share_vocab,
        opt.src_vocab, opt.src_vocab_size, opt.src_words_min_frequency,
        opt.tgt_vocab, opt.tgt_vocab_size, opt.tgt_words_min_frequency,
        vocab_size_multiple=opt.vocab_size_multiple
    )

    vocab_path = opt.save_data + '.vocab.pt'
    torch.save(fields, vocab_path)


def count_features(path):
    """
    path: location of a corpus file with whitespace-delimited tokens and
                    ￨-delimited features within the token
    returns: the number of features in the dataset
    """
    with codecs.open(path, "r", "utf-8") as f:
        first_tok = f.readline().split(None, 1)[0]
        return len(first_tok.split(u"￨")) - 1


def main(opt):
    ArgumentParser.validate_preprocess_args(opt)
    torch.manual_seed(opt.seed)
    check_existing_pt_files(opt)

    init_logger(opt.log_file)
    logger.info("Extracting features...")

    src_nfeats = count_features(opt.train_src) if opt.data_type == 'text' \
        else 0
    tgt_nfeats = count_features(opt.train_tgt)  # tgt always text so far
    logger.info(" * number of source features: %d." % src_nfeats)
    logger.info(" * number of target features: %d." % tgt_nfeats)

    logger.info("Building `Fields` object...")
    fields = inputters.get_fields(
        opt.data_type,
        src_nfeats,
        tgt_nfeats,
        dynamic_dict=opt.dynamic_dict,
        src_truncate=opt.src_seq_length_trunc,
        tgt_truncate=opt.tgt_seq_length_trunc,
        bert_src=opt.bert_src,
        bert_tgt=opt.bert_tgt
        )

    src_reader = inputters.str2reader[opt.data_type].from_opt(opt)
    tgt_reader = inputters.str2reader["text"].from_opt(opt)

    logger.info("Building & saving training data...")
    train_dataset_files = build_save_dataset(
        'train', fields, src_reader, tgt_reader, opt)

    if opt.valid_src and opt.valid_tgt:
        logger.info("Building & saving validation data...")
        build_save_dataset('valid', fields, src_reader, tgt_reader, opt)

    logger.info("Building & saving vocabulary...")
    build_save_vocab(train_dataset_files, fields, opt)


class StoreLoggingLevelAction(configargparse.Action):
    """ Convert string to logging level """
    import logging
    LEVELS = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET
    }

    CHOICES = list(LEVELS.keys()) + [str(_) for _ in LEVELS.values()]

    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(StoreLoggingLevelAction, self).__init__(
            option_strings, dest, help=help, **kwargs)

    def __call__(self, parser, namespace, value, option_string=None):
        # Get the key 'value' in the dict, or just use 'value'
        level = StoreLoggingLevelAction.LEVELS.get(value, value)
        setattr(namespace, self.dest, level)


def preprocess_opts(parser):
    """ Pre-procesing options """
    # Data options
    group = parser.add_argument_group('Data')
    group.add('--data_type', '-data_type', default="text",
              help="Type of the source input. "
                   "Options are [text|img|audio].")

    group.add('--train_src', '-train_src', required=True,
              help="Path to the training source data")
    group.add('--train_tgt', '-train_tgt', required=True,
              help="Path to the training target data")
    group.add('--valid_src', '-valid_src',
              help="Path to the validation source data")
    group.add('--valid_tgt', '-valid_tgt',
              help="Path to the validation target data")

    group.add('--src_dir', '-src_dir', default="",
              help="Source directory for image or audio files.")

    group.add('--save_data', '-save_data', required=True,
              help="Output file for the prepared data")

    group.add('--max_shard_size', '-max_shard_size', type=int, default=0,
              help="""Deprecated use shard_size instead""")

    group.add('--shard_size', '-shard_size', type=int, default=1000000,
              help="Divide src_corpus and tgt_corpus into "
                   "smaller multiple src_copus and tgt corpus files, then "
                   "build shards, each shard will have "
                   "opt.shard_size samples except last shard. "
                   "shard_size=0 means no segmentation "
                   "shard_size>0 means segment dataset into multiple shards, "
                   "each shard has shard_size samples")

    # Dictionary options, for text corpus

    group = parser.add_argument_group('Vocab')
    group.add('--src_vocab', '-src_vocab', default="",
              help="Path to an existing source vocabulary. Format: "
                   "one word per line.")
    group.add('--tgt_vocab', '-tgt_vocab', default="",
              help="Path to an existing target vocabulary. Format: "
                   "one word per line.")
    group.add('--features_vocabs_prefix', '-features_vocabs_prefix',
              type=str, default='',
              help="Path prefix to existing features vocabularies")
    group.add('--src_vocab_size', '-src_vocab_size', type=int, default=50000,
              help="Size of the source vocabulary")
    group.add('--tgt_vocab_size', '-tgt_vocab_size', type=int, default=50000,
              help="Size of the target vocabulary")
    group.add('--vocab_size_multiple', '-vocab_size_multiple',
              type=int, default=1,
              help="Make the vocabulary size a multiple of this value")

    group.add('--src_words_min_frequency',
              '-src_words_min_frequency', type=int, default=0)
    group.add('--tgt_words_min_frequency',
              '-tgt_words_min_frequency', type=int, default=0)

    group.add('--dynamic_dict', '-dynamic_dict', action='store_true',
              help="Create dynamic dictionaries")
    group.add('--share_vocab', '-share_vocab', action='store_true',
              help="Share source and target vocabulary")

    # Truncation options, for text corpus
    group = parser.add_argument_group('Pruning')
    group.add('--src_seq_length', '-src_seq_length', type=int, default=50,
              help="Maximum source sequence length")
    group.add('--src_seq_length_trunc', '-src_seq_length_trunc',
              type=int, default=None,
              help="Truncate source sequence length.")
    group.add('--tgt_seq_length', '-tgt_seq_length', type=int, default=50,
              help="Maximum target sequence length to keep.")
    group.add('--tgt_seq_length_trunc', '-tgt_seq_length_trunc',
              type=int, default=None,
              help="Truncate target sequence length.")
    group.add('--lower', '-lower', action='store_true', help='lowercase data')
    group.add('--filter_valid', '-filter_valid', action='store_true',
              help='Filter validation data by src and/or tgt length')

    group.add('--bert_src', '-bert_src', type=str, default=None,
              choices=['vinai/phobert-base'],
              help='Use bert preprocessing on src side.')
    group.add('--bert_tgt', '-bert_tgt', type=str, default=None,
              choices=['vinai/phobert-base'],
              help='Use bert preprocessing on tgt side.')

    # Data processing options
    group = parser.add_argument_group('Random')
    group.add('--shuffle', '-shuffle', type=int, default=0,
              help="Shuffle data")
    group.add('--seed', '-seed', type=int, default=3435,
              help="Random seed")

    group = parser.add_argument_group('Logging')
    group.add('--report_every', '-report_every', type=int, default=100000,
              help="Report status every this many sentences")
    group.add('--log_file', '-log_file', type=str, default="",
              help="Output logs to a file under this path.")
    group.add('--log_file_level', '-log_file_level', type=str,
              action=StoreLoggingLevelAction,
              choices=StoreLoggingLevelAction.CHOICES,
              default="0")

    # Options most relevant to speech
    group = parser.add_argument_group('Speech')
    group.add('--sample_rate', '-sample_rate', type=int, default=16000,
              help="Sample rate.")
    group.add('--window_size', '-window_size', type=float, default=.02,
              help="Window size for spectrogram in seconds.")
    group.add('--window_stride', '-window_stride', type=float, default=.01,
              help="Window stride for spectrogram in seconds.")
    group.add('--window', '-window', default='hamming',
              help="Window type for spectrogram generation.")

    # Option most relevant to image input
    group.add('--image_channel_size', '-image_channel_size',
              type=int, default=3,
              choices=[3, 1],
              help="Using grayscale image can training "
                   "model faster and smaller")


def _get_parser():
    parser = ArgumentParser(description='preprocess.py')

    opts.config_opts(parser)
    preprocess_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
