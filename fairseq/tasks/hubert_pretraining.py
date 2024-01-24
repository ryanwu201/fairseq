# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import pathlib
import string
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from omegaconf import MISSING

from fairseq.data import Dictionary, HubertDataset
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask

import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary: Dictionary) -> None:
        self.dictionary = dictionary

    def __call__(self, label: str) -> List[str]:
        return self.dictionary.encode_line(
            label,
            append_eos=False,
            add_if_not_exist=False,
        )


@dataclass
class HubertPretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    target_data: Optional[str] = field(default=None, metadata={"help": "path to data directory"})
    fine_tuning: bool = field(
        default=False, metadata={"help": "set to true if fine-tuning Hubert"}
    )
    labels: List[str] = field(
        default_factory=lambda: ["ltr"],
        metadata={
            "help": (
                "extension of the label files to load, frame-level labels for"
                " pre-training, and sequence-level label for fine-tuning"
            )
        },
    )
    label_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, looks for labels in this directory instead",
        },
    )
    target_label_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, looks for labels in this directory instead",
        },
    )
    label_rate: float = field(
        default=-1.0,
        metadata={"help": "label frame rate. -1.0 for sequence label"},
    )
    label_rates: Optional[List[float]] = field(
        default_factory=lambda: None,
        metadata={"help": "label frame rate. -1.0 for sequence label"},
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
                    "sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"},
    )
    max_keep_size: Optional[int] = field(
        default=None,
        metadata={"help": "exclude sample longer than this"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to crop to for batching"},
    )
    single_target: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if set, AddTargetDatasets outputs same keys " "as AddTargetDataset"
        },
    )
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    pad_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )
    dummy_size: Optional[int] = field(
        default=None,
        metadata={"help": "size of dummy train/val sets"},
    )


@register_task("hubert_pretraining", dataclass=HubertPretrainingConfig)
class HubertPretrainingTask(FairseqTask):
    cfg: HubertPretrainingConfig

    def __init__(
            self,
            cfg: HubertPretrainingConfig,
    ) -> None:
        super().__init__(cfg)

        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"HubertPretrainingTask Config {cfg}")

        self.cfg = cfg
        self.fine_tuning = cfg.fine_tuning
        if cfg.fine_tuning:
            self.state.add_factory("target_dictionary", lambda: self.load_dictionaries()[0])
        if not cfg.single_target:
            self.state.add_factory("dictionaries", self.load_dictionaries)

        self.blank_symbol = "<s>"

    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return self.state.target_dictionary

    @property
    def dictionaries(self) -> Optional[List[Dictionary]]:
        if 'dictionaries' not in self.state.state_dict.keys() and 'dictionaries' not in self.state.factories_dict.keys():
            return None
        return self.state.dictionaries

    @classmethod
    def setup_task(
            cls, cfg: HubertPretrainingConfig, **kwargs
    ) -> "HubertPretrainingTask":
        return cls(cfg)

    def load_dictionaries(self):
        label_dir = self.cfg.data if self.cfg.label_dir is None else self.cfg.label_dir
        dictionaries = [
            Dictionary.load(f"{label_dir}/dict.{label}.txt")
            for label in self.cfg.labels
        ]
        return dictionaries

    def get_label_dir(self) -> str:
        if self.cfg.label_dir is None:
            return self.cfg.data
        return self.cfg.label_dir

    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.cfg.data}/{split}.tsv" if not self.cfg.target_data else [
            f"{self.cfg.data}/{split}.tsv", f"{self.cfg.target_data}/{split}.tsv"]
        dicts = [self.target_dictionary] if self.cfg.fine_tuning and self.cfg.single_target else self.dictionaries
        pad_list = [dict.pad() for dict in dicts]
        eos_list = [dict.eos() for dict in dicts]
        procs = [LabelEncoder(dict) for dict in dicts]
        paths = [f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.labels]

        if self.cfg.target_label_dir:
            paths = [paths, [f"{self.cfg.target_label_dir}/{split}.{l}" for l in self.cfg.labels]]

        # hubert v1: pad_audio=True, random_crop=False;
        self.datasets[split] = HubertDataset(
            manifest,
            sample_rate=self.cfg.sample_rate,
            label_paths=paths,
            label_rates=self.cfg.label_rate if not self.cfg.label_rates else self.cfg.label_rates,
            pad_list=pad_list,
            eos_list=eos_list,
            label_processors=procs,
            max_keep_sample_size=self.cfg.max_keep_size,
            min_keep_sample_size=self.cfg.min_sample_size,
            max_sample_size=self.cfg.max_sample_size,
            pad_audio=self.cfg.pad_audio,
            normalize=self.cfg.normalize,
            store_labels=False,
            random_crop=self.cfg.random_crop,
            single_target=self.cfg.single_target,
            dummy_size=self.cfg.dummy_size
        )

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(self, indices: np.array, *args, **kwargs) -> np.array:
        return indices

    def inference_step(
            self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        hypos = super().inference_step(generator=generator, models=models, sample=sample, prefix_tokens=prefix_tokens,
                                       constraints=constraints)
        if generator.cfg.align and (generator.emissions is not None):
            all_outputs = F.log_softmax(generator.emissions, dim=-1)

            song_preds = all_outputs.numpy()
            for i, song_pred in enumerate(song_preds):
                # smoothing
                P_noise = np.random.uniform(low=1e-11, high=1e-10, size=song_pred.shape)
                song_pred = np.log(np.exp(song_pred) + P_noise)
                words = generator.tgt_dict.string(sample['target'][i],
                                                  extra_symbols_to_ignore=[generator.tgt_dict.pad()])
                words = words.split('|')[:-1]
                words_phone = [word.strip().split(' ') for word in words]
                lyrics_p, words_p, idx_word_p, _ = gen_phone_gt(words_phone)

                target = sample['target'][i][sample['target'][0] != generator.tgt_dict.pad()]
                title = pathlib.Path(sample['title'][i]).stem

                word_align, score = alignment(song_pred, idx_word_p, generator.tgt_dict.eos(), target.cpu().numpy())

                write_csv(generator.cfg.pred_dir, title, word_align, words)
                write_lrc(generator.cfg.pred_dir, title, word_align, words)

        return hypos


def write_csv(path, file_name, word_align, words):
    pred_file = pathlib.Path(path)
    pred_file.mkdir(exist_ok=True, parents=True)
    pred_file = pred_file / f'{file_name}.csv'
    resolution = 320 / 16000

    with open(pred_file, 'w') as f:
        for j in range(len(word_align)):
            word_time = word_align[j]
            start_time, end_time = word_time[0] * resolution, word_time[1] * resolution
            f.write(f"{start_time},{end_time},{words[j]}\n")

    logger.info(f'{pred_file} saved')


def write_lrc(path, file_name, word_align, words):
    pred_file = pathlib.Path(path)
    pred_file.mkdir(exist_ok=True, parents=True)
    pred_file = pred_file / f'{file_name}.lrc'
    resolution = 320 / 16000

    with open(pred_file, 'w') as f:
        index_end_of_line = len(word_align) // 2
        for j in range(len(word_align)):
            word_time = word_align[j]

            # seconds
            start_time = word_time[0] * resolution
            minutes = int(start_time // 60)
            remaining_seconds = int(start_time % 60)
            hundredths = int((remaining_seconds - int(remaining_seconds)) * 100)

            start_time = f"{minutes:02d}:{remaining_seconds:02d}.{hundredths:02d}"

            if index_end_of_line < 10:
                if j == 0 or j == index_end_of_line:
                    if j == index_end_of_line:
                        f.write('\n')
                    f.write(f'[{start_time}]')
                f.write(f"<{start_time}>{words[j]}")
            else:
                if j % 10 == 0:
                    f.write(f'[{start_time}]')
                f.write(f"<{start_time}>{words[j]}")
                if (j + 1) % 10 == 0:
                    f.write('\n')

    logger.info(f'{pred_file} saved')


def gen_phone_gt(words, raw_lines=[]):
    # helper function
    def getsubidx(x, y):  # find y in x
        l1, l2 = len(x), len(y)
        for i in range(l1 - l2 + 1):
            if x[i:i + l2] == y:
                return i

    words_p = []
    lyrics_p = []
    for word in words:
        out = word
        out = [phone if phone[-1] not in string.digits else phone[:-1] for phone in out]
        words_p.append(out)
        if len(lyrics_p) > 0:
            lyrics_p.append('|')
        lyrics_p += out

    len_words_p = [len(phones) for phones in words_p]
    idx_in_full_p = []
    s1 = 0
    s2 = s1
    for l in len_words_p:
        s2 = s1 + l
        idx_in_full_p.append([s1, s2])
        s1 = s2 + 1

        # beginning of a line
        idx_line_p = []
        last_end = 0
        for i in range(len(raw_lines)):
            line = []
            line_phone = [word for word in raw_lines[i].split()]
            for l in line_phone:
                line += l + [' ']
            line = line[:-1]
            line = [phone if phone[-1] not in string.digits else phone[:-1] for phone in line]
            offset = getsubidx(lyrics_p[last_end:], line)
            assert (offset >= 0)
            assert (line == lyrics_p[last_end + offset:last_end + offset + len(line)])
            idx_line_p.append([last_end + offset, last_end + offset + len(line)])
            last_end += offset + len(line)

    return lyrics_p, words_p, idx_in_full_p, idx_line_p


def alignment(song_pred, idx, phone_blank, lyrics_int):
    audio_length, num_class = song_pred.shape
    lyrics_length = len(lyrics_int)

    s = np.zeros((audio_length, 2 * lyrics_length + 1)) - np.Inf
    opt = np.zeros((audio_length, 2 * lyrics_length + 1))

    blank = phone_blank

    # init
    s[0][0] = song_pred[0][blank]
    # insert eps
    for i in np.arange(1, audio_length):
        s[i][0] = s[i - 1][0] + song_pred[i][blank]

    for j in np.arange(lyrics_length):
        if j == 0:
            s[j + 1][2 * j + 1] = s[j][2 * j] + song_pred[j + 1][lyrics_int[j]]
            opt[j + 1][2 * j + 1] = 1  # 45 degree
        else:
            s[j + 1][2 * j + 1] = s[j][2 * j - 1] + song_pred[j + 1][lyrics_int[j]]
            opt[j + 1][2 * j + 1] = 2  # 28 degree

        s[j + 2][2 * j + 2] = s[j + 1][2 * j + 1] + song_pred[j + 2][blank]
        opt[j + 2][2 * j + 2] = 1  # 45 degree

    for audio_pos in np.arange(2, audio_length):

        for ch_pos in np.arange(1, 2 * lyrics_length + 1):

            if ch_pos % 2 == 1 and (ch_pos + 1) / 2 >= audio_pos:
                break
            if ch_pos % 2 == 0 and ch_pos / 2 + 1 >= audio_pos:
                break

            if ch_pos % 2 == 1:  # ch
                ch_idx = int((ch_pos - 1) / 2)
                # cur ch -> ch
                a = s[audio_pos - 1][ch_pos] + song_pred[audio_pos][lyrics_int[ch_idx]]
                # last ch -> ch
                b = s[audio_pos - 1][ch_pos - 2] + song_pred[audio_pos][lyrics_int[ch_idx]]
                # eps -> ch
                c = s[audio_pos - 1][ch_pos - 1] + song_pred[audio_pos][lyrics_int[ch_idx]]
                if a > b and a > c:
                    s[audio_pos][ch_pos] = a
                    opt[audio_pos][ch_pos] = 0
                elif b >= a and b >= c:
                    s[audio_pos][ch_pos] = b
                    opt[audio_pos][ch_pos] = 2
                else:
                    s[audio_pos][ch_pos] = c
                    opt[audio_pos][ch_pos] = 1

            if ch_pos % 2 == 0:  # eps
                # cur ch -> ch
                a = s[audio_pos - 1][ch_pos] + song_pred[audio_pos][blank]
                # eps -> ch
                c = s[audio_pos - 1][ch_pos - 1] + song_pred[audio_pos][blank]
                if a > c:
                    s[audio_pos][ch_pos] = a
                    opt[audio_pos][ch_pos] = 0
                else:
                    s[audio_pos][ch_pos] = c
                    opt[audio_pos][ch_pos] = 1

    score = s[audio_length - 1][2 * lyrics_length]

    # retrive optimal path
    path = []
    x = audio_length - 1
    y = 2 * lyrics_length
    path.append([x, y])
    while x > 0 or y > 0:
        if opt[x][y] == 1:
            x -= 1
            y -= 1
        elif opt[x][y] == 2:
            x -= 1
            y -= 2
        else:
            x -= 1
        path.append([x, y])

    path = list(reversed(path))
    word_align = []
    path_i = 0

    word_i = 0
    while word_i < len(idx):
        # e.g. "happy day"
        # find the first time "h" appears
        if path[path_i][1] == 2 * idx[word_i][0] + 1:
            st = path[path_i][0]
            # find the first time " " appears after "h"
            while path_i < len(path) - 1 and (path[path_i][1] != 2 * idx[word_i][1] + 1):
                path_i += 1
            ed = path[path_i][0]
            # append
            word_align.append([st, ed])
            # move to next word
            word_i += 1
        else:
            # move to next audio frame
            path_i += 1
    assert len(idx) == len(word_align)
    return word_align, score
