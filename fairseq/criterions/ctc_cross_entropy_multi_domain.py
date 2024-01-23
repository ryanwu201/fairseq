# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.ctc import CtcCriterion, CtcCriterionConfig
from fairseq.data.data_utils import post_process
from fairseq.logging.meters import safe_round
from fairseq.tasks import FairseqTask


@dataclass
class CtcCrossEntropyMultiDomainCriterionConfig(CtcCriterionConfig):
    ctc_loss_source_w: float = field(
        default=1.0,
        metadata={"help": "update probability of ctc source"},
    )
    loss_w: float = field(
        default=0.5,
        metadata={"help": "weight of cross entropy loss"},
    )


@register_criterion("ctc_cross_entropy_multi_domain", dataclass=CtcCrossEntropyMultiDomainCriterionConfig)
class CtcCrossEntropyMultiDomainCriterion(CtcCriterion):
    def __init__(self, cfg: CtcCrossEntropyMultiDomainCriterionConfig, task: FairseqTask, rdrop_alpha: int = 0.0):

        super().__init__(cfg, task, rdrop_alpha)
        self.cfg = cfg

    def normalize_frm_label(self, time_label, input_len, feat_len):
        tar = np.full(shape=(time_label.shape[0], feat_len), fill_value=self.task.dictionaries[1].pad_index,
                      dtype=np.short)
        label2feat_ratio = feat_len * self.task.cfg.sample_rate / input_len

        for i in np.arange(time_label.shape[0]):
            for j in np.arange(0, time_label.shape[1], 3):
                st, ed = map(int, time_label[i][j:j + 2] * label2feat_ratio)
                symbol = str(int(time_label[i][j + 2].item()))
                if int(symbol) == self.task.dictionaries[1].pad_index: continue

                symbol_index = self.task.dictionaries[1].index(symbol)

                tar[i][st:ed] = symbol_index
        return torch.tensor(tar, dtype=torch.long).to(time_label.device)

    def calc(self, model, sample, melody=False):
        net_output = model(**sample["net_input"])  # (T, B, C): (time, batch, phone, pitch) from the encoder
        y_lyrics = net_output["encoder_out"]  # (time, batch, n_ch)

        lprobs = model.get_normalized_probs(
            {'encoder_out': y_lyrics}, log_probs=True, dim=2
        ).contiguous()

        sample["target_lengths"] = sample["target_lengths_list"][0]
        sample["ntokens"] = sample["ntokens_list"][0]
        sample["target"] = sample["target_list"][0]

        # CTC loss is calculated over duplicated inputs
        # sample is already duplicated for R-Drop
        if self.rdrop_alpha > 0:
            for k, v in sample.items():
                if k in ["target", "target_lengths"]:
                    sample[k] = torch.cat([v, v.clone()], dim=0)
                elif k == "net_input":
                    if sample[k]["src_tokens"].size(1) != sample[k]["src_lengths"].size(
                            0
                    ):
                        # for decoder CTC loss
                        sample[k]["src_lengths"] = torch.cat(
                            [
                                sample[k]["src_lengths"],
                                sample[k]["src_lengths"].clone(),
                            ],
                            dim=0,
                        )

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            if net_output["padding_mask"] is not None:
                non_padding_mask = ~net_output["padding_mask"]
                input_lengths = non_padding_mask.long().sum(-1)
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )

        pad_mask = (sample["target"] != self.pad_idx) & (
                sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens

        logging_output = {
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }
        output = lprobs, targets_flat, input_lengths, target_lengths, sample
        with torch.backends.cudnn.flags(enabled=False):
            loss_ctc = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )
            output = output, logging_output, loss_ctc
            if melody:
                y_melody = net_output["melody_out"]  # (time, batch, n_p)

                y_melody = y_melody.transpose(1, 2)  # (time, n_p, batch)
                y_melody = y_melody.transpose(0, 2)  # (batch, n_p, time)

                melody_target = self.normalize_frm_label(sample["target_list"][1],
                                                         sample['net_input']['source'].shape[-1],
                                                         y_melody.shape[2])

                loss_ce = F.cross_entropy(y_melody, melody_target)
                output = *output, loss_ce
        return output

    def forward(self, model, sample, reduce=True, **kwargs):
        model.w2v_encoder.w2v_model.feature_grad_mult = 1 if model.w2v_encoder.ft else 0
        _, logging_output, loss_ctc_source = self.calc(model, sample)

        sample_size = logging_output['sample_size']
        sample = sample['target_domain_batch']
        (lprobs, targets_flat, input_lengths, target_lengths,
         sample), logging_output_target, loss_ctc_target, loss_ce = self.calc(model, sample, melody=True)

        loss = self.cfg.ctc_loss_source_w * loss_ctc_source + loss_ctc_target + self.cfg.loss_w * loss_ce
        logging_output['losses'] = {
            'loss_ctc_source': utils.item(loss_ctc_source.data),
            'loss_ctc_target': utils.item(loss_ctc_target.data),
            'loss_ce': utils.item(loss_ce.data),
            'loss': utils.item(loss.data),
        }
        # logging_output = {
        #     # "ntokens": (logging_output["ntokens"], logging_output_target["ntokens"]),
        #     # "nsentences": (logging_output["nsentences"], logging_output_target["nsentences"]),
        #     # "sample_size": sample_size,
        #     "losses": {
        #         'loss_ctc_source': utils.item(loss_ctc_source.data),
        #         'loss_ctc_target': utils.item(loss_ctc_target.data),
        #         'loss_ce': utils.item(loss_ce.data),
        #         'loss': utils.item(loss.data),
        #     }}

        if not model.training:
            import editdistance

            with torch.no_grad():
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                for lp, t, inp_l in zip(
                        lprobs_t,
                        sample["target_label"]
                        if "target_label" in sample
                        else sample["target"],
                        input_lengths,
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    decoded = None
                    if self.w2l_decoder is not None:
                        decoded = self.w2l_decoder.decode(lp)
                        if len(decoded) < 1:
                            decoded = None
                        else:
                            decoded = decoded[0]
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]

                    p = (t != self.task.target_dictionary.pad()) & (
                            t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    toks = lp.argmax(dim=-1).unique_consecutive()
                    pred_units_arr = toks[toks != self.blank_idx].tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    if decoded is not None and "words" in decoded:
                        pred_words = decoded["words"]
                        w_errs += editdistance.eval(pred_words, targ_words)
                        wv_errs += editdistance.eval(pred_words_raw, targ_words)
                    else:
                        dist = editdistance.eval(pred_words_raw, targ_words)
                        w_errs += dist
                        wv_errs += dist

                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_errs
                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        assert len(logging_outputs) > 0
        assert 'losses' in logging_outputs[0]

        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        loss_sum = 0
        loss_names = logging_outputs[0]['losses'].keys()
        for loss_name in loss_names:
            _loss = utils.item(sum(log['losses'].get(loss_name, 0) for log in logging_outputs))
            if loss_name == 'loss':
                loss_sum = _loss
            metrics.log_scalar(
                loss_name, _loss / sample_size / math.log(2), sample_size, round=3
            )

        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
