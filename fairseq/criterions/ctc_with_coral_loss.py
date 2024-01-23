# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.ctc import CtcCriterion, CtcCriterionConfig
from fairseq.data.data_utils import post_process
from fairseq.logging.meters import safe_round
from fairseq.tasks import FairseqTask


# Refer to https://github.com/DenisDsh/PyTorch-Deep-CORAL/blob/f42f4786ecaf94f8c2e537c11648d90ecf66b9dc/coral.py

def coral_loss(features_source, features_target, pooling_style, dim=0):
    def pooling(features, pooling_style, dim=0):
        return features.mean(dim=dim) if pooling_style == 'mean' else features.max(dim=dim)[0]

    features_source, features_target = pooling(features_source, pooling_style, dim=dim), pooling(
        features_target, pooling_style, dim=dim)
    return coral(features_source, features_target)


def coral(source, target):
    d = source.size(1)  # dim vector

    source_c = compute_covariance(source)
    target_c = compute_covariance(target)

    loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

    loss = loss / (4 * d * d)
    return loss


def compute_covariance(input_data):
    """
    Compute Covariance matrix of the input data
    """
    n = input_data.size(0)  # batch_size

    # Check if using gpu or cpu
    if input_data.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    id_row = torch.ones(n).resize(1, n).to(device=device)
    sum_column = torch.mm(id_row, input_data.float())
    mean_column = torch.div(sum_column, n)
    term_mul_2 = torch.mm(mean_column.t(), mean_column)
    d_t_d = torch.mm(input_data.t(), input_data)
    c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

    return c


@dataclass
class CtcCoralLossCriterionConfig(CtcCriterionConfig):
    ctc_loss_w: float = field(
        default=1.0,
        metadata={"help": "update probability of ctc"},
    )
    coral_loss_w: float = field(
        default=0.5,
        metadata={"help": "Weight that trades off the coral_loss for output"},
    )
    coral_loss_w4conv: float = field(
        default=0.0,
        metadata={"help": "Weight that trades off the coral_loss for align conv features"},
    )
    pooling_style: str = field(
        default='mean',
        metadata={"help": "pooling_style[mean/max],default mean"},
    )
    pass


@register_criterion("ctc_with_coral_loss", dataclass=CtcCoralLossCriterionConfig)
class CtcCoralLossCriterion(CtcCriterion):

    def __init__(self, cfg: CtcCoralLossCriterionConfig, task: FairseqTask, rdrop_alpha: int = 0.0):

        super().__init__(cfg, task, rdrop_alpha)
        self.cfg = cfg

    def forward(self, model, sample, reduce=True, **kwargs):
        net_output = model(
            **sample["net_input"],
            ret_conv=self.cfg.coral_loss_w4conv > 0)  # (T, B, C): (time, batch, phone, pitch) from the encoder
        out_source = net_output["encoder_out"]  # (time, batch, n_ch)
        # features_source = net_output["features"]  # (time, batch, _)
        features_source = out_source  # (time, batch, _)
        lprobs = model.get_normalized_probs(
            {'encoder_out': out_source}, log_probs=True, dim=2
        ).contiguous()

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

        with torch.backends.cudnn.flags(enabled=False):
            losses = {}

            loss_ctc = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

            loss = loss_ctc * self.cfg.ctc_loss_w

            losses['loss_ctc'] = utils.item(loss_ctc.data)

            if model.w2v_encoder.ft and (self.cfg.coral_loss_w > 0 or self.cfg.coral_loss_w4conv > 0):
                target_domain_net_output = model(
                    **sample["target_domain_net_input"],
                    ret_conv=self.cfg.coral_loss_w4conv > 0)  # (T, B, C): (time, batch, phone, pitch) from the encoder

                # align final output
                if self.cfg.coral_loss_w > 0:
                    features_target = target_domain_net_output["encoder_out"]  # (time, batch, _)
                    loss_coral = coral_loss(features_source, features_target, self.cfg.pooling_style, dim=0)

                    loss += (self.cfg.coral_loss_w * loss_coral)

                    losses['loss_coral'] = utils.item(loss_coral.data)

                # align conv features
                if self.cfg.coral_loss_w4conv > 0:
                    features_source = net_output["features"]  # (time, batch, _)
                    features_target = target_domain_net_output["features"]  # (time, batch, _)
                    loss_coral_conv = coral_loss(features_source, features_target, self.cfg.pooling_style, dim=0)

                    loss += (self.cfg.coral_loss_w4conv * loss_coral_conv)

                    losses['loss_coral_conv'] = utils.item(loss_coral_conv.data)

        losses['loss'] = utils.item(loss.data)
        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "losses": losses,
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        if not model.training:
            sample = sample['target_domain_batch']
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
