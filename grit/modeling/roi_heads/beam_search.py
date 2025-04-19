# Modified by Jialian Wu from
# https://github.com/microsoft/GenerativeImage2Text/blob/main/generativeimage2text/layers/decoder.py
# and https://github.com/kdexd/virtex
from torch import nn
import torch
import functools
from torch.nn import functional as F
import warnings

class AutoRegressiveBeamSearch(object):
    def __init__(
        self,
        end_token_id: int,
        max_steps: int = 50,
        beam_size: int = 5,
        objectdet=True,
        per_node_beam_size: int = 2,
    ):
        self._eos_index = end_token_id
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.objectdet = objectdet
        self.per_node_beam_size = per_node_beam_size or beam_size

    def search(self, begin_tokens, step):
        if self.beam_size > 1 and self.objectdet:
            only_return_best = False
        else:
            only_return_best = True

        batch_size = begin_tokens.size()[0]

        predictions = begin_tokens.unsqueeze(1).expand((batch_size, self.beam_size, begin_tokens.shape[-1]))
        # Calculate the first timestep. This is done outside the main loop
        # because we are going from a single decoder input (the output from the
        # encoder) to the top `beam_size` decoder outputs. On the other hand,
        # within the main loop we are going from the `beam_size` elements of the
        # beam to `beam_size`^2 candidates from which we will select the top
        # `beam_size` elements for the next iteration.
        # shape: (batch_size, num_classes)
        start_class_logits = step(begin_tokens)

        # Convert logits to logprobs.
        # shape: (batch_size * beam_size, vocab_size)
        start_class_logprobs = F.log_softmax(start_class_logits, dim=1)

        num_classes = start_class_logprobs.size()[1]

        # shape: (batch_size, beam_size), (batch_size, beam_size)
        start_top_logprobs, start_predicted_classes = start_class_logprobs.topk(
            self.beam_size
        )

        if (
            self.beam_size == 1
            and (start_predicted_classes == self._eos_index).all()
        ):
            warnings.warn(
                "Empty object description predicted. You may want to increase beam"
                "size or ensure your step function is working properly.",
                RuntimeWarning,
            )
            if only_return_best:
                return start_predicted_classes, start_top_logprobs
            else:
                return start_predicted_classes.unsqueeze(-1), start_top_logprobs

        # The log probs for the last time step.
        # shape: (batch_size, beam_size)
        last_logprobs = start_top_logprobs

        # shape: (batch_size, beam_size, sequence_length)
        predictions = torch.cat([predictions, start_predicted_classes.unsqueeze(-1)], dim=-1)

        # Log probability tensor that mandates that the end token is selected.
        # shape: (batch_size * beam_size, num_classes)
        logprobs_after_end = start_class_logprobs.new_full(
            (batch_size * self.beam_size, num_classes), float("-inf")
        )
        logprobs_after_end[:, self._eos_index] = 0.0

        logits_after_end = start_class_logprobs.new_full(
            (batch_size * self.beam_size, num_classes), float("-inf")
        )
        logits_after_end[:, self._eos_index] = 0

        while predictions.shape[-1] < self.max_steps:
            # shape: (batch_size * beam_size,)
            last_predictions = predictions[:, :, -1].reshape(batch_size * self.beam_size)

            # If every predicted token from the last step is `self._eos_index`,
            # then we can stop early.
            if (last_predictions == self._eos_index).all():
                break

            predictions_so_far = predictions.view(
                batch_size * self.beam_size, -1
            )
            # shape: (batch_size * beam_size, num_classes)
            class_logits = step(predictions_so_far)

            # Set logprobs of last predicted tokens as high negative value to avoid
            # repetition in description.
            class_logits = class_logits.scatter(1, predictions_so_far[:, -1].view((-1, 1)), -10000)

            # shape: (batch_size * beam_size, num_classes)
            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(
                batch_size * self.beam_size, num_classes
            )

            # Here we are finding any beams where we predicted the end token in
            # the previous timestep and replacing the distribution with a
            # one-hot distribution, forcing the beam to predict the end token
            # this timestep as well.
            class_logits = torch.where(
                last_predictions_expanded == self._eos_index,
                logits_after_end,
                class_logits,
            )

            # Convert logits to logprobs.
            # shape: (batch_size * beam_size, vocab_size)
            class_logprobs = F.log_softmax(class_logits, dim=1)

            # shape (both): (batch_size * beam_size, per_node_beam_size)
            top_logprobs, predicted_classes = class_logprobs.topk(
                self.per_node_beam_size
            )

            # Here we expand the last log probs to `(batch_size * beam_size,
            # per_node_beam_size)` so that we can add them to the current log
            # probs for this timestep. This lets us maintain the log
            # probability of each element on the beam.
            # shape: (batch_size * beam_size, per_node_beam_size)
            expanded_last_logprobs = (
                last_logprobs.unsqueeze(2)
                .expand(batch_size, self.beam_size, self.per_node_beam_size)
                .reshape(batch_size * self.beam_size, self.per_node_beam_size)
            )
            # shape: (batch_size * beam_size, per_node_beam_size)
            summed_top_logprobs = top_logprobs + expanded_last_logprobs

            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_summed = summed_top_logprobs.reshape(
                batch_size, self.beam_size * self.per_node_beam_size
            )
            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_predicted_classes = predicted_classes.reshape(
                batch_size, self.beam_size * self.per_node_beam_size
            )
            # Append the predictions to the current beam.
            reshaped_beam = (
                predictions.view(batch_size * self.beam_size, 1, -1)
                .repeat(1, self.per_node_beam_size, 1)
                .reshape(batch_size, self.beam_size * self.per_node_beam_size, -1)
            )
            # batch_size, (beam_size * per_node_beach_size), #token
            reshaped_beam = torch.cat([reshaped_beam, reshaped_predicted_classes.unsqueeze(-1)], dim=-1)

            # Keep only the top `beam_size` beam indices.
            # shape: (batch_size, beam_size), (batch_size, beam_size)
            restricted_beam_logprobs, restricted_beam_indices = reshaped_summed.topk(
                self.beam_size
            )
            predictions = reshaped_beam.gather(
                1, restricted_beam_indices.unsqueeze(-1).repeat(1,1,reshaped_beam.shape[-1])
            )

            # shape: (batch_size, beam_size)
            last_logprobs = restricted_beam_logprobs

        if not torch.isfinite(last_logprobs).all():
            warnings.warn(
                "Infinite log probs encountered. Some final descriptions may not "
                "make sense. This can happen when the beam size is larger than"
                " the number of valid (non-zero probability) transitions that "
                "the step function produces.",
                RuntimeWarning,
            )

        # Optionally select best beam and its logprobs.
        if only_return_best:
            # shape: (batch_size, sequence_length)
            predictions = predictions[:, 0, :]
            last_logprobs = last_logprobs[:, 0]
        num_valid = (predictions != self._eos_index).sum(dim=-1)
        num_valid += (predictions == self._eos_index).sum(dim=-1) > 0
        num_valid = num_valid - begin_tokens.shape[1]
        num_valid = num_valid.clip(min=1)

        last_logprobs = last_logprobs / num_valid

        return predictions, last_logprobs




