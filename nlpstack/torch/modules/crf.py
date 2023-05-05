from typing import List, Optional, Tuple, cast

import torch

ViterbiDecoding = Tuple[List[int], float]


class ConditionalRandomField(torch.nn.Module):
    def __init__(self, num_tags: int) -> None:
        super(ConditionalRandomField, self).__init__()
        self.num_tags = num_tags
        self.transitions = torch.nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = torch.nn.Parameter(torch.randn(num_tags))
        self.stop_transitions = torch.nn.Parameter(torch.randn(num_tags))

    def forward(
        self,
        inputs: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = cast(torch.BoolTensor, torch.ones_like(tags, dtype=torch.bool))

        batch_size, sequence_length = tags.shape

        log_denominator = self._compute_log_partition_function(inputs, mask)
        log_numerator = self._compute_log_score(inputs, tags, mask)

        return log_numerator - log_denominator

    def _compute_log_partition_function(self, inputs: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        batch_size, sequence_length, _ = inputs.shape

        alpha = inputs.new_zeros(batch_size, self.num_tags)
        alpha += self.start_transitions

        for t in range(sequence_length):
            alpha_t = alpha.unsqueeze(1) + inputs[:, t] + self.transitions
            alpha = torch.logsumexp(alpha_t, dim=-1)
            alpha = torch.where(mask[:, t].unsqueeze(-1), alpha, alpha_t.squeeze(1))

        alpha += self.stop_transitions
        return torch.logsumexp(alpha, dim=-1)

    def _compute_log_score(
        self,
        inputs: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = inputs.shape

        score = inputs.new_zeros(batch_size)
        score += self.start_transitions[tags[:, 0]]

        for t in range(sequence_length - 1):
            transition_scores = self.transitions[tags[:, t], tags[:, t + 1]]
            emission_scores = inputs[:, t, tags[:, t]]
            score += transition_scores * mask[:, t + 1] + emission_scores * mask[:, t]

        last_tag_mask = mask.new_zeros(batch_size, self.num_tags).scatter_(1, tags[:, -1].unsqueeze(-1), 1)
        last_tag_mask = last_tag_mask * mask[:, -1].unsqueeze(-1)
        score += torch.sum(self.stop_transitions * last_tag_mask, dim=-1)
        score += torch.sum(inputs[:, -1] * last_tag_mask, dim=-1)

        return score

    def viterbi_decode(
        self,
        logits: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        topk: int = 1,
    ) -> List[List[ViterbiDecoding]]:
        if mask is None:
            mask = cast(torch.BoolTensor, torch.ones(logits.shape[:2], dtype=torch.bool, device=logits.device))

        batch_size, sequence_length, _ = logits.shape

        scores = logits.new_zeros(batch_size, self.num_tags)
        scores += self.start_transitions

        pointers = []

        for t in range(sequence_length):
            scores_t = scores.unsqueeze(1) + logits[:, t] + self.transitions
            max_scores, max_indices = torch.topk(scores_t.view(batch_size, -1), topk, dim=-1)
            scores = max_scores.view(batch_size, -1)
            scores = torch.where(mask[:, t].unsqueeze(-1), scores, scores_t.view(batch_size, -1))
            pointers.append(max_indices)

        scores += self.stop_transitions
        max_scores, max_indices = torch.topk(scores.view(batch_size, -1), topk, dim=-1)

        best_paths = []
        for b in range(batch_size):
            paths = []
            for k in range(topk):
                path = []
                prev_tag = max_indices[b, k]
                for t in range(sequence_length - 1, 0, -1):
                    prev_tag = pointers[t][b, prev_tag // self.num_tags]
                    path.append(prev_tag % self.num_tags)
                path.reverse()
                paths.append((path, max_scores[b, k].item()))
            best_paths.append(paths)

        return best_paths
