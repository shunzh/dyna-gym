from dyna_gym.default_policy.default_policy import DefaultPolicy

import gym
import torch
from transformers import PreTrainedModel


class HuggingFaceDefaultPolicy(DefaultPolicy):
    """
    Default policy that uses a HuggingFace transformer model.
    """
    def __init__(
            self,
            env: gym.Env,
            horizon: int,
            model: PreTrainedModel,
            generation_args: dict = {},
    ):
        super().__init__(env, horizon)
        self.model = model
        self.generate_args = generation_args

    @torch.no_grad()
    def get_predicted_sequence(self, state, horizon=None):
        ids, attention_mask = state

        horizon = horizon if horizon is not None else self.horizon

        # Create a batch dimension
        input_data = ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

        outputs = self.model.generate(
            inputs=input_data,
            attention_mask=attention_mask,
            max_length=horizon,
            early_stopping=True,
            return_dict_in_generate=True,
            use_cache=True,
            **self.generate_args
        )

        sequence = outputs.sequences.squeeze(0)
        attention_mask = attention_mask.squeeze(0)

        # update attention mask
        num_new_tokens = sequence.shape[-1] - input_data.shape[-1]
        attention_mask = torch.cat([attention_mask, torch.ones(num_new_tokens).to(attention_mask.device)])

        return sequence, attention_mask

    @torch.no_grad()
    def get_top_k_tokens(self, state):
        k = self.generate_args['top_k']
        p = self.generate_args['top_p']

        ids, attention_mask = state

        # Create a batch dimension
        input_data = ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

        outputs = self.model(
            input_ids=input_data,
            attention_mask=attention_mask,
        )

        # Assuming the model returns logits for tokens
        logits = outputs.logits[0][-1]  # First (and only) batch, last token

        # Convert logits to probabilities
        all_probs = torch.softmax(logits, dim=-1)

        # Get the top k probabilities and their indices, already sorted
        topk_probs, topk_indices = torch.topk(all_probs, k, sorted=True)

        # Compute the cumulative sum of the sorted probabilities
        cumsum_probs = torch.cumsum(topk_probs, dim=-1)

        # Find tokens where the cumulative sum exceeds p
        exceed_p_mask = cumsum_probs > p

        # Find the smallest set of tokens whose cumulative probability exceeds p
        mask = exceed_p_mask.cumsum(dim=-1) <= 1

        # Apply the mask to get final tokens and their probabilities
        final_indices = topk_indices[mask].tolist()
        final_probs = topk_probs[mask].tolist()

        return final_indices, final_probs
