import torch
from torch.nn import functional as F
from typing import Any, Dict
from a3_utils import *

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration
)

class TopKSamplerForT5(GeneratorForT5):
    ###########################################################################
    # NOTE: Caution - do not modify the args to the class + the args of 
    # the sample function.
    # 
    # However, feel free to add as many helper functions in this class as you want.
    ###########################################################################
    def __init__(self, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer):
        super().__init__(model, tokenizer)
    
    def get_top_k(self, logits, temperature, top_k):
        proba = F.softmax(logits/temperature, dim = 0)
        proba, indices = torch.sort(proba, dim =0, descending=True)
        proba, indices = proba.tolist(), indices.tolist()
        return proba[:top_k], indices[:top_k]
    
    def sample(
        self,
        inputs: dict,
        top_k: int,
        temperature: float,
        max_new_tokens: int,
    ) -> torch.LongTensor:
        """Generates sequences of token ids for T5ForConditionalGeneration 
        (which has a language modeling head) using top-k sampling. 
        This means that we sample the next token from the top-k scoring tokens 
        by using their probability values.

        This function always does early stopping and does not handle the case 
        where we don't do early stopping. 
        It also only handles inputs of batch size = 1.
        It also only handles top_k => 1.
        The temperature variable that helps modulate the probability by scaling the logits.
        distribution we sample from by scaling the logits before softmax.

        Inherits variables and helper functions from GeneratorForT5().

        Args:
            inputs (dict): the tokenized input dictionary returned by the T5 tokenizer
            top_k (int): the number of highest probability vocabulary tokens to keep for top-k filtering/sampling
            temperature (float): the value used to modulate the next token probabilities, scales logits before softmax
            max_new_tokens (int): a limit for the amount of decoder outputs 
                                  we desire to generate

        Returns:
            torch.LongTensor: top-k sampled sequence made of token ids of size (1,generated_seq_len)
                              This should include the starting pad token!
        """
        ########################################################################
        # NOTE: Don't change this part, it's to help you debug!
        constraint_return = self.input_constraints(inputs, max_new_tokens, top_k=top_k)
        if constraint_return is None:
            return None
        else:
            max_new_tokens = constraint_return
        ########################################################################

        ########################################################################
        # TODO: Implement me! Read the docstring above and this comment carefully.
        #
        # For top-k sampling, keep in mind of the following:
        #   - do not handle input batch size != 1.
        #   - return the sampled sequence as it is (not in a dictionary).
        #     You should not return a score you get for the sequence.
        #   - always do early stopping: this means that if the next token is an EOS
        #     (end-of-sentence) token, you should stop decoding.
        #   - don't forget to implement the temperature functionality!
        #   - you might want to use the self.prepare_next_inputs function inherited
        #     by this class as shown here:
        #
        #       First token use: 
        #           model_inputs = self.prepare_next_inputs(model_inputs=inputs)
        #       Future use: 
        #           model_inputs = self.prepare_next_inputs(
        #               model_inputs = model_inputs,
        #               new_token_id = new_token_id,
        #           )
        ########################################################################
        model_inputs = inputs.copy()
        
        # Set the first input we want to give to the decoder 
        model_inputs['decoder_input_ids'] = torch.tensor([self.tokenizer.pad_token_id] ).unsqueeze(0)
        model_inputs['decoder_attention_mask'] = torch.tensor([1]).unsqueeze(0)
        
        # Get the output hidden state of the encoder 
        init_output = self.model(**model_inputs)
        encoder_outputs = (init_output.encoder_last_hidden_state,)
        
        #Get the log probs of the output
        logits = init_output.logits[0][-1]
        top_k_proba,  top_k_tokens = self.get_top_k(logits, temperature, top_k)
        new_token_id = np.random.choice(top_k_tokens, 1, top_k_proba)
        output_sentence = [self.tokenizer.pad_token_id, new_token_id.item()]
        
        for i in range(max_new_tokens-1):
            if new_token_id == self.tokenizer.eos_token_id :
                break
            # Get the distibution of the token using the hidden state of the encoded sequence and the already generated tokens
            logits = self.model(None, decoder_input_ids = torch.tensor([output_sentence]) ,encoder_outputs = encoder_outputs)['logits']
            
            #We take the top k Tokens
            top_k_proba,  top_k_tokens = self.get_top_k(logits[0][-1], temperature, top_k)
            #We transform the logits to a distribution
            new_token_id = np.random.choice(top_k_tokens, 1, top_k_proba)
            
            output_sentence.append(new_token_id.item())
        return torch.tensor([output_sentence])
        


class TopPSamplerForT5(GeneratorForT5):
    ###########################################################################
    # NOTE: Caution - do not modify the args to the class + the args of 
    # the sample function.
    # 
    # However, feel free to add as many helper functions in this class as you want.
    ###########################################################################
    def __init__(self, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer):
        super().__init__(model, tokenizer)
    
    def get_top_p(self, logits, temperature, top_p):
        proba = F.softmax(logits/temperature, dim = 0)
        proba, indices = torch.sort(proba, dim =0, descending=True)
        proba, indices = proba.tolist(), indices.tolist()
        sum_p, i = 0, 0
        while sum_p < top_p and i < len(proba):
            sum_p += proba[i]+1
            i +=1
        return proba[:i], indices[:i]
        
    def sample(
        self,
        inputs: dict,
        top_p: float,
        temperature: float,
        max_new_tokens: int
    ) -> torch.LongTensor:
        """Generates sequences of token ids for T5ForConditionalGeneration 
        (which has a language modeling head) using top-p sampling. 
        This means that we sample the next token from the smallest set of most 
        probable tokens with probabilities that cumulatively add up to top_p or higher.

        This function always does early stopping and does not handle the case 
        where we don't do early stopping. 
        It also only handles inputs of batch size = 1.
        If there are no tokens falling in the top_p cumulative probability mass 
        (e.g. because the top scoring tokens probability is larger than top_p) then sample the top scoring token.
        The temperature variable that helps modulate the probability by scaling the logits.
        distribution we sample from by scaling the logits before softmax.

        Inherits variables and helper functions from GeneratorForT5().

        Args:
            inputs (dict): the tokenized input dictionary returned by the T5 tokenizer
            top_p (float): the cumulative probability mass to select the smallest 
                           set of most probable tokens with probabilities that 
                           cumulatively add up to top_p or higher.
            temperature (float): the value used to modulate the next token probabilities, scales logits before softmax
            max_new_tokens (int): a limit for the amount of decoder outputs 
                                  we desire to generate

        Returns:
            torch.LongTensor: top-p sampled sequence made of token ids of size (1,generated_seq_len)
                              This should include the starting pad token!
        """
        ########################################################################
        # NOTE: Don't change this part, it's to help you debug!
        constraint_return = self.input_constraints(inputs, max_new_tokens)
        if constraint_return is None:
            return None
        else:
            max_new_tokens = constraint_return
        ########################################################################

        ########################################################################
        # TODO: Implement me! Read the docstring above and this comment carefully.
        #
        # For top-p sampling, keep in mind of the following:
        #   - do not handle input batch size != 1.
        #   - return the sampled sequence as it is (not in a dictionary).
        #     You should not return a score you get for the sequence.
        #   - always do early stopping: this means that if the next token is an EOS
        #     (end-of-sentence) token, you should stop decoding.
        #   - don't forget to handle the edge case when top scoring tokens probability > top_p,
        #     sample that token only.
        #   - don't forget to implement the temperature functionality!
        #   - you might want to use the self.prepare_next_inputs function inherited
        #     by this class as shown here:
        #
        #       First token use: 
        #           model_inputs = self.prepare_next_inputs(model_inputs=inputs)
        #       Future use: 
        #           model_inputs = self.prepare_next_inputs(
        #               model_inputs = model_inputs,
        #               new_token_id = new_token_id,
        #           )
        ########################################################################
        model_inputs = inputs.copy()
        
        # Set the first input we want to give to the decoder 
        model_inputs['decoder_input_ids'] = torch.tensor([self.tokenizer.pad_token_id] ).unsqueeze(0)
        model_inputs['decoder_attention_mask'] = torch.tensor([1]).unsqueeze(0)
        
        # Get the output hidden state of the encoder 
        init_output = self.model(**model_inputs)
        encoder_outputs = (init_output.encoder_last_hidden_state,)
        
        #Get the log probs of the output
        logits = init_output.logits[0][-1]
        top_p_proba, top_p_tokens = self.get_top_p(logits, temperature, top_p)
        
        new_token_id = np.random.choice(top_p_tokens, 1, top_p_proba)
        output_sentence = [self.tokenizer.pad_token_id, new_token_id.item()]
        
        for i in range(max_new_tokens-1):
            if new_token_id == self.tokenizer.eos_token_id :
                break
            # Get the distibution of the token using the hidden state of the encoded sequence and the already generated tokens
            logits = self.model(None, decoder_input_ids = torch.tensor([output_sentence]) ,encoder_outputs = encoder_outputs)['logits']
            
            #We take the top p Tokens
            top_p_proba, top_p_tokens = self.get_top_p(logits[0][-1], temperature, top_p)
            
            #We transform the logits to a distribution
            new_token_id = np.random.choice(top_p_tokens, 1, top_p_proba)
            
            output_sentence.append(new_token_id.item())
            
        return torch.tensor([output_sentence])

def main():
    ############################################################################
    # NOTE: You can use this space for testing but you are not required to do so!
    ############################################################################
    seed = 421
    torch.manual_seed(seed)
    torch.set_printoptions(precision=16)
    model_name = "t5-small"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)


if __name__ == '__main__':
    main()