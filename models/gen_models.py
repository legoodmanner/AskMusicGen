import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

import torchaudio


def get_gen_model(config):
    modelConfig = config.model.gen_model
    model_dict = {
        'MusicGenSmall': MusicGenModule,
        'MusicGenMedium': MusicGenModule,
    }

    return model_dict[modelConfig.name](**modelConfig)
    

class MusicGenModule(torch.nn.Module):
    def __init__(self, extract_layer=-1, version='small', **kwargs) -> None:
        super().__init__()
        # self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        self.model = MusicgenForConditionalGeneration.from_pretrained(f"facebook/musicgen-{version}")
        self.layer = extract_layer
        self.model.generation_config.max_length = 3080
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, wav):
        """
        wav: shape [batch, channel, seq_len] or [ channel, seq_len]

        """
        if len(wav.shape) ==1:
            wav = wav[None, None, :]

        if len(wav.shape) ==2:
            assert wav.shape[0] == 1
            wav = wav[None, :]

        output = self.model.audio_encoder(
            input_values=wav,
            padding_mask=torch.ones_like(wav).to(device=wav.device),
        )
        audio_codes = output['audio_codes']
        frames, bsz, codebooks, seq_len = audio_codes.shape 
        assert frames == 1

        inputs = self.model.get_unconditional_inputs(bsz)
        inputs['decoder_input_ids'] = audio_codes[0, ...].reshape(bsz * codebooks, seq_len)
        # inputs['decoder_input_ids'] = self.model.prepare_inputs_for_generation(**inputs)['decoder_input_ids']
        decoder_outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
        )
        if self.layer is not None:
            return decoder_outputs.decoder_hidden_states[self.layer] # torch.Size([bs, seq_len, 1024])
        else:
            return decoder_outputs.decoder_hidden_states # tuple( torch.Size([bs, seq_len, 1024]) * layer_number )

    
        #return Seq2SeqLMOutput(
        #     loss=loss,
        #     logits=decoder_outputs.logits,
        #     past_key_values=decoder_outputs.past_key_values,
        #     decoder_hidden_states=decoder_outputs.hidden_states,
        #     decoder_attentions=decoder_outputs.attentions,
        #     cross_attentions=decoder_outputs.cross_attentions,
        #     encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        #     encoder_hidden_states=encoder_outputs.hidden_states,
        #     encoder_attentions=encoder_outputs.attentions,
        # )
    
 
