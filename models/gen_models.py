import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

import torchaudio


def get_gen_model(config):
    modelConfig = config.model.gen_model
    model_dict = {
        'MusicGenSmall': MusicGenModule,
        'MusicGenMedium': MusicGenModule,
    }

    return model_dict[modelConfig.name](config=config, **modelConfig)
    

class MusicGenModule(torch.nn.Module):
    def __init__(self, config=None, extract_layer=-1, version='small', **kwargs) -> None:
        super().__init__()
        # self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        self.config = config
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


from vampnet.interface import Interface as VampNetInterface

class VampNetModule(torch.nn.Module):
    def __init__(self, config=None, extract_layer=-1, version='coarse', **kwargs) -> None:
        super().__init__()
        self.config = config
        self.model = VampNetInterface(
            coarse_ckpt="./cache/vampnet/coarse.pth", 
            coarse2fine_ckpt="./cache/vampnet/c2f.pth", 
            codec_ckpt="./cache/vampnet/codec.pth",
            device="cuda", 
            wavebeat_ckpt=None,
        )
        self.layer = extract_layer
        self.requires_grad_(False)

    # TODO: fine2coarse still not implemented
    @torch.no_grad()
    def forward(self, wav):
        z = self.model.codec.encode(wav, self.config.data.sample_rate)['z']
        z = z[:, : self.model.coarse.n_codebooks, :].clone()
        # no mask
        latent = self.model.coarse.embedding.from_codes(z, self.model.codec)
        _, activations = self.model.coarse.forward(latent, return_activations=True) # activations: [torch.Size([bs, seq_len, 1024])] * layer_number
        # extract activation from every/assigned layer
 
        return activations[self.layer]
    


    
 
