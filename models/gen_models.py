import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

import torchaudio


def get_gen_model(config):
    modelConfig = config.model.gen_model
    model_dict = {
        'MusicGenSmall': MusicGenModule,
        'MusicGenMedium': MusicGenModule,
        'VampNetCoarse': VampNetModule,
        'VampNetC2F': VampNetModule,
        'MFCC': MFCCModule,
        'Mel': MelModule,
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
from vampnet import mask as pmask 

class VampNetModule(torch.nn.Module):
    def __init__(self, config=None, extract_layer=-1, version='coarse', **kwargs) -> None:
        super().__init__()
        self.config = config
        self.model = VampNetInterface(
            coarse_ckpt="./cache/models/vampnet/coarse.pth", 
            coarse2fine_ckpt="./cache/models/vampnet/c2f.pth", 
            codec_ckpt="./cache/models/vampnet/codec.pth",
            device="cuda", 
            wavebeat_ckpt=None,
        )

        self.layer = extract_layer
        self.version = version
        print(f"VampNetModule: Extracting from {version} version")
        self.requires_grad_(False)
        self.r = kwargs.get('r', 0.7)

    # TODO: fine2coarse still not implemented
    @torch.no_grad()
    def forward(self, wav):
        z = self.model.codec.encode(wav, self.config.data.sample_rate)['codes']
        if self.version == 'coarse':
            z = z[:, : self.model.coarse.n_codebooks, :].clone()

            # apply mask
            mask = pmask.random(z, self.r) # shape [bs, n_codebooks, seq_len]
            mask = pmask.codebook_unmask(mask, self.model.coarse.n_conditioning_codebooks) # shape [bs, n_codebooks, seq_len]
            z_mask, mask = pmask.apply_mask(z, mask, self.model.coarse.mask_token) 

            latent = self.model.coarse.embedding.from_codes(z, self.model.codec)
            _, activations = self.model.coarse.forward(latent, return_activations=True) # activations: [torch.Size([bs, seq_len, 1280])] * layer_number
        elif self.version == 'c2f':
            z = z[:, : self.model.c2f.n_codebooks, :].clone()
            # apply mask
            mask = pmask.random(z, self.r)
            mask = pmask.codebook_unmask(mask, self.model.coarse.n_conditioning_codebooks)
            z_mask, mask = pmask.apply_mask(z, mask, self.model.coarse.mask_token)

            latent = self.model.c2f.embedding.from_codes(z_mask, self.model.codec)
            _, activations = self.model.c2f.forward(latent, return_activations=True) # activations: [torch.Size([bs, seq_len, 1280])] * layer_number
        
        # gather activations that are masked (mask == 1)
        # TODO: This method is only feasible for discriminative tasks
        # activations = activations[:][:]
        if self.layer is not None:
            return activations[self.layer]
        else:
            return activations    


class MFCCModule(torch.nn.Module):
    def __init__(self, config=None, extract_layer=-1, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.layer = extract_layer
        self.requires_grad_(False)
        self.func = torchaudio.transforms.MFCC(
            sample_rate=self.config.model.gen_model.sample_rate,
            n_mfcc=self.config.model.gen_model.n_mfcc,
            melkwargs=self.config.model.gen_model.melkwargs,
        )

    @torch.no_grad()
    def forward(self, wav):
        mfcc = self.func(wav) # shape [batch, channel, seq_len]
        print(mfcc.shape)

        return mfcc
    

class MelModule(torch.nn.Module):
    def __init__(self, config=None, extract_layer=-1, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.layer = extract_layer
        self.requires_grad_(False)
        self.func = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.data.sample_rate,
            n_fft=self.config.model.gen_model.n_fft,
            win_length=self.config.model.gen_model.win_length,
            hop_length=self.config.model.gen_model.hop_length,
            n_mels=self.config.model.gen_model.n_mels,
        )
    @torch.no_grad()
    def forward(self, wav):
       
        mel = self.func(wav) # shape [batch, channel, n_mel, seq_len]
        mel = mel.transpose(-1, -2).squeeze(1) # shape [batch, seq_len,  n_mel]
        if self.layer is not None:
            return mel
        else:
            return (mel,) # return as tuple for compatibility

    
 
