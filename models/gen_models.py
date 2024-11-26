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
        self.aggregation = config.model.gen_model.get('aggregation', True)
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
            if self.aggregation:
                return decoder_outputs.decoder_hidden_states[self.layer].mean(-2)  # torch.Size([bs, seq_len, 1024]) -> torch.Size([bs, 1024])
            return decoder_outputs.decoder_hidden_states[self.layer]
        else:
            if self.aggregation:
                return torch.stack([decoder_outputs.decoder_hidden_states[i].mean(-2) for i in range(len(decoder_outputs.decoder_hidden_states))]) # torch.Size([layer_number, bs, 1024])
            return decoder_outputs.decoder_hidden_states # tuple( torch.Size([bs, seq_len, 1024]) * layer_number )_states # tuple( torch.Size([bs, seq_len, 1024]) * layer_number )


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
        self.aggregation = config.model.gen_model.get('aggregation', True)
        self.layer = extract_layer
        self.version = version
        print(f"VampNetModule: Extracting from {version} version")
        self.requires_grad_(False)
        self.r = kwargs.get('r', 0.95)

    # TODO: fine2coarse still not implemented
    @torch.no_grad()
    def forward(self, wav):
        z = self.model.codec.encode(wav, self.config.data.sample_rate)['codes']
        seq_len = z.shape[-1]
        if self.version == 'coarse':
            z = z[:, : self.model.coarse.n_codebooks, :].clone()

            # apply mask
            
            mask = pmask.random(z, self.r)
            mask[:, 1:, :] = 0
            z_mask, mask = pmask.apply_mask(z, mask, self.model.coarse.mask_token) 
            latent = self.model.coarse.embedding.from_codes(z_mask, self.model.codec)
            _, activations = self.model.coarse.forward(latent, return_activations=True) # activations: [torch.Size([bs, seq_len, 1280])] * layer_number
        elif self.version == 'c2f':
            z = z[:, : self.model.c2f.n_codebooks, :].clone()
            # apply mask
            mask = pmask.inpaint(
                z, 
                n_prefix=int(self.r*0.5*seq_len),
                n_suffix=int(self.r*0.5*seq_len)
            ) # shape [bs, n_codebooks, seq_len]
            mask = pmask.codebook_unmask(mask, self.model.c2f.n_conditioning_codebooks)
            z_mask, mask = pmask.apply_mask(z, mask, self.model.c2f.mask_token)
            latent = self.model.c2f.embedding.from_codes(z_mask, self.model.codec)
            _, activations = self.model.c2f.forward(latent, return_activations=True) # activations: [torch.Size([bs, seq_len, 1280])] * layer_number
        if self.layer is not None:
            if self.aggregation:
                return self.aggregate(activations[self.layer], mask, agg_type='all')
            else:
                return activations[self.layer]
        else:
            if self.aggregation:
                return self.aggregate(activations, mask, agg_type='all')
            else:
                return activations
    
    def aggregate(self, activations, mask=None, agg_type='all'):
        """
        activations: torch.Size([(layer), bs, seq_len, 1280])
        mask: torch.Size([bs, n_codebooks, seq_len]) 
        """
        # gather activations that are masked (mask == 1)
        # TODO: This method is only feasible for discriminative tasks
        if agg_type == 'mask_only':
            mask = mask[:,-1,:, None] # take only last codebook -> shape [bs, seq_len, 1]
            if len(activations.shape) == 4: # multiple layers
                mask = mask[None,:]

            masked_activations = activations * mask  # Shape: ((layer), bs, seq_len, vector_dim)
            sum_vectors = masked_activations.sum(dim=-2)  # Shape: ((layer), bs, vector_dim)
            
            # Count number of 1s in mask for each batch
            mask_sum = mask.sum(dim=-2)  # Shape: (bs, 1)
            # Compute mean (avoiding division by zero)
            mean_vectors = sum_vectors / (mask_sum + 1e-4)  # Shape: ((layer), bs, vector_dim)
        else:
            mean_vectors = activations.mean(dim=-2)  # Shape: ((layer), bs, vector_dim)
        return mean_vectors
            
           


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

    
 
