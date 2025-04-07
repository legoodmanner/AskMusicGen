import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

import torchaudio


def get_gen_model(config):
    modelConfig = config.model.gen_model
    model_dict = {
        'MusicGenSmall': MusicGenModule,
        'MusicGenMedium': MusicGenModule,
        'MusicGenLarge': MusicGenModule,
        'VampNetCoarse': VampNetModule,
        'VampNetC2F': VampNetModule,
        'MFCC': MFCCModule,
        'Mel': MelModule,
        'StableAudioOpen': DiffModule,
    }

    return model_dict[modelConfig.name](config=config, **modelConfig)
    

    


class MusicGenModule(torch.nn.Module):
    def __init__(self, config=None, extract_layer=-1, version='small', **kwargs) -> None:
        super().__init__()
        # self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        self.config = config
        self.model = MusicgenForConditionalGeneration.from_pretrained(f"facebook/musicgen-{version}")
        self.processor = AutoProcessor.from_pretrained(f"facebook/musicgen-{version}")
        self.aggregation = config.model.gen_model.get('aggregation', True)
        self.layer = extract_layer
        self.model.generation_config.max_length = 3080
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, wav):
        """
        wav: shape [batch, channel, seq_len] or [ channel, seq_len]

        """
        # if len(wav.shape) ==1:
        #     wav = wav[None, None, :]

        # if len(wav.shape) ==2:
        #     assert wav.shape[0] == 1
        #     wav = wav[None, :]

        # output = self.model.audio_encoder(
        #     input_values=wav,
        #     padding_mask=torch.ones_like(wav).to(device=wav.device),
        # )
        # audio_codes = output['audio_codes']
        # frames, bsz, codebooks, seq_len = audio_codes.shape 
        # assert frames == 1

        # inputs = self.model.get_unconditional_inputs(bsz)
        # inputs['decoder_input_ids'] = audio_codes[0, ...].reshape(bsz * codebooks, seq_len)
        # # inputs['decoder_input_ids'] = self.model.prepare_inputs_for_generation(**inputs)['decoder_input_ids']
        # decoder_outputs = self.model(
        #             **inputs,
        #             output_hidden_states=True,
        #             return_dict=True,
        # )
        wav = wav.detach().cpu()
        wav = [w.squeeze().numpy() for w in wav]
        inputs = self.processor(
            audio=wav,
            text=[""]*len(wav),
            sampling_rate=self.config.data.sample_rate,
            padding=True,
            return_tensors="pt",
        )
        for k in inputs:
            try:
                inputs[k] = inputs[k].cuda()
            except:
                continue
        # extract representations from decoder LM
        decoder_outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = decoder_outputs.decoder_hidden_states


        if self.layer is not None:
            return self.aggregate(hidden_states[self.layer], agg_type=self.config.model.gen_model.get('agg_type', 'mean'))
        else:
            # When self.layer is None, we either stack the aggregated results or return the original tuple.
            return torch.stack(
                [self.aggregate(h, agg_type=self.config.model.gen_model.get('agg_type', 'mean')) for h in hidden_states]
            ) if self.aggregation else hidden_states
        
    def aggregate(self, activations, agg_type='mean', **kwargs):
        """
        hidden_state: torch.Size([bs, seq_len, 1280])
        """
        if agg_type == 'mean':
            return activations.mean(dim=-2)
        
        elif agg_type == 'downsample':
            ratio = kwargs.get('ratio', 8)
            # interpolate
            activations = activations.transpose(1, 2) # [bs, 1280, seq_len]
            activations =  torch.nn.functional.interpolate(activations, scale_factor=1/ratio, mode='area')
            return activations.transpose(1, 2) # [bs, seq_len, 1280]

        elif agg_type == 'k-means':
            bs, seq_len, feature_dim = activations.shape
            aggregated = []  # This will collect the aggregated representation per sample.

            # Process each sample in the batch independently.
            for i in range(bs):
                # Get the activations for the i-th sample: shape [seq_len, feature_dim]
                x = activations[i]

                # Randomly initialize centroids from the time sequence tokens.
                init_indices = torch.randperm(seq_len)[:kwargs.get('n_clusters', 4)]
                centroids = x[init_indices]  # shape: [n_clusters, feature_dim]

                # Run the k-means algorithm for a fixed number of iterations.
                for _ in range(kwargs.get('n_iter', 50)):
                    # Compute squared Euclidean distances between tokens and centroids.
                    # x.unsqueeze(1): [seq_len, 1, feature_dim]
                    # centroids.unsqueeze(0): [1, n_clusters, feature_dim]
                    distances = ((x.unsqueeze(1) - centroids.unsqueeze(0)) ** 2).sum(dim=-1)  # [seq_len, n_clusters]
                    
                    # Assign each token (time step) to the closest centroid.
                    labels = distances.argmin(dim=1)  # shape: [seq_len]

                    # Update centroids based on the mean of assigned tokens.
                    new_centroids = []
                    for cluster in range(kwargs.get('n_clusters', 4)):
                        cluster_mask = (labels == cluster)
                        if cluster_mask.any():
                            new_centroid = x[cluster_mask].mean(dim=0)
                        else:
                            # If no tokens are assigned to this cluster, keep the old centroid.
                            new_centroid = centroids[cluster]
                        new_centroids.append(new_centroid)
                    centroids = torch.stack(new_centroids)

                
                # 
                aggregated_representation = centroids.mean(dim=0)  # shape: [feature_dim]
                aggregated.append(aggregated_representation)


            return torch.stack(aggregated)  # shape: [bs, feature_dim]
        
        else:
            raise ValueError(f"Unknown aggregation type: {agg_type}")
    
    # return un-aggregated representations
    def predict_step(self, wav, batch_idx):
        with torch.no_grad():
            wav = wav.detach().cpu()
            wav = [w.squeeze().numpy() for w in wav]
            inputs = self.processor(
                audio=wav,
                text=[""]*len(wav),
                sampling_rate=self.config.data.sample_rate,
                padding=True,
                return_tensors="pt",
            )
            for k in inputs:
                try:
                    inputs[k] = inputs[k].cuda()
                except:
                    continue
            # extract representations from decoder LM
            decoder_outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = decoder_outputs.decoder_hidden_states
        return hidden_states

            

    

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
            
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_uncond

import k_diffusion as K
from stable_audio_tools.inference.generation import prepare_audio

class DiffModule(torch.nn.Module):
    def __init__(self, config=None, extract_time_step=None, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.time_step = extract_time_step # can be scalar or list
        self.requires_grad_(False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        self.model.to(self.device)
        self.sample_rate = self.model_config["sample_rate"]
        self.hidden_states = []
    
    def callback(self, model_output):
        # ({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        # save for every 50 steps
        if model_output['i'] % self.config.model.gen_model.get('save_every', 50) == 0:
            self.hidden_states.append(
                (model_output['denoised'] - model_output['x']).transpose(-1, -2)
            )

    @torch.no_grad()
    def forward(self, wav):
        if len(wav.shape) == 3:
            wav = wav.squeeze(1)
        if isinstance(self.time_step, list):
            representation = []
            for t in self.time_step:
                repr = self.extract_representation(
                    (self.sample_rate, wav), 
                    timestep=t, 
                    device=self.device, 
                    **self.config.model.gen_model
                )
                representation.append()
        else:
            representation = self.extract_representation(
                (self.sample_rate, wav), 
                timestep=self.time_step,
                device=self.device, 
                **self.config.model.gen_model
            )

        return representation.unsqueeze(0)
    
    def extract_representation(
            self,
            input_audio,
            timestep: int = 100,  # specify which timestep to use
            device: str = "cuda",
            **sampler_kwargs
            ) -> torch.Tensor:
        
        # Get input audio and sample rate
        in_sr, audio = input_audio
        
        # Prepare the audio and convert to latent if using latent diffusion
        io_channels = self.model.pretransform.io_channels if self.model.pretransform is not None else self.model.io_channels
        
        # Prepare audio to correct format
        processed_audio = prepare_audio(
            audio, 
            in_sr=in_sr, 
            target_sr=self.model.sample_rate, 
            target_channels=io_channels, 
            device=device,
            target_length=self.config.data.max_length
        )

        # Convert to latent if using latent diffusion
        if self.model.pretransform is not None:
            z = self.model.pretransform.encode(processed_audio)
        else:
            z = processed_audio

        # Calculate noise level for the specified timestep
        # You'll need to adjust this based on your noise schedule
        t = torch.tensor([timestep], device=device)
        noise_level = self.get_noise_level(t, **sampler_kwargs)  # You'll need to implement this based on your scheduler
        
        # Add noise to the latent
        noise = torch.randn_like(z)
        noisy_latent = z + noise_level * noise

        # Run single diffusion step
        if self.model.diffusion_objective == "v":
            # Adapt your k-diffusion sampler for single step
            representation = self.single_step_k(
                x=noisy_latent,
                timestep=timestep,
                device=device,
                **sampler_kwargs
            )
        elif self.model.diffusion_objective == "rectified_flow":
            # Adapt your RF sampler for single step
            representation = self.single_step_rf(noisy_latent, t, **sampler_kwargs)
        representation = representation.transpose(-1, -2)

        #TODO aggregation
        representation = representation.mean(dim=-2)
        return representation

    def get_noise_level(self, timestep: torch.Tensor, sigma_min=0.01, sigma_max=100, steps=100, device="cuda", **extra_args):
        """
        Convert timestep to noise level (sigma)
        Args:
            timestep: Current timestep (between 0 and steps-1)
            sigma_min: Minimum sigma value
            sigma_max: Maximum sigma value
            steps: Total number of steps
            device: Device to put tensor on
        Returns:
            sigma: Noise level for the timestep
        """
        # Using polyexponential sigma scheduling from k-diffusion
        sigmas = K.sampling.get_sigmas_polyexponential(
            steps, 
            sigma_min, 
            sigma_max, 
            rho=1.0, 
            device=device
        )
        return sigmas[timestep]

    def single_step_k(self, x, timestep, sigma_min=0.01, sigma_max=100, steps=250, device="cuda", **extra_args):
        """
        Run a single step of k-diffusion and return predicted noise
        """
        denoiser = K.external.VDenoiser(self.model.model)
        sigma = self.get_noise_level(timestep, sigma_min, sigma_max, steps, device)
        s_in = torch.ones([x.shape[0]], device=x.device)
        
        # Get denoised output
        denoised = denoiser(x, sigma * s_in, )
        
        # Calculate predicted noise (following DDPM formulation)
        pred_noise = (x - denoised) / sigma

        return pred_noise # or return both: return pred_noise, denoised


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

    

if __name__ == "__main__":
    from omegaconf import OmegaConf
    test_config = OmegaConf.create(
        dict(data = {
            "sample_rate": 16000,
            "batch_size": 1,
            "max_length": 32000,
        },
        model = {
            "gen_model": {
                "name": "StableAudioOpen",
                "extract_layer": 1,
            }
        }
        )
    )
    model = DiffModule(config=test_config, extract_time_step=50)
    print(model.model.diffusion_objective)
    wav = torch.randn(1, 16000)
    output = model(wav)
    breakpoint()