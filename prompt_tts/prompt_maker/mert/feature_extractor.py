import os
import torch
import torch.nn as nn
import torchaudio
import hydra
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor
from .model import MusicHubertModel
from .configuration import MusicHubertConfig
from .audio_utils import *

class MERTFeature(nn.Module):
    def __init__(self, pre_trained_folder, sample_rate, force_half=False,
                 disable_backprop=True, processor_normalize=True,):
        super().__init__()
        self.sample_rate = sample_rate
        self.processor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=sample_rate,
            padding_value=0.0,
            return_attention_mask=True,
            do_normalize=processor_normalize,
        )

        if pre_trained_folder == 'random_model' or None:
            print('Using random HuBERT model.')
            self.model = MusicHubertModel(MusicHubertConfig())
        else:
            print(f'Loading HuBERT model from {pre_trained_folder}')
            self.model = MusicHubertModel.from_pretrained(pre_trained_folder)
            
        self.force_half = force_half
        if disable_backprop:
            self.model.eval()
            if self.force_half:
                self.model.half()

            for param in self.model.parameters():
                param.requires_grad = False
    
    @torch.no_grad()
    def process_wav(self, waveform):
        return self.processor(waveform, return_tensors='pt', sampling_rate=self.sample_rate, 
                              padding=True).input_values[0]
        
    def forward(self, input_values, layer=-1, reduction='mean'):
        if not self.force_half:
            out = self.model(input_values, output_hidden_states=True).hidden_states
        else:
            out = self.model(input_values.half(), output_hidden_states=True).hidden_states
            out = [o.float() for o in out]
        
        if layer != None:
            out = out[layer]
        else: 
            out = torch.stack(out)
        if reduction == 'mean':
            return out.mean(-2)
        elif reduction == 'max':
            return out.max(-2)[0]
        elif reduction == 'none':
            return out
        else:
            raise NotImplementedError
        
    def sliding_window_forward(self, input_values, window_size_in_sample, 
                               stride_in_sample, layer=-1, reduction='mean',
                               allow_non_full_window=True):
        B, T = input_values.shape
        out = []
        for i in range(0, T-window_size_in_sample+1, stride_in_sample):
            out.append(self.forward(input_values[:, i:i+window_size_in_sample], layer=layer, reduction=reduction))
        if allow_non_full_window and T % stride_in_sample != 0:
            out.append(self.forward(input_values[:, -window_size_in_sample:], layer=layer, reduction=reduction))
        return torch.stack(out)

def main(cfg, audio_dir):
    audio_files = find_audios(audio_dir)
    feature_extractor = MERTFeature(cfg.pre_trained_folder if cfg.pre_trained_folder else cfg.huggingface_model_name,
                                    cfg.target_sr, cfg.force_half, processor_normalize=cfg.processor_normalize)
    feature_extractor.to(cfg.device)
    feature_extractor.eval()
    with torch.no_grad():
        for audio_file in tqdm(audio_files):
            try:
                waveform = load_audio(audio_file, target_sr=cfg.target_sr,
                                      is_mono=cfg.is_mono, is_normalize=cfg.is_normalize,
                                      crop_to_length_in_sec=cfg.crop_to_length_in_sec,
                                      crop_randomly=cfg.crop_randomly, device=cfg.device)
            except Exception as e:
                print(f'skipt audio {audio_file} because of {e}')
                continue
            
            wav = feature_extractor.process_wav(waveform)
            wav = wav.to(cfg.device)
            if cfg.sliding_window_size_in_sec:
                assert cfg.sliding_window_size_in_sec > 0, 'sliding_window_size_in_sec must be positive'
                overlap_in_sec = cfg.sliding_window_size_in_sec * cfg.sliding_window_overlap_in_percent / 100
                wavs = []
                for i in range(0, wav.shape[-1], int(cfg.target_sr * (cfg.sliding_window_size_in_sec - overlap_in_sec))):
                    wavs.append(wav[:, i : i + int(cfg.target_sr * cfg.sliding_window_size_in_sec)])
                # discard the last chunk if it is shorter than 1s
                if wavs[-1].shape[-1] < cfg.target_sr * 1:
                    wavs = wavs[:-1]
                features = []
                for wav in wavs:
                    features.append(feature_extractor(wav, layer=cfg.layer, reduction=cfg.reduction))
                features = torch.cat(features, dim=1) #.mean(dim=1, keepdim=True)
            else:
                features = feature_extractor(wav, layer=cfg.layer, reduction=cfg.reduction)
        
        return features