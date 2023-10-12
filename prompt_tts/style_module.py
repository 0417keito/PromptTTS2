import torch
import torchaudio
import omegaconf
import torch.nn as nn
import typing as tp
from transformers import BertModel
from transformers import BertTokenizer
from prompt_tts.ref_encoder.ref_encoder import SpeechPromptEncoder
from prompt_tts.ref_encoder.utils import Attention, exists
from prompt_tts.variation_net.nets import Variation_Net
from prompt_tts.conditioner import BERTConditioner
from dataset.audio_processor import AudioTokenizer, tokenize_audio
from utils.utils import dict_from_config

class StyleModule(nn.Module):
    def __init__(self, name: str, output_dim: int, device: str, 
                 n_txt_queries: int, txt_hidden_dim: int, txt_ctx_dim: int, 
                 n_ref_queries: int, ref_hidden_dim: int, ref_ctx_dim: int, 
                 dim_head: int, heads: int, dim_codebook: int = 128):
        
        super().__init__()
        self.device = device
        self.bert = BERTConditioner(name=name, output_dim=output_dim, finetune=True,
                                    device=device, normalize_text=True)
        self.codec = AudioTokenizer(device)
        self.reference_speech_encoder = SpeechPromptEncoder(dim_codebook=dim_codebook, dims=(768,))
        
        self.txt_q = nn.Embedding(n_txt_queries, txt_hidden_dim)
        self.ref_q = nn.Embedding(n_ref_queries, ref_hidden_dim)
        
        self.txt_attn = Attention(dim=txt_hidden_dim, dim_context=txt_ctx_dim,
                                  dim_head=dim_head, heads=heads)  
        self.ref_attn = Attention(dim=ref_hidden_dim, dim_context=ref_ctx_dim,
                                  dim_head=dim_head, heads=heads)
        
        self.variation_net = Variation_Net()
    
    def forward(self, batch):
        texts = batch['prompt']
        ref = batch['ref']
        txt_tokens = self.bert.tokenize(texts)
        txt_enc, mask = self.bert(txt_tokens) #([1, 5, 768])
        ref, _ = tokenize_audio(self.codec, ref)
        ref_enc = self.reference_speech_encoder(ref) #([1, 1500, 768])
        
        prompt_rep = self.txt_attn(self.txt_q, txt_enc)
        prompt_rep = self.txt_attn(x=self.txt_q.expand(txt_enc.size(0), -1, -1),
                                   context=txt_enc)
        ref_rep = self.ref_attn(x=self.ref_q.expand(ref_enc.size(0), -1, -1),
                                context=ref_enc)
        ref_rep = self.ref_attn(self.ref_q, ref_enc)
        variation_loss = self.variation_net(ref_rep, prompt_rep)
        
        return variation_loss, prompt_rep, ref_rep
    
    def infer(self): ...
        
def get_style_module(cfg: omegaconf.DictConfig) -> StyleModule:
    kwargs = dict_from_config(getattr(cfg, 'style_module'))
    name = kwargs.pop('name')
    output_dim = kwargs.pop('output_dim')
    device = kwargs.pop('device')
    n_txt_queries = kwargs.pop('n_txt_queries')
    txt_hidden_dim = kwargs.pop('txt_hidden_dim')
    txt_ctx_dim = kwargs.pop('txt_ctx_dim')
    n_ref_queries = kwargs.pop('n_ref_queries')
    ref_hidden_dim = kwargs.pop('ref_hidden_dim')
    ref_ctx_dim = kwargs.pop('ref_ctx_dim')
    dim_head = kwargs.pop('dim_head')
    heads = kwargs.pop('heads')
    dim_codebook = kwargs.pop('dim_codebook')
    return StyleModule(name=name, output_dim=output_dim, device=device,
                       n_txt_queries=n_txt_queries, txt_hidden_dim=txt_hidden_dim,
                       txt_ctx_dim=txt_ctx_dim, n_ref_queries=n_ref_queries,
                       ref_hidden_dim=ref_hidden_dim, ref_ctx_dim=ref_ctx_dim,
                       dim_head=dim_head, heads=heads, dim_codebook=dim_codebook)