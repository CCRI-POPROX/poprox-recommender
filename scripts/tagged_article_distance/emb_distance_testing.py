import numpy as np
import torch as th
import torch.nn.functional as F
from local_article_emb import embed_article, load_news_encoder

def cosine_similarity(emb1: th.Tensor, emb2: th.Tensor) -> th.Tensor:
    emb1 = F.normalize(emb1, dim=0)
    emb2 = F.normalize(emb2, dim=0)
    return th.matmul(emb1, emb2)

if __name__ == "__main__":
    news_encoder, tokenizer = load_news_encoder()
    
    busines_description= "All commercial, industrial, financial and \
        economic activities involving individuals, corporations, \
        financial markets, governments and other organizations \
        across all countries and regions."
    business_emb= embed_article( busines_description , news_encoder, tokenizer)
        
    technology_description= "Tools, machines, systems or techniques, \
        especially those derived from scientific knowledge and \
        often electronic or digital in nature, for implementation \
        in industry and/or everyday human activities. \
        Includes all types of technological innovations and \
        products, such as computers, communication and \
        entertainment devices, software, industrial advancements, \
        and the issues and controversies that technology gives rise to."
        
    technology_emb= embed_article( technology_description , news_encoder, tokenizer)
        
    headline01= "Amazon and Starbucks workers are on strike. Trump might have something to do with it"
    
    headline01_emb= embed_article( headline01 , news_encoder, tokenizer)
    
    headline02= "Ex-OpenAI engineer who raised legal concerns about the technology he helped build has died"
    
    headline02_emb= embed_article( headline02 , news_encoder, tokenizer)
    breakpoint()
    
    headline01_vs_headline02= cosine_similarity(headline01_emb, headline02_emb)
    
    breakpoint()
    
    