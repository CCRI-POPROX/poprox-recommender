from dataclasses import dataclass
from os import PathLike

import torch
from lenskit.pipeline import Component
from safetensors.torch import load_file

from poprox_concepts.domain import CandidateSet, InterestProfile
from poprox_recommender.pytorch.decorators import torch_inference
from poprox_recommender.model.miner.model import Miner
from poprox_recommender.model.miner.news_encoder import NewsEncoder
from transformers import RobertaConfig

from safetensors.torch import load_model
from transformers import AutoTokenizer, RobertaConfig


@dataclass
class MinerArticleScorerConfig:
    model_path: PathLike
    device: str = "cpu"
    max_clicks_per_user: int = 50
    title_max_len: int = 32
    sapo_max_len: int = 32


class MinerArticleScorer(Component):
    config: MinerArticleScorerConfig

    def __init__(self, config: MinerArticleScorerConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)
    #     checkpoint = load_file(self.config.model_path)

        self.device = torch.device(self.config.device)
        self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
        config = RobertaConfig.from_pretrained("FacebookAI/roberta-base")

        news_encoder = NewsEncoder.from_pretrained("FacebookAI/roberta-base", config=config,
            apply_reduce_dim=True, use_sapo=True,
            dropout=0.2, freeze_transformer=True,
            word_embed_dim=256, combine_type="linear")

        self.model = Miner(news_encoder=news_encoder, use_category_bias=False, num_context_codes=32, context_code_dim=200, score_type="weighted", dropout=0.2).to(self.device)

        print("Loading model weights...")
        loaded = load_model(self.model,"/store/trsl/erimini/poprox/poprox-recommender/models/miner/miner_2025-09-19.safetensors",)
        self.model.eval()
        print("Done")

    def _headline(self, article) -> str:
        # matches your NRMS embedder
        return (getattr(article, "headline", None) or "").strip()   

    @torch_inference
    def __call__(self, candidate_articles: CandidateSet, clicked_articles: CandidateSet, interest_profile: InterestProfile, ) -> CandidateSet:
        with_scores = candidate_articles.model_copy()

        candidates = candidate_articles.articles
        if not candidates:
            with_scores.scores = None
            return with_scores

        # ---- history----
        click_ids = [c.article_id for c in interest_profile.click_history]

        # lookup Articles by id from the articles you actually have loaded
        lookup = {a.article_id: a for a in clicked_articles.articles}
        

        his_len = self.config.max_clicks_per_user
        hist_real = [lookup[i] for i in click_ids if i in lookup]
        print("hist_real headlines:", [self._headline(a) for a in hist_real[:10]])
        hist_titles = [self._headline(a) for a in hist_real]
        pad_n = his_len - len(hist_titles)
        if pad_n > 0:
            hist_titles += [""] * pad_n

        # MINER expects sapo too; if you don't have it, reuse headline.
        cand_titles = [self._headline(a) for a in candidates]
        cand_sapo = cand_titles
        hist_sapo = hist_titles

        # tokenize candidates
        ct = self.tokenizer(
            cand_titles,
            padding=True,
            truncation=True,
            max_length=self.config.title_max_len,
            return_tensors="pt",
        )
        cs = self.tokenizer(
            cand_sapo,
            padding=True,
            truncation=True,
            max_length=self.config.sapo_max_len,
            return_tensors="pt",
        )

        # tokenize history
        ht = self.tokenizer(
            hist_titles,
            padding=True,
            truncation=True,
            max_length=self.config.title_max_len,
            return_tensors="pt",
        )
        hs = self.tokenizer(
            hist_sapo,
            padding=True,
            truncation=True,
            max_length=self.config.sapo_max_len,
            return_tensors="pt",
        )

        # reshape to MINER expected dims (batch_size=1)
        title = ct["input_ids"].unsqueeze(0).to(self.device)
        title_mask = ct["attention_mask"].unsqueeze(0).to(self.device)
        sapo = cs["input_ids"].unsqueeze(0).to(self.device)
        sapo_mask = cs["attention_mask"].unsqueeze(0).to(self.device)

        his_title = ht["input_ids"].unsqueeze(0).to(self.device)
        his_title_mask = ht["attention_mask"].unsqueeze(0).to(self.device)
        his_sapo = hs["input_ids"].unsqueeze(0).to(self.device)
        his_sapo_mask = hs["attention_mask"].unsqueeze(0).to(self.device)

        # bool mask: True where history slot is real
        his_mask = torch.zeros((1, his_len), dtype=torch.bool, device=self.device)
        if clicked_articles.articles:
            his_mask[0, : len(hist_real)] = True

        _mui, matching_scores = self.model(
            title=title,
            title_mask=title_mask,
            his_title=his_title,
            his_title_mask=his_title_mask,
            his_mask=his_mask,
            sapo=sapo,
            sapo_mask=sapo_mask,
            his_sapo=his_sapo,
            his_sapo_mask=his_sapo_mask,
        )

        with_scores.scores = matching_scores.squeeze(0).detach().cpu().numpy()
        print("=== MINER score debug ===")

        s = matching_scores.squeeze(0).detach().cpu().numpy()
        topk = s.argsort()[::-1][:10]
        print("TOP 10 by MINER score:")
        for r, i in enumerate(topk, 1):
            a = candidates[int(i)]
            print(r, "score=", float(s[int(i)]), "headline=", self._headline(a)[:90])


        return with_scores
