import torch
import torch.nn as nn
from poprox_recommender.model.nrms.news_encoder import NewsEncoder
from poprox_recommender.model.nrms.user_encoder import UserEncoder
from poprox_recommender.model.general.click_predictor.dot_product import (
    DotProductClickPredictor,
)


class NRMS(torch.nn.Module):
    """
    NRMS network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """

    def __init__(self, args):
        super(NRMS, self).__init__()
        
        self.news_encoder = NewsEncoder(args)
        self.user_encoder = UserEncoder(args)
        self.click_predictor = DotProductClickPredictor()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, candidate_news, clicked_news, clicked, mode = 'train'):
       
        # batch_size, 1 + K, word_embedding_dim
        candidate_news = candidate_news.permute(1, 0, 2)
        clicked_news = clicked_news.permute(1, 0, 2)

        candidate_news_vector = torch.stack(
            [self.news_encoder(x) for x in candidate_news], dim=1
        )

        # batch_size, num_clicked_news_a_user, word_embedding_dim
        clicked_news_vector = torch.stack(
            [self.news_encoder(x) for x in clicked_news], dim=1
        )

        # batch_size, word_embedding_dim
        user_vector = self.user_encoder(clicked_news_vector)
        # batch_size, 1 + K
        click_probability = self.click_predictor(candidate_news_vector, user_vector)
        
        if mode == 'train': 
            return {'click_prob': click_probability, 
                    'loss': self.loss_fn(click_probability, clicked)
            }
        
        return click_probability

    def get_news_vector(self, news):
        """
        Args:
            news:
                {
                    "title": batch_size * num_words_title
                },
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, word_embedding_dim
        return self.news_encoder(news)

    def get_user_vector(self, clicked_news_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, word_embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, word_embedding_dim
        return self.user_encoder(clicked_news_vector)

    def get_prediction(self, news_vector, user_vector):
        """
        Args:
            news_vector: candidate_size, word_embedding_dim
            user_vector: word_embedding_dim
        Returns:
            click_probability: candidate_size
        """
        # candidate_size
        return self.click_predictor(
            news_vector.unsqueeze(dim=0), user_vector.unsqueeze(dim=0)
        ).squeeze(dim=0)
