try:
    from poprox_recommender.components.embedders.article import NRMSArticleEmbedder, NRMSArticleEmbedderConfig
    from poprox_recommender.components.embedders.user import NRMSUserEmbedder, NRMSUserEmbedderConfig
    from poprox_recommender.components.embedders.user_persona import UserPersonaEmbedder, UserPersonaConfig
    __all__ = ["NRMSArticleEmbedder", "NRMSArticleEmbedderConfig", "NRMSUserEmbedder", "NRMSUserEmbedderConfig", "UserPersonaEmbedder", "UserPersonaConfig"]
except ImportError:
    # For testing without full dependencies
    __all__ = []
