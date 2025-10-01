from SocialPostInteractionMatrix import SocialPostInteractionMatrix
from SocialPostDAO import SocialPostDAO
import logging


class SocialPostRecommender:

    def __init__(self, model, dao):
        self.model = model
        self.dao = dao
        self.interaction_matrix = SocialPostInteractionMatrix()

    def fit(self):
        
        training_set = self.dao.get_training_set()
        self.csr_matrix = self.interaction_matrix.build(training_set)
        self.model.fit(self.csr_matrix)

    def get_top_recommendations(self, N=10):
        logging.info(f"Generating top {N} recommendations for all users")
        
        user_ids = list(self.interaction_matrix.user_to_idx.keys())
        user_indices = [self.interaction_matrix.user_to_idx[user_id] for user_id in user_ids]
        
        recommended_post_indices, scores = self.model.recommend(
            user_indices, self.csr_matrix, N=N, filter_already_liked_items=True
        )
        
        recommended_post_ids = [
            [self.interaction_matrix.idx_to_post[post_idx] for post_idx in user_post_indices]
            for user_post_indices in recommended_post_indices
        ]
        
        recommendations = dict(zip(user_ids, recommended_post_ids))
        logging.info(f"Generated recommendations for {len(recommendations)} users")
        return recommendations