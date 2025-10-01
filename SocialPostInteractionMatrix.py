from scipy.sparse import csr_matrix
import polars as pl


class SocialPostInteractionMatrix:

    def build(self, training_set):
        user_ids = training_set.select("user_id").unique().sort("user_id")["user_id"]
        post_ids = training_set.select("post_id").unique().sort("post_id")["post_id"]
        
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self.post_to_idx = {post_id: idx for idx, post_id in enumerate(post_ids)}
        self.idx_to_post = {idx: post_id for post_id, idx in self.post_to_idx.items()}
        
        user_indices = training_set.with_columns(
            pl.col("user_id").replace(self.user_to_idx)
        )["user_id"].to_numpy()
        
        post_indices = training_set.with_columns(
            pl.col("post_id").replace(self.post_to_idx)
        )["post_id"].to_numpy()
        
        engagement_scores = training_set["engagement_score"].to_numpy()
        
        sparse_matrix = csr_matrix(
            (engagement_scores, (user_indices, post_indices)),
            shape=(len(self.user_to_idx), len(self.post_to_idx))
        )
        
        return sparse_matrix