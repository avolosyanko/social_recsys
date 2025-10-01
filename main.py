import os
import logging
import implicit
import json

from SocialPostRecommender import SocialPostRecommender
from SocialPostDAO import SocialPostDAO

logging.basicConfig(level=logging.INFO)

def main() -> None:
    logging.info("Starting social post recommendation prototype")
    
    test_recs = 10
    
    social_dao = SocialPostDAO()
    model = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.01)
    recommender = SocialPostRecommender(model=model, dao=social_dao)
    
    try:
        recommender.fit()
        logging.info("Model training completed successfully")
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        return
    
    try:
        all_recommendations = recommender.get_top_recommendations(N=test_recs)
        logging.info(f"Generated recommendations for {len(all_recommendations)} users")
    except Exception as e:
        logging.error(f"Error generating recommendations: {e}")
        return
    
    with open('social_recommendations.json', 'w') as f:
        json.dump(all_recommendations, f, indent=2)
    
    logging.info(f"Prototype complete.")

if __name__ == "__main__":
    main()