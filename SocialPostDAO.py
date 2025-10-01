import logging
import polars as pl
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1
from datetime import datetime, timedelta, timezone


class SocialPostDAO:
    TEMP_DATASET = "profile_recommendation_service"

    def __init__(self):
        self.client = bigquery.Client(project="model-academy-434609-h8")
        self.bqstorage_client = bigquery_storage_v1.BigQueryReadClient()
        self.project_id = "model-academy-434609-h8"

    def get_training_set(self):
        logging.info(f"Starting social engagement training.")
        
        query = """
        SELECT 
            social_profile_id as user_id,
            thing_id as post_id,
            1 as engagement_score
        FROM `prod.sociallikes`
        WHERE thing_id LIKE 'pos_%'
        AND timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)

        UNION ALL

        SELECT 
            author_id as user_id,
            post_id,
            3 as engagement_score
        FROM `prod.socialcomments`
        WHERE created_at > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
        """
        
        query_job = self.client.query(query)
        results = query_job.result()
        
        df = results.to_dataframe()
        return pl.from_pandas(df)