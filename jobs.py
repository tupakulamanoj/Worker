from dotenv import load_dotenv
import dramatiq
import os
import logging
import uuid
import time
import json
import asyncio
import httpx
import pandas as pd
from openai import APIError
from typing import List, Union, Type, Tuple
from datetime import datetime, timedelta, timezone
from redis import Redis
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from supabase_client import supabase
from tech_news_analyzer import EnhancedTechNewsAnalyzer

from dramatiq.brokers.redis import RedisBroker
from dramatiq.results import Results
from dramatiq.results.backends import RedisBackend
from dramatiq.encoder import JSONEncoder

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enterprise_news_intel.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Redis setup
redis_url = os.getenv("REDIS_URL")
redis_client = Redis.from_url(
    os.getenv("REDIS_URL"),
    decode_responses=False,
    health_check_interval=30,
    socket_timeout=5,
    socket_connect_timeout=5
)

broker = RedisBroker(client=redis_client, encoder=JSONEncoder())
# Health check
try:
    redis_client.ping()
    logger.info("✅ Redis connection OK")
except Exception as e:
    logger.error(f"❌ Redis connection failed: {str(e)}")


broker.add_middleware(Results(backend=RedisBackend(client=redis_client)))
dramatiq.set_broker(broker)

# Network-related error classes
NETWORK_ERRORS = (httpx.NetworkError, httpx.ReadError, httpx.ConnectError, httpx.TimeoutException, APIError)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(NETWORK_ERRORS)
)
def supabase_operation(operation, *args, **kwargs):
    """Wrapper for Supabase operations with retry logic"""
    try:
        return operation(*args, **kwargs)
    except Exception as e:
        logger.error(f"Supabase operation failed: {str(e)}")
        raise

@dramatiq.actor(
    # Fix for DecodeError
    max_retries=3,
    time_limit=60 * 60 * 1000, 
    priority=5,
    queue_name="enterprise_news"
)
def run_user_job(user_id: str, email: str, company_names: str, interval: str):
    """Process a user's email job with all companies at once"""
    worker_id = str(uuid.uuid4())
    lock_timeout = timedelta(minutes=5)

    try:
        # Check for existing job with retry
        existing_job = supabase_operation(
            supabase.table("email_tracker").select("*").eq("user_id", user_id).execute
        )

        if existing_job.data:
            job = existing_job.data[0]
            if job.get("last_status") == "in_progress":
                updated_at_str = job.get("updated_at")
                if updated_at_str:
                    updated_at = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00') if updated_at_str.endswith('Z') else updated_at_str)
                    if updated_at.tzinfo is None:
                        updated_at = updated_at.replace(tzinfo=timezone.utc)
                    
                    now = datetime.now(timezone.utc)
                    lock_age = now - updated_at

                    if lock_age < lock_timeout:
                        logger.warning(f"Job already in progress for {email}")
                        return

        # Acquire lock
        supabase_operation(
            supabase.table("email_tracker").upsert({
                "user_id": user_id,
                "last_status": "in_progress",
                "last_email": email[:100],
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "worker_id": worker_id
            }).execute
        )

        # Verify lock
        verify_start = time.time()
        while True:
            current_status = supabase_operation(
                supabase.table("email_tracker").select("*").eq("user_id", user_id).execute
            )
            if current_status.data and current_status.data[0].get("worker_id") == worker_id:
                break
            if time.time() - verify_start > 30:
                logger.warning(f"Lock verification timeout for {email}")
                return
            time.sleep(1)

        logger.info(f"Starting job for {email} with companies: {company_names}")
        companies = [name.strip().lower() for name in company_names.split(',') if name.strip()]

        # Run analysis asynchronously
        logger.info(f"Starting job for {email} with companies: {company_names},{interval},{email},{tavily_api_key}")
        async def run_analysis():
            async with EnhancedTechNewsAnalyzer(
                mail=email,
                tavily_api_key=os.getenv('TAVILY_API_KEY'), 
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                companies=companies,
                frequency=interval,
                max_articles_per_company=10,
            ) as analyzer:
                logger.info(f"Starting analysis for {len(companies)} companies")
                return await analyzer.run_comprehensive_analysis()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(run_analysis())
            logger.info(f"Analysis completed successfully for {email}")
            supabase_operation(
                supabase.table("email_tracker").upsert({
                    "user_id": user_id,
                    "last_status": "success",
                    "last_email": email[:100],
                    "last_sent": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "worker_id": None
                }).execute
            )
            return results
        except Exception as e:
            logger.error(f"Analysis failed for {email}: {str(e)}", exc_info=True)
            raise
        finally:
            loop.close()

    except Exception as e:
        logger.error(f"Job failed for {email}: {str(e)}", exc_info=True)
        try:
            supabase_operation(
                supabase.table("email_tracker").upsert({
                    "user_id": user_id,
                    "last_status": f"failed: {str(e)[:100]}",
                    "last_email": email[:100],
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "worker_id": None
                }).execute
            )
        except Exception as db_error:
            logger.error(f"Failed to update failed status: {str(db_error)}")
        raise
