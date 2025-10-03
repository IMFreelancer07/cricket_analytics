"""
Data Collection and Processing Module for Cricket Analytics
"""

import os
import pandas as pd
import requests
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CricketDataCollector:
    """Collect cricket data from various sources"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_cricinfo_commentary(self, match_id: str) -> Dict[str, Any]:
        """
        Collect match commentary from Cricinfo
        
        Args:
            match_id: Cricinfo match identifier
            
        Returns:
            Dictionary containing commentary data
        """
        try:
            # Placeholder for Cricinfo API integration
            # In real implementation, use proper Cricinfo API endpoints
            commentary_data = {
                "match_id": match_id,
                "commentary": [],
                "match_info": {},
                "scorecard": {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to file
            output_file = self.data_dir / f"commentary_{match_id}.json"
            import json
            with open(output_file, 'w') as f:
                json.dump(commentary_data, f, indent=2)
                
            logger.info(f"Commentary data saved for match {match_id}")
            return commentary_data
            
        except Exception as e:
            logger.error(f"Failed to collect commentary for match {match_id}: {e}")
            return {}
    
    def collect_ball_by_ball_data(self, dataset_name: str = "cricsheet") -> pd.DataFrame:
        """
        Collect ball-by-ball cricket data
        
        Args:
            dataset_name: Name of the dataset source
            
        Returns:
            DataFrame containing ball-by-ball data
        """
        try:
            # Placeholder for actual data collection
            # In real implementation, integrate with Kaggle API or Cricsheet
            
            sample_data = {
                'match_id': ['m1', 'm1', 'm1', 'm2', 'm2'],
                'inning': [1, 1, 1, 1, 1],
                'over': [1, 1, 2, 1, 1],
                'ball': [1, 2, 1, 1, 2],
                'batsman': ['V Kohli', 'V Kohli', 'R Sharma', 'K Williamson', 'K Williamson'],
                'bowler': ['J Bumrah', 'J Bumrah', 'M Shami', 'B Kumar', 'B Kumar'],
                'runs_scored': [4, 1, 0, 2, 6],
                'wicket_type': [None, None, 'bowled', None, None],
                'venue': ['Eden Gardens', 'Eden Gardens', 'Eden Gardens', 'MCG', 'MCG']
            }
            
            df = pd.DataFrame(sample_data)
            
            # Save to file
            output_file = self.data_dir / f"ball_by_ball_{dataset_name}.csv"
            df.to_csv(output_file, index=False)
            
            logger.info(f"Ball-by-ball data collected: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to collect ball-by-ball data: {e}")
            return pd.DataFrame()
    
    def collect_match_reports(self, match_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Collect match reports and analysis
        
        Args:
            match_ids: List of match identifiers
            
        Returns:
            List of match report dictionaries
        """
        reports = []
        
        for match_id in match_ids:
            try:
                # Placeholder for match report collection
                report = {
                    "match_id": match_id,
                    "title": f"Match Report - {match_id}",
                    "content": f"Detailed analysis of match {match_id}...",
                    "key_moments": [],
                    "player_performances": {},
                    "tactical_insights": [],
                    "timestamp": datetime.now().isoformat()
                }
                reports.append(report)
                
                # Save individual report
                output_file = self.data_dir / f"report_{match_id}.json"
                import json
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2)
                    
            except Exception as e:
                logger.error(f"Failed to collect report for match {match_id}: {e}")
        
        logger.info(f"Collected {len(reports)} match reports")
        return reports


class CricketDataProcessor:
    """Process and clean cricket data for RAG system"""
    
    def __init__(self, raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def process_commentary_for_rag(self, commentary_file: str) -> List[Dict[str, str]]:
        """
        Process commentary data for RAG ingestion
        
        Args:
            commentary_file: Path to commentary JSON file
            
        Returns:
            List of processed text chunks with metadata
        """
        try:
            import json
            
            with open(self.raw_data_dir / commentary_file, 'r') as f:
                data = json.load(f)
            
            processed_chunks = []
            
            # Process commentary into chunks
            for i, comment in enumerate(data.get('commentary', [])):
                chunk = {
                    'text': comment.get('text', ''),
                    'over': comment.get('over', ''),
                    'ball': comment.get('ball', ''),
                    'match_id': data.get('match_id', ''),
                    'chunk_id': f"{data.get('match_id', '')}_{i}",
                    'type': 'commentary'
                }
                processed_chunks.append(chunk)
            
            # Save processed chunks
            output_file = self.processed_data_dir / f"processed_{commentary_file}"
            with open(output_file, 'w') as f:
                json.dump(processed_chunks, f, indent=2)
            
            logger.info(f"Processed {len(processed_chunks)} commentary chunks")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Failed to process commentary file {commentary_file}: {e}")
            return []
    
    def create_player_performance_summaries(self, ball_by_ball_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create player performance summaries from ball-by-ball data
        
        Args:
            ball_by_ball_df: DataFrame with ball-by-ball data
            
        Returns:
            DataFrame with player performance summaries
        """
        try:
            # Batting summaries
            batting_stats = ball_by_ball_df.groupby(['match_id', 'batsman']).agg({
                'runs_scored': ['sum', 'count'],
                'wicket_type': lambda x: x.notna().sum()
            }).reset_index()
            
            batting_stats.columns = ['match_id', 'player', 'total_runs', 'balls_faced', 'dismissed']
            batting_stats['strike_rate'] = (batting_stats['total_runs'] / batting_stats['balls_faced']) * 100
            batting_stats['role'] = 'batsman'
            
            # Bowling summaries
            bowling_stats = ball_by_ball_df.groupby(['match_id', 'bowler']).agg({
                'runs_scored': 'sum',
                'wicket_type': lambda x: x.notna().sum(),
                'ball': 'count'
            }).reset_index()
            
            bowling_stats.columns = ['match_id', 'player', 'runs_conceded', 'wickets', 'balls_bowled']
            bowling_stats['economy'] = (bowling_stats['runs_conceded'] / (bowling_stats['balls_bowled'] / 6))
            bowling_stats['role'] = 'bowler'
            
            # Combine stats
            all_stats = pd.concat([
                batting_stats[['match_id', 'player', 'role', 'total_runs', 'balls_faced', 'strike_rate']],
                bowling_stats[['match_id', 'player', 'role', 'runs_conceded', 'wickets', 'economy']]
            ], ignore_index=True)
            
            # Save processed stats
            output_file = self.processed_data_dir / "player_performance_summaries.csv"
            all_stats.to_csv(output_file, index=False)
            
            logger.info(f"Created performance summaries for {len(all_stats)} player-match combinations")
            return all_stats
            
        except Exception as e:
            logger.error(f"Failed to create player performance summaries: {e}")
            return pd.DataFrame()


# Utility functions
def download_sample_data():
    """Download sample cricket datasets for development"""
    try:
        collector = CricketDataCollector()
        
        # Collect sample data
        sample_matches = ['m1', 'm2', 'm3']
        
        for match_id in sample_matches:
            collector.collect_cricinfo_commentary(match_id)
        
        ball_by_ball_df = collector.collect_ball_by_ball_data()
        collector.collect_match_reports(sample_matches)
        
        # Process the data
        processor = CricketDataProcessor()
        processor.create_player_performance_summaries(ball_by_ball_df)
        
        logger.info("Sample data download and processing completed")
        
    except Exception as e:
        logger.error(f"Failed to download sample data: {e}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Download sample data
    download_sample_data()