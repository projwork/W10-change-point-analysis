"""
Oil Market Events Data Module

This module contains major geopolitical events, OPEC decisions, and economic shocks
that have historically impacted oil markets, compiled for change point analysis.
"""

import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional


class OilMarketEvents:
    """
    Repository of major oil market events for analysis.
    """
    
    def __init__(self):
        """Initialize with major oil market events."""
        self.events = self._compile_major_events()
    
    def _compile_major_events(self) -> List[Dict]:
        """
        Compile major oil market events with dates and descriptions.
        
        Returns:
            List[Dict]: List of events with date, type, description, and impact
        """
        events = [
            {
                'date': '1987-10-19',
                'type': 'Economic Crisis',
                'event': 'Black Monday Stock Market Crash',
                'description': 'Global stock market crash affecting commodity prices',
                'expected_impact': 'Negative',
                'magnitude': 'High'
            },
            {
                'date': '1990-08-02',
                'type': 'Geopolitical',
                'event': 'Iraq Invasion of Kuwait',
                'description': 'Start of Gulf War crisis, major oil supply disruption',
                'expected_impact': 'Positive',
                'magnitude': 'Very High'
            },
            {
                'date': '1991-01-17',
                'type': 'Geopolitical',
                'event': 'Gulf War Begins',
                'description': 'Coalition forces attack Iraq, oil facilities targeted',
                'expected_impact': 'Positive',
                'magnitude': 'Very High'
            },
            {
                'date': '1997-07-02',
                'type': 'Economic Crisis',
                'event': 'Asian Financial Crisis',
                'description': 'Asian financial crisis reducing oil demand',
                'expected_impact': 'Negative',
                'magnitude': 'High'
            },
            {
                'date': '1998-12-28',
                'type': 'OPEC Decision',
                'event': 'OPEC Production Cuts',
                'description': 'OPEC agrees to significant production cuts to support prices',
                'expected_impact': 'Positive',
                'magnitude': 'Medium'
            },
            {
                'date': '2001-09-11',
                'type': 'Geopolitical',
                'event': '9/11 Terrorist Attacks',
                'description': 'Terrorist attacks in US causing economic uncertainty',
                'expected_impact': 'Positive',
                'magnitude': 'High'
            },
            {
                'date': '2003-03-20',
                'type': 'Geopolitical',
                'event': 'Iraq War Begins',
                'description': 'US-led invasion of Iraq, oil infrastructure concerns',
                'expected_impact': 'Positive',
                'magnitude': 'High'
            },
            {
                'date': '2005-08-29',
                'type': 'Natural Disaster',
                'event': 'Hurricane Katrina',
                'description': 'Major hurricane disrupting Gulf of Mexico oil production',
                'expected_impact': 'Positive',
                'magnitude': 'High'
            },
            {
                'date': '2008-09-15',
                'type': 'Economic Crisis',
                'event': 'Lehman Brothers Collapse',
                'description': 'Global financial crisis beginning, demand destruction',
                'expected_impact': 'Negative',
                'magnitude': 'Very High'
            },
            {
                'date': '2010-04-20',
                'type': 'Natural Disaster',
                'event': 'Deepwater Horizon Oil Spill',
                'description': 'Major oil spill in Gulf of Mexico affecting production',
                'expected_impact': 'Positive',
                'magnitude': 'Medium'
            },
            {
                'date': '2011-02-17',
                'type': 'Geopolitical',
                'event': 'Arab Spring - Libya',
                'description': 'Libyan civil war disrupting oil production',
                'expected_impact': 'Positive',
                'magnitude': 'High'
            },
            {
                'date': '2014-06-01',
                'type': 'Supply Shock',
                'event': 'US Shale Oil Boom Peak',
                'description': 'US shale oil production surge changing global dynamics',
                'expected_impact': 'Negative',
                'magnitude': 'Very High'
            },
            {
                'date': '2014-11-27',
                'type': 'OPEC Decision',
                'event': 'OPEC Maintains Production',
                'description': 'OPEC decides not to cut production despite falling prices',
                'expected_impact': 'Negative',
                'magnitude': 'High'
            },
            {
                'date': '2016-02-16',
                'type': 'OPEC Decision',
                'event': 'Doha Meeting Agreement',
                'description': 'Oil producers agree to freeze production levels',
                'expected_impact': 'Positive',
                'magnitude': 'Medium'
            },
            {
                'date': '2016-11-30',
                'type': 'OPEC Decision',
                'event': 'OPEC Production Cut Agreement',
                'description': 'First OPEC production cut agreement in 8 years',
                'expected_impact': 'Positive',
                'magnitude': 'High'
            },
            {
                'date': '2018-05-08',
                'type': 'Geopolitical',
                'event': 'US Withdraws from Iran Nuclear Deal',
                'description': 'US reimposition of sanctions on Iranian oil',
                'expected_impact': 'Positive',
                'magnitude': 'High'
            },
            {
                'date': '2020-03-06',
                'type': 'OPEC Decision',
                'event': 'OPEC+ Deal Collapse',
                'description': 'Russia refuses additional cuts, Saudi Arabia responds with price war',
                'expected_impact': 'Negative',
                'magnitude': 'Very High'
            },
            {
                'date': '2020-03-11',
                'type': 'Economic Crisis',
                'event': 'WHO Declares COVID-19 Pandemic',
                'description': 'Global pandemic lockdowns causing massive demand destruction',
                'expected_impact': 'Negative',
                'magnitude': 'Very High'
            },
            {
                'date': '2020-04-12',
                'type': 'OPEC Decision',
                'event': 'Historic OPEC+ Production Cuts',
                'description': 'Largest production cut in OPEC history (9.7M barrels/day)',
                'expected_impact': 'Positive',
                'magnitude': 'Very High'
            },
            {
                'date': '2022-02-24',
                'type': 'Geopolitical',
                'event': 'Russia Invades Ukraine',
                'description': 'Major oil producer (Russia) involved in conflict, supply concerns',
                'expected_impact': 'Positive',
                'magnitude': 'Very High'
            }
        ]
        
        return events
    
    def get_events_dataframe(self) -> pd.DataFrame:
        """
        Get events as a pandas DataFrame.
        
        Returns:
            pd.DataFrame: Events data with parsed dates
        """
        df = pd.DataFrame(self.events)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        return df
    
    def get_events_for_period(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get events within a specific time period.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Events within the specified period
        """
        df = self.get_events_dataframe()
        return df.loc[start_date:end_date]
    
    def get_events_by_type(self, event_type: str) -> pd.DataFrame:
        """
        Get events of a specific type.
        
        Args:
            event_type (str): Type of event ('Geopolitical', 'OPEC Decision', 'Economic Crisis', etc.)
            
        Returns:
            pd.DataFrame: Events of the specified type
        """
        df = self.get_events_dataframe()
        return df[df['type'] == event_type]
    
    def export_events(self, output_path: str) -> None:
        """
        Export events data to CSV file.
        
        Args:
            output_path (str): Path for output CSV file
        """
        df = self.get_events_dataframe()
        df.to_csv(output_path)
        print(f"Events data exported to {output_path}")
    
    def add_event(self, date: str, event_type: str, event_name: str, 
                  description: str, expected_impact: str, magnitude: str) -> None:
        """
        Add a new event to the events list.
        
        Args:
            date (str): Event date in YYYY-MM-DD format
            event_type (str): Type of event
            event_name (str): Name of the event
            description (str): Detailed description
            expected_impact (str): Expected impact on oil prices ('Positive'/'Negative')
            magnitude (str): Impact magnitude ('Low'/'Medium'/'High'/'Very High')
        """
        new_event = {
            'date': date,
            'type': event_type,
            'event': event_name,
            'description': description,
            'expected_impact': expected_impact,
            'magnitude': magnitude
        }
        self.events.append(new_event)
    
    def get_summary_statistics(self) -> dict:
        """
        Get summary statistics of events data.
        
        Returns:
            dict: Summary statistics
        """
        df = self.get_events_dataframe()
        
        summary = {
            'total_events': len(df),
            'event_types': df['type'].value_counts().to_dict(),
            'impact_distribution': df['expected_impact'].value_counts().to_dict(),
            'magnitude_distribution': df['magnitude'].value_counts().to_dict(),
            'date_range': {
                'first_event': df.index.min(),
                'last_event': df.index.max()
            }
        }
        
        return summary


def load_oil_market_events() -> pd.DataFrame:
    """
    Convenience function to load oil market events data.
    
    Returns:
        pd.DataFrame: Oil market events data
    """
    events = OilMarketEvents()
    return events.get_events_dataframe()


if __name__ == "__main__":
    # Example usage
    events = OilMarketEvents()
    
    # Get events DataFrame
    events_df = events.get_events_dataframe()
    print(f"Loaded {len(events_df)} major oil market events")
    
    # Get summary statistics
    summary = events.get_summary_statistics()
    print("\nEvents Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Export to CSV
    events.export_events("data/oil_market_events.csv")