# -*- coding: utf-8 -*-
"""
Fetches historical F1 race results (driver and finishing position)
from the Ergast API (http://ergast.com/mrd/)
"""

import requests
import csv
import os
import time

# Configuration
BASE_URL = "http://ergast.com/api/f1"
START_YEAR = 2023 # Align with OpenF1 data availability
END_YEAR = 2024   # Adjust as needed
OUTPUT_DIR = "../data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ergast_race_results.csv")
REQUEST_DELAY = 0.2 # Seconds to wait between requests to be polite

def fetch_race_results(year, round_num):
    """Fetches race results for a specific year and round."""
    results = []
    limit = 100 # Max results per page for Ergast
    offset = 0
    print(f"   Fetching {year} Round {round_num}...", end="", flush=True)

    while True:
        url = f"{BASE_URL}/{year}/{round_num}/results.json?limit={limit}&offset={offset}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            race_data = data.get('MRData', {}).get('RaceTable', {}).get('Races', [])
            if not race_data:
                print(f" No race data found.")
                return None # No race found for this year/round combination

            race_results = race_data[0].get('Results', [])
            if not race_results:
                print(f" No results found.")
                break # No more results for this race

            for result in race_results:
                driver_info = result.get('Driver', {})
                results.append({
                    'year': year,
                    'round': round_num,
                    'driver_id': driver_info.get('driverId'),
                    'driver_number': result.get('number'),
                    'position': result.get('position'),
                    'points': result.get('points'),
                    'status': result.get('status')
                })

            # Check if we need to fetch more pages
            total_results = int(data.get('MRData', {}).get('total', 0))
            if offset + limit >= total_results:
                break # Got all results
            else:
                offset += limit
                time.sleep(REQUEST_DELAY)

        except requests.exceptions.RequestException as e:
            print(f"\n      Error fetching {year} Round {round_num}: {e}")
            return None # Indicate failure for this race
        except Exception as e:
            print(f"\n      Unexpected error processing {year} Round {round_num}: {e}")
            return None

    print(f" Done ({len(results)} drivers).", flush=True)
    return results

def get_max_round(year):
    """Gets the maximum round number for a given year."""
    url = f"{BASE_URL}/{year}.json"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        races = data.get('MRData', {}).get('RaceTable', {}).get('Races', [])
        return len(races)
    except requests.exceptions.RequestException as e:
        print(f"\nError fetching schedule for {year}: {e}")
        return 0
    except Exception as e:
        print(f"\nUnexpected error fetching schedule for {year}: {e}")
        return 0

def main():
    """Main function to fetch results for the specified year range."""
    all_results = []
    print(f"Fetching race results from {START_YEAR} to {END_YEAR}...")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"Processing Year: {year}")
        max_round = get_max_round(year)
        if max_round == 0:
            print(f"   Could not determine rounds for {year}, skipping.")
            continue
        print(f"   Found {max_round} rounds for {year}.")

        for round_num in range(1, max_round + 1):
            race_results = fetch_race_results(year, round_num)
            if race_results:
                all_results.extend(race_results)
            time.sleep(REQUEST_DELAY) # Be polite between rounds

    if not all_results:
        print("\nNo results fetched. Exiting.")
        return

    # Write to CSV
    print(f"\nWriting {len(all_results)} total results to {OUTPUT_FILE}...")
    try:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['year', 'round', 'driver_id', 'driver_number', 'position', 'points', 'status']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(all_results)
        print("Successfully wrote results to CSV.")
    except IOError as e:
        print(f"Error writing to CSV file {OUTPUT_FILE}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during CSV writing: {e}")

if __name__ == "__main__":
    main()

