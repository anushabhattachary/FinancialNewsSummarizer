import pandas as pd
import argparse
from datetime import datetime
import os
import json

# Import our custom classes
from scraper import FinancialNewsScraper
from summarizer import FinancialNewsSummarizer

def save_results(df, output_dir='./results'):
    """Save results to CSV and JSON"""
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"financial_news_summary_{timestamp}"
    
    # Save as CSV
    csv_path = os.path.join(output_dir, f"{base_filename}.csv")
    df.to_csv(csv_path, index=False)
    
    # Save as JSON
    json_path = os.path.join(output_dir, f"{base_filename}.json")
    df.to_json(json_path, orient='records', indent=4)
    
    return csv_path, json_path

def main():
    parser = argparse.ArgumentParser(description='AI-Powered Financial News Summarizer')
    parser.add_argument('--model', type=str, default='t5-base', 
                        help='Model to use for summarization (default: t5-base)')
    parser.add_argument('--articles', type=int, default=5, 
                        help='Number of articles to fetch per source (default: 5)')
    parser.add_argument('--summary_length', type=int, default=150, 
                        help='Maximum length of summary (default: 150)')
    parser.add_argument('--output', type=str, default='./results', 
                        help='Output directory for results (default: ./results)')
    parser.add_argument('--sources', nargs='+', default=['yahoo_finance', 'marketwatch', 'cnbc'],
                        help='Sources to fetch news from (default: all available sources)')
    parser.add_argument('--save', action='store_true', help='Save results to file')
    
    args = parser.parse_args()
    
    print(f"Initializing with model: {args.model}")
    
    # Initialize scraper and summarizer
    scraper = FinancialNewsScraper()
    summarizer = FinancialNewsSummarizer(model_name=args.model)
    
    # Fetch articles
    all_articles = []
    for source in args.sources:
        print(f"Fetching news from {source}...")
        articles = scraper.fetch_articles(source, limit=args.articles)
        all_articles.extend(articles)
    
    if not all_articles:
        print("No articles found. Exiting.")
        return
    
    # Create DataFrame
    news_df = pd.DataFrame(all_articles)
    
    # Generate summaries
    print(f"Summarizing {len(news_df)} articles...")
    results_df = summarizer.batch_summarize(
        news_df, 
        content_column='content',
        max_length=args.summary_length
    )
    
    # Display results
    for idx, row in results_df.iterrows():
        print("\n" + "="*80)
        print(f"ARTICLE {idx+1}: {row['title']}")
        print(f"SOURCE: {row['source']} | DATE: {row['date']}")
        print(f"URL: {row['url']}")
        print("-"*80)
        print(f"SUMMARY:\n{row['summary']}")
    
    # Save results if requested
    if args.save:
        csv_path, json_path = save_results(results_df, output_dir=args.output)
        print(f"\nResults saved to:\n- CSV: {csv_path}\n- JSON: {json_path}")

if __name__ == "__main__":
    main()