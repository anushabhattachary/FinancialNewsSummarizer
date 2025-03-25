import pandas as pd
import numpy as np
from datasets import Dataset
from summarizer import FinancialNewsSummarizer
import argparse
import os

def create_training_dataset(financial_news_csv, financial_summaries_csv=None):
    """
    Create a training dataset for fine-tuning
    
    Parameters:
    financial_news_csv (str): Path to CSV file with financial news articles
    financial_summaries_csv (str, optional): Path to CSV file with expert summaries
    
    Returns:
    pandas.DataFrame: DataFrame with 'text' and 'summary' columns
    """
    # Load news articles
    news_df = pd.read_csv(financial_news_csv)
    
    if financial_summaries_csv:
        # If we have expert summaries, use those
        summaries_df = pd.read_csv(financial_summaries_csv)
        
        # Merge news with summaries
        training_df = pd.merge(
            news_df, 
            summaries_df, 
            left_on='id', 
            right_on='article_id', 
            how='inner'
        )
        
        # Select relevant columns
        training_data = training_df[['content', 'expert_summary']].rename(
            columns={'content': 'text', 'expert_summary': 'summary'}
        )
    else:
        # If we don't have expert summaries, use the base model to create initial summaries
        print("No expert summaries provided. Using base model to generate initial summaries.")
        summarizer = FinancialNewsSummarizer()
        
        # Generate summaries
        news_df['summary'] = news_df['content'].apply(
            lambda x: summarizer.summarize(x, max_length=150, min_length=50)
        )
        
        # Select relevant columns
        training_data = news_df[['content', 'summary']].rename(
            columns={'content': 'text'}
        )
    
    return training_data

def main():
    parser = argparse.ArgumentParser(description='Fine-tune summarizer for financial domain')
    parser.add_argument('--news_data', type=str, required=True, 
                        help='Path to CSV file with financial news articles')
    parser.add_argument('--summaries', type=str, default=None,
                        help='Path to CSV file with expert summaries (optional)')
    parser.add_argument('--base_model', type=str, default='t5-base', 
                        help='Base model to fine-tune (default: t5-base)')
    parser.add_argument('--output_dir', type=str, default='./fine_tuned_model', 
                        help='Directory to save fine-tuned model')
    parser.add_argument('--epochs', type=int, default=3, 
                        help='Number of training epochs (default: 3)')
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='Batch size for training (default: 4)')
    
    args = parser.parse_args()
    
    # Create training dataset
    print("Creating training dataset...")
    training_data = create_training_dataset(args.news_data, args.summaries)
    
    print(f"Training data created with {len(training_data)} examples")
    
    # Initialize summarizer
    print(f"Initializing summarizer with model: {args.base_model}")
    summarizer = FinancialNewsSummarizer(model_name=args.base_model)
    
    # Fine-tune the model
    print("Starting fine-tuning...")
    summarizer.fine_tune_for_finance(
        training_data, 
        epochs=args.epochs, 
        batch_size=args.batch_size
    )
    
    # Save the fine-tuned model
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    summarizer.model.save_pretrained(args.output_dir)
    summarizer.tokenizer.save_pretrained(args.output_dir)
    
    print(f"Fine-tuned model saved to {args.output_dir}")
    
    # Test the fine-tuned model
    sample_text = training_data['text'].iloc[0]
    print("\nTesting fine-tuned model with sample article:")
    summary = summarizer.summarize(sample_text)
    print(f"Summary: {summary}")

if __name__ == "__main__":
    main()