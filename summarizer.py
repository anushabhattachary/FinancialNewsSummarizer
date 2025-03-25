from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pandas as pd
import numpy as np
import torch
import re

class FinancialNewsSummarizer:
    def __init__(self, model_name='t5-base'):
        """
        Initialize the summarizer with a pre-trained model
        
        Parameters:
        model_name (str): Name of the Hugging Face model to use for summarization
                         Default options: 't5-base', 't5-large', 'bart-large-cnn'
        """
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        # Create summarization pipeline
        self.summarizer = pipeline(
            "summarization", 
            model=self.model, 
            tokenizer=self.tokenizer,
            device=0 if self.device == 'cuda' else -1
        )
        
        # Financial terms dictionary for post-processing
        self.financial_terms = {
            'bull market': 'a market characterized by rising prices',
            'bear market': 'a market characterized by falling prices',
            'ETF': 'Exchange Traded Fund',
            'IPO': 'Initial Public Offering',
            'ROI': 'Return On Investment',
            'EPS': 'Earnings Per Share',
            'P/E': 'Price to Earnings ratio',
            'EBITDA': 'Earnings Before Interest, Taxes, Depreciation, and Amortization',
            'YOY': 'Year Over Year',
            'QOQ': 'Quarter Over Quarter'
        }
    
    def preprocess_text(self, text):
        """Clean and prepare the text for summarization"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove any URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Limit text length to model's max input length
        max_length = self.tokenizer.model_max_length
        if len(text) > max_length:
            text = text[:max_length]
            
        return text.strip()
    
    def postprocess_summary(self, summary):
        """Enhance summary with financial context"""
        processed_summary = summary
        
        # Expand financial acronyms on first mention
        for term, definition in self.financial_terms.items():
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, processed_summary, re.IGNORECASE) and term.isupper():
                replacement = f"{term} ({definition})"
                processed_summary = re.sub(pattern, replacement, processed_summary, count=1, flags=re.IGNORECASE)
        
        return processed_summary
    
    def summarize(self, text, max_length=150, min_length=50):
        """
        Summarize financial news text
        
        Parameters:
        text (str): The text to summarize
        max_length (int): Maximum length of the summary
        min_length (int): Minimum length of the summary
        
        Returns:
        str: Summarized text
        """
        # Handle empty or very short text
        if not text or len(text) < 100:
            return "Text too short to summarize."
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # For T5 models, we need to add a prefix
        if 't5' in self.model_name.lower():
            processed_text = "summarize: " + processed_text
        
        try:
            # Generate summary
            summary = self.summarizer(
                processed_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            
            # Process the summary output
            if isinstance(summary, list) and len(summary) > 0:
                summary_text = summary[0]['summary_text']
                return self.postprocess_summary(summary_text)
            else:
                return "Failed to generate summary."
                
        except Exception as e:
            return f"Error during summarization: {str(e)}"
    
    def batch_summarize(self, df, content_column='content', max_length=150, min_length=50):
        """
        Summarize a batch of articles in a DataFrame
        
        Parameters:
        df (pandas.DataFrame): DataFrame containing news articles
        content_column (str): Name of the column containing article text
        max_length (int): Maximum length of each summary
        min_length (int): Minimum length of each summary
        
        Returns:
        pandas.DataFrame: Original DataFrame with added summary column
        """
        if content_column not in df.columns:
            raise ValueError(f"Column '{content_column}' not found in DataFrame")
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Generate summaries
        result_df['summary'] = result_df[content_column].apply(
            lambda x: self.summarize(x, max_length=max_length, min_length=min_length)
        )
        
        return result_df

    def fine_tune_for_finance(self, train_data, epochs=3, batch_size=4):
        """
        Fine-tune the model on financial data (optional enhancement)
        
        Parameters:
        train_data (DataFrame): DataFrame with columns 'text' and 'summary'
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
        Note: This is a simplified fine-tuning function. For production,
        you would want to implement proper dataset handling, validation, etc.
        """
        from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
        from datasets import Dataset
        
        # Convert DataFrame to Hugging Face Dataset
        train_dataset = Dataset.from_pandas(train_data)
        
        # Prepare the training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=epochs,
            predict_with_generate=True,
        )
        
        # Define data collator
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        
        # Prepare inputs for T5-style models
        def preprocess_function(examples):
            inputs = ["summarize: " + doc for doc in examples["text"]]
            model_inputs = self.tokenizer(inputs, max_length=1024, truncation=True)
            
            # Set up the summary labels
            labels = self.tokenizer(examples["summary"], max_length=128, truncation=True)
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Preprocess dataset
        train_dataset = train_dataset.map(preprocess_function, batched=True)
        
        # Create trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train model
        trainer.train()
        
        # Update model and pipeline
        self.model = trainer.model
        self.summarizer = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer, 
            device=0 if self.device == 'cuda' else -1
        )
        
        print("Fine-tuning complete!")