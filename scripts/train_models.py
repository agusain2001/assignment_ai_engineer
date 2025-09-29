"""
Model Training Script for AI Safety Models - Complete Version
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Custom dataset for text classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class LSTMEscalationModel(nn.Module):
    """LSTM model for escalation detection"""
    
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2):
        super(LSTMEscalationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(lstm_out * attention_weights, dim=1)
        
        output = self.sigmoid(self.fc(attended))
        return output


class ModelTrainer:
    """Main model training class"""
    
    def __init__(self, data_dir='data/', models_dir='trained_models/'):
        self.data_dir = data_dir
        self.models_dir = models_dir
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        self.models = {}
        self.metrics = {}
        
    def generate_comprehensive_data(self) -> Dict[str, pd.DataFrame]:
        """Generate comprehensive synthetic training data"""
        logger.info("Generating comprehensive synthetic training data...")
        
        datasets = {}
        
        # Expanded Abuse Detection Dataset
        abuse_texts = [
            # Toxic examples
            "You're so stupid and worthless", "I hate you so much", 
            "Go kill yourself", "You're an idiot", "You're terrible at this",
            "Nobody likes you", "Get lost, loser", "You disgust me",
            "You're a waste of space", "Shut up, moron", "You're pathetic",
            "I hope you fail", "You're the worst", "Die already",
            "You make me sick", "You're garbage", "Trash human",
            
            # Non-toxic examples
            "Great job on the project!", "Thank you for your help",
            "Have a wonderful day", "I appreciate your effort",
            "You're amazing!", "Well done!", "I disagree with your opinion",
            "That's not correct", "Let me explain differently",
            "I see your point", "Thanks for sharing", "Interesting perspective",
            "Could you clarify?", "I understand", "Nice work",
            "Keep it up!", "You're doing great", "Excellent point"
        ]
        
        abuse_labels = [1]*17 + [0]*18  # More balanced dataset
        
        # Add variations
        for _ in range(10):  # Create variations
            abuse_texts.extend([t.lower() for t in abuse_texts[:10]])
            abuse_labels.extend([1]*10)
            abuse_texts.extend([t.upper() for t in abuse_texts[17:27]])
            abuse_labels.extend([0]*10)
        
        datasets['abuse'] = pd.DataFrame({'text': abuse_texts, 'label': abuse_labels})
        
        # Expanded Escalation Detection Dataset
        escalation_texts = [
            # Escalating
            "This is getting worse and worse", "I can't take this anymore",
            "I've had enough of this", "Things are escalating quickly",
            "I'm getting angrier by the minute", "This is out of control",
            "STOP DOING THAT NOW", "I'm feeling overwhelmed",
            "Everything is falling apart", "I'm losing my patience",
            "This is unbearable", "I can't handle this stress",
            "My anger is boiling over", "This situation is deteriorating",
            "I'm at my breaking point", "Can't deal with this anymore",
            
            # Non-escalating
            "Let's calm down and talk", "Everything is fine",
            "No problems here", "We can work this out",
            "Let's take a break", "I understand your concern",
            "We should discuss this calmly", "Tomorrow is a new day",
            "Let's find a solution", "I'm listening to you",
            "We'll figure this out", "Stay positive",
            "Things will improve", "Let's be rational",
            "We can compromise", "I respect your opinion"
        ]
        
        escalation_labels = [1]*16 + [0]*16
        
        # Add context patterns
        for i in range(5):
            escalation_texts.extend([f"Message {i}: " + t for t in escalation_texts[:8]])
            escalation_labels.extend([1]*8)
        
        datasets['escalation'] = pd.DataFrame({'text': escalation_texts, 'label': escalation_labels})
        
        # Expanded Crisis Detection Dataset
        crisis_texts = [
            # Crisis indicators
            "I want to end it all", "Life isn't worth living",
            "I'm going to hurt myself", "Nobody would care if I died",
            "I can't go on like this", "What's the point anymore",
            "I want to disappear", "Feeling hopeless",
            "I see no future", "Death seems peaceful",
            "Planning to overdose", "Want to jump off a bridge",
            "Cutting myself again", "Pills are the answer",
            "Goodbye cruel world", "This is my last message",
            
            # Non-crisis
            "I'm having a bad day", "Things will get better",
            "I need help with homework", "I'm feeling sad today",
            "Tomorrow is a new day", "I'm stressed about work",
            "I need a vacation", "Feeling tired",
            "Work is challenging", "Missing my family",
            "Weather is gloomy", "Monday blues",
            "Need more coffee", "Running late today",
            "Traffic is terrible", "Deadline approaching"
        ]
        
        crisis_labels = [1]*16 + [0]*16
        datasets['crisis'] = pd.DataFrame({'text': crisis_texts, 'label': crisis_labels})
        
        # Expanded Content Filter Dataset
        content_texts = [
            # Inappropriate
            "Want to see something violent?", "Check out this gore video",
            "Let's talk about drugs", "Adult content warning",
            "Extreme violence depicted", "Explicit sexual content",
            "Alcohol and substance abuse", "Graphic injury images",
            "Mature themes only", "NSFW content ahead",
            "Drug dealing tips", "How to make weapons",
            "Pornographic material", "Disturbing imagery",
            "18+ only content", "Violent game footage",
            
            # Appropriate
            "Let's play a fun game", "Science homework help",
            "Math problem solving", "Cartoon recommendations",
            "Educational video about animals", "Family friendly content",
            "Children's story time", "Learn about history",
            "Fun facts for kids", "Safe search enabled",
            "Kid-friendly jokes", "Educational resources",
            "School project ideas", "Nature documentary",
            "Cooking with kids", "Art and crafts"
        ]
        
        content_labels = [1]*16 + [0]*16
        datasets['content'] = pd.DataFrame({'text': content_texts, 'label': content_labels})
        
        # Save all datasets
        for name, df in datasets.items():
            # Shuffle the data
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            df.to_csv(os.path.join(self.data_dir, f'{name}_data.csv'), index=False)
            logger.info(f"Saved {name} dataset with {len(df)} samples")
        
        return datasets
    
    def train_sklearn_model(self, X_train, y_train, X_test, y_test, model_name: str):
        """Train a scikit-learn based model with hyperparameter tuning"""
        logger.info(f"Training {model_name} model with scikit-learn...")
        
        # Create pipeline with TF-IDF and classifier
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        
        if model_name == 'abuse':
            classifier = LogisticRegression(class_weight='balanced', random_state=42)
            param_grid = {
                'classifier__C': [0.1, 1, 10],
                'classifier__penalty': ['l2'],
                'tfidf__max_features': [500, 1000, 1500]
            }
        elif model_name == 'escalation':
            classifier = RandomForestClassifier(class_weight='balanced', random_state=42)
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [10, 20, None],
                'tfidf__max_features': [500, 1000]
            }
        else:
            classifier = GradientBoostingClassifier(random_state=42)
            param_grid = {
                'classifier__n_estimators': [50, 100],
                'classifier__learning_rate': [0.05, 0.1, 0.2],
                'tfidf__max_features': [500, 1000]
            }
        
        pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('classifier', classifier)
        ])
        
        # Grid search for best parameters
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        
        # Evaluate
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        auc_score = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0
        
        metrics = {
            'precision': report.get('1', report.get('1.0', {})).get('precision', 0),
            'recall': report.get('1', report.get('1.0', {})).get('recall', 0),
            'f1_score': report.get('1', report.get('1.0', {})).get('f1-score', 0),
            'auc': auc_score,
            'accuracy': report['accuracy'],
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Save model
        model_path = os.path.join(self.models_dir, f'{model_name}_sklearn_model.pkl')
        joblib.dump(best_model, model_path)
        logger.info(f"Saved {model_name} model to {model_path}")
        
        return best_model, metrics
    
    def train_lstm_escalation_model(self, texts, labels):
        """Train LSTM model for escalation detection"""
        logger.info("Training LSTM escalation model...")
        
        # Simple tokenization
        vocab = set()
        for text in texts:
            vocab.update(text.lower().split())
        vocab_size = len(vocab) + 2  # +2 for padding and unknown
        word2idx = {word: idx+1 for idx, word in enumerate(vocab)}
        word2idx['<PAD>'] = 0
        
        # Convert texts to sequences
        sequences = []
        max_len = 50
        for text in texts:
            seq = [word2idx.get(word.lower(), 0) for word in text.split()][:max_len]
            seq += [0] * (max_len - len(seq))  # Padding
            sequences.append(seq)
        
        X = torch.tensor(sequences, dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create model
        model = LSTMEscalationModel(vocab_size)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        epochs = 10
        batch_size = 16
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 2 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_preds = (test_outputs > 0.5).float()
            accuracy = (test_preds == y_test).float().mean()
            logger.info(f"LSTM Test Accuracy: {accuracy:.4f}")
        
        # Save model
        model_path = os.path.join(self.models_dir, 'escalation_lstm_model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab_size': vocab_size,
            'word2idx': word2idx
        }, model_path)
        
        return model
    
    def train_all_models(self):
        """Train all safety models"""
        logger.info("Starting comprehensive model training...")
        
        # Generate data
        datasets = self.generate_comprehensive_data()
        
        # Train each model
        for model_name, df in datasets.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_name} model")
            logger.info(f"{'='*50}")
            
            # Prepare data
            X = df['text'].values
            y = df['label'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train sklearn model
            model, metrics = self.train_sklearn_model(
                X_train, y_train, X_test, y_test, model_name
            )
            
            self.models[model_name] = model
            self.metrics[model_name] = metrics
            
            # Print metrics
            logger.info(f"\n{model_name.upper()} Model Metrics:")
            logger.info(f"Precision: {metrics['precision']:.3f}")
            logger.info(f"Recall: {metrics['recall']:.3f}")
            logger.info(f"F1-Score: {metrics['f1_score']:.3f}")
            logger.info(f"AUC: {metrics['auc']:.3f}")
            logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
            
            # Special handling for escalation (also train LSTM)
            if model_name == 'escalation':
                self.train_lstm_escalation_model(X, y)
        
        # Save metrics summary
        metrics_path = os.path.join(self.models_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"\n{'='*50}")
        logger.info("All models trained successfully!")
        logger.info(f"Models saved to: {self.models_dir}")
        logger.info(f"Metrics saved to: {metrics_path}")
        
        return self.models, self.metrics
    
    def validate_models(self):
        """Validate all trained models exist and work"""
        logger.info("\nValidating trained models...")
        
        test_text = "This is a test message"
        
        for model_name in ['abuse', 'escalation', 'crisis', 'content']:
            model_path = os.path.join(self.models_dir, f'{model_name}_sklearn_model.pkl')
            
            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    prediction = model.predict([test_text])
                    probability = model.predict_proba([test_text])
                    logger.info(f"✓ {model_name} model loaded and working")
                except Exception as e:
                    logger.error(f"✗ {model_name} model validation failed: {e}")
            else:
                logger.warning(f"✗ {model_name} model not found at {model_path}")
        
        # Check LSTM model
        lstm_path = os.path.join(self.models_dir, 'escalation_lstm_model.pt')
        if os.path.exists(lstm_path):
            logger.info("✓ LSTM escalation model found")
        else:
            logger.warning("✗ LSTM escalation model not found")


def main():
    """Main training execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train AI Safety Models')
    parser.add_argument('--data_dir', default='data/', help='Data directory')
    parser.add_argument('--models_dir', default='trained_models/', help='Models directory')
    parser.add_argument('--validate_only', action='store_true', help='Only validate existing models')
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(data_dir=args.data_dir, models_dir=args.models_dir)
    
    if args.validate_only:
        trainer.validate_models()
    else:
        # Train all models
        models, metrics = trainer.train_all_models()
        
        # Validate after training
        trainer.validate_models()
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING COMPLETE - SUMMARY")
        print("="*60)
        for model_name, metric in metrics.items():
            print(f"\n{model_name.upper()}:")
            print(f"  F1-Score: {metric['f1_score']:.3f}")
            print(f"  AUC: {metric['auc']:.3f}")
        print("\nAll models ready for deployment!")


if __name__ == '__main__':
    main()
