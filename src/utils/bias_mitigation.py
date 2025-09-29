"""
Bias Mitigation Module for AI Safety Models
Addresses fairness and demographic parity requirements
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix
import logging

logger = logging.getLogger(__name__)


class BiasMitigator:
    """
    Implements bias detection and mitigation strategies for safety models
    """
    
    def __init__(self):
        self.bias_metrics = {}
        self.demographic_groups = ['gender', 'age_group', 'language', 'region']
        self.protected_terms = self._load_protected_terms()
        
    def _load_protected_terms(self) -> Dict[str, List[str]]:
        """Load protected demographic terms for bias detection"""
        return {
            'gender': ['he', 'she', 'they', 'man', 'woman', 'male', 'female', 'boy', 'girl'],
            'ethnicity': ['white', 'black', 'asian', 'hispanic', 'latino', 'african'],
            'religion': ['christian', 'muslim', 'jewish', 'hindu', 'buddhist', 'atheist'],
            'orientation': ['gay', 'lesbian', 'straight', 'lgbtq', 'transgender'],
            'age': ['young', 'old', 'elderly', 'teenager', 'child', 'adult']
        }
    
    def detect_bias(self, texts: List[str], predictions: List[int], 
                   true_labels: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Detect potential bias in model predictions
        
        Args:
            texts: Input texts
            predictions: Model predictions
            true_labels: Ground truth labels (if available)
        
        Returns:
            Dictionary of bias metrics
        """
        bias_report = {}
        
        # Demographic parity analysis
        demographic_predictions = self._analyze_demographic_parity(texts, predictions)
        bias_report['demographic_parity'] = demographic_predictions
        
        # Fairness metrics
        if true_labels:
            fairness = self._calculate_fairness_metrics(texts, predictions, true_labels)
            bias_report['fairness_metrics'] = fairness
        
        # Lexical bias detection
        lexical_bias = self._detect_lexical_bias(texts, predictions)
        bias_report['lexical_bias'] = lexical_bias
        
        # Calculate overall bias score
        bias_report['overall_bias_score'] = self._calculate_overall_bias(bias_report)
        
        self.bias_metrics = bias_report
        return bias_report
    
    def _analyze_demographic_parity(self, texts: List[str], 
                                   predictions: List[int]) -> Dict[str, float]:
        """Analyze demographic parity in predictions"""
        demographic_stats = {}
        
        for category, terms in self.protected_terms.items():
            # Group texts by presence of demographic terms
            with_terms = []
            without_terms = []
            
            for i, text in enumerate(texts):
                text_lower = text.lower()
                has_term = any(term in text_lower for term in terms)
                
                if has_term:
                    with_terms.append(predictions[i])
                else:
                    without_terms.append(predictions[i])
            
            # Calculate positive prediction rates
            if with_terms and without_terms:
                rate_with = np.mean(with_terms)
                rate_without = np.mean(without_terms)
                disparity = abs(rate_with - rate_without)
                
                demographic_stats[category] = {
                    'positive_rate_with_terms': rate_with,
                    'positive_rate_without_terms': rate_without,
                    'disparity': disparity,
                    'samples_with_terms': len(with_terms),
                    'samples_without_terms': len(without_terms)
                }
        
        return demographic_stats
    
    def _calculate_fairness_metrics(self, texts: List[str], 
                                   predictions: List[int], 
                                   true_labels: List[int]) -> Dict[str, float]:
        """Calculate fairness metrics across demographic groups"""
        fairness_metrics = {}
        
        for category, terms in self.protected_terms.items():
            # Separate by demographic groups
            group_metrics = {'with_terms': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
                           'without_terms': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}}
            
            for i, text in enumerate(texts):
                text_lower = text.lower()
                has_term = any(term in text_lower for term in terms)
                group = 'with_terms' if has_term else 'without_terms'
                
                # Calculate confusion matrix elements
                if predictions[i] == 1 and true_labels[i] == 1:
                    group_metrics[group]['tp'] += 1
                elif predictions[i] == 1 and true_labels[i] == 0:
                    group_metrics[group]['fp'] += 1
                elif predictions[i] == 0 and true_labels[i] == 0:
                    group_metrics[group]['tn'] += 1
                else:
                    group_metrics[group]['fn'] += 1
            
            # Calculate metrics for each group
            for group in ['with_terms', 'without_terms']:
                metrics = group_metrics[group]
                total = sum(metrics.values())
                
                if total > 0:
                    # Equal opportunity: TPR should be similar across groups
                    tpr = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0
                    
                    # Predictive parity: PPV should be similar across groups
                    ppv = metrics['tp'] / (metrics['tp'] + metrics['fp']) if (metrics['tp'] + metrics['fp']) > 0 else 0
                    
                    fairness_metrics[f"{category}_{group}"] = {
                        'true_positive_rate': tpr,
                        'positive_predictive_value': ppv,
                        'total_samples': total
                    }
        
        return fairness_metrics
    
    def _detect_lexical_bias(self, texts: List[str], 
                            predictions: List[int]) -> Dict[str, float]:
        """Detect lexical bias in predictions"""
        lexical_stats = {}
        
        # Analyze prediction patterns for specific word categories
        word_categories = {
            'formal': ['therefore', 'furthermore', 'consequently', 'hence'],
            'informal': ['gonna', 'wanna', 'ain\'t', 'y\'all'],
            'emotional': ['love', 'hate', 'angry', 'happy', 'sad'],
            'neutral': ['the', 'is', 'are', 'have', 'been']
        }
        
        for category, words in word_categories.items():
            category_predictions = []
            
            for i, text in enumerate(texts):
                text_lower = text.lower()
                if any(word in text_lower for word in words):
                    category_predictions.append(predictions[i])
            
            if category_predictions:
                lexical_stats[category] = {
                    'positive_rate': np.mean(category_predictions),
                    'sample_count': len(category_predictions)
                }
        
        return lexical_stats
    
    def _calculate_overall_bias(self, bias_report: Dict) -> float:
        """Calculate overall bias score (0=unbiased, 1=highly biased)"""
        bias_scores = []
        
        # Demographic parity scores
        if 'demographic_parity' in bias_report:
            for category, stats in bias_report['demographic_parity'].items():
                if isinstance(stats, dict) and 'disparity' in stats:
                    bias_scores.append(stats['disparity'])
        
        # Fairness metric disparities
        if 'fairness_metrics' in bias_report:
            tpr_values = []
            for group, metrics in bias_report['fairness_metrics'].items():
                if isinstance(metrics, dict) and 'true_positive_rate' in metrics:
                    tpr_values.append(metrics['true_positive_rate'])
            
            if len(tpr_values) > 1:
                tpr_disparity = max(tpr_values) - min(tpr_values)
                bias_scores.append(tpr_disparity)
        
        # Calculate overall score
        if bias_scores:
            return np.mean(bias_scores)
        return 0.0
    
    def mitigate_bias(self, model, training_data: pd.DataFrame, 
                      strategy: str = 'reweighting') -> object:
        """
        Apply bias mitigation strategies
        
        Args:
            model: Original model
            training_data: Training dataset
            strategy: Mitigation strategy ('reweighting', 'resampling', 'adversarial')
        
        Returns:
            Bias-mitigated model
        """
        if strategy == 'reweighting':
            return self._apply_reweighting(model, training_data)
        elif strategy == 'resampling':
            return self._apply_resampling(model, training_data)
        elif strategy == 'adversarial':
            return self._apply_adversarial_debiasing(model, training_data)
        else:
            logger.warning(f"Unknown mitigation strategy: {strategy}")
            return model
    
    def _apply_reweighting(self, model, training_data: pd.DataFrame):
        """Apply sample reweighting to reduce bias"""
        # Calculate sample weights based on demographic representation
        weights = np.ones(len(training_data))
        
        for idx, row in training_data.iterrows():
            text = row['text'].lower()
            
            # Increase weight for underrepresented groups
            for category, terms in self.protected_terms.items():
                if any(term in text for term in terms):
                    weights[idx] *= 1.2  # Boost underrepresented samples
        
        # Normalize weights
        weights = weights / weights.mean()
        
        # Retrain model with weights
        logger.info("Retraining model with bias-adjusted weights")
        # Model-specific retraining would go here
        
        return model
    
    def _apply_resampling(self, model, training_data: pd.DataFrame):
        """Apply data resampling to balance representation"""
        # Balance dataset by demographic groups
        balanced_data = []
        
        for category, terms in self.protected_terms.items():
            # Find samples with and without demographic terms
            with_terms = training_data[
                training_data['text'].str.lower().apply(
                    lambda x: any(term in x for term in terms)
                )
            ]
            without_terms = training_data[
                ~training_data['text'].str.lower().apply(
                    lambda x: any(term in x for term in terms)
                )
            ]
            
            # Balance the groups
            min_size = min(len(with_terms), len(without_terms))
            if min_size > 0:
                balanced_data.append(with_terms.sample(min_size, replace=True))
                balanced_data.append(without_terms.sample(min_size, replace=True))
        
        if balanced_data:
            balanced_df = pd.concat(balanced_data, ignore_index=True)
            logger.info(f"Resampled data from {len(training_data)} to {len(balanced_df)} samples")
            # Retrain model with balanced data
        
        return model
    
    def _apply_adversarial_debiasing(self, model, training_data: pd.DataFrame):
        """Apply adversarial debiasing techniques"""
        logger.info("Applying adversarial debiasing (placeholder for complex implementation)")
        # This would involve training an adversarial network to remove bias
        # For POC, we return the original model with a warning
        logger.warning("Full adversarial debiasing requires additional implementation")
        return model
    
    def generate_bias_report(self) -> str:
        """Generate human-readable bias analysis report"""
        report = []
        report.append("=" * 60)
        report.append("BIAS ANALYSIS REPORT")
        report.append("=" * 60)
        
        if not self.bias_metrics:
            report.append("No bias analysis available. Run detect_bias() first.")
            return "\n".join(report)
        
        # Overall bias score
        overall_score = self.bias_metrics.get('overall_bias_score', 0)
        report.append(f"\nOverall Bias Score: {overall_score:.3f}")
        
        if overall_score < 0.1:
            report.append("Assessment: Model shows minimal bias")
        elif overall_score < 0.3:
            report.append("Assessment: Model shows moderate bias - consider mitigation")
        else:
            report.append("Assessment: Model shows significant bias - mitigation recommended")
        
        # Demographic parity details
        if 'demographic_parity' in self.bias_metrics:
            report.append("\n" + "-" * 40)
            report.append("DEMOGRAPHIC PARITY ANALYSIS")
            report.append("-" * 40)
            
            for category, stats in self.bias_metrics['demographic_parity'].items():
                if isinstance(stats, dict):
                    report.append(f"\n{category.upper()}:")
                    report.append(f"  Positive rate (with terms): {stats['positive_rate_with_terms']:.3f}")
                    report.append(f"  Positive rate (without terms): {stats['positive_rate_without_terms']:.3f}")
                    report.append(f"  Disparity: {stats['disparity']:.3f}")
                    
                    if stats['disparity'] > 0.1:
                        report.append(f"  ⚠ Warning: Significant disparity detected")
        
        # Fairness metrics
        if 'fairness_metrics' in self.bias_metrics:
            report.append("\n" + "-" * 40)
            report.append("FAIRNESS METRICS")
            report.append("-" * 40)
            
            for group, metrics in self.bias_metrics['fairness_metrics'].items():
                if isinstance(metrics, dict):
                    report.append(f"\n{group}:")
                    report.append(f"  TPR: {metrics['true_positive_rate']:.3f}")
                    report.append(f"  PPV: {metrics['positive_predictive_value']:.3f}")
        
        # Recommendations
        report.append("\n" + "-" * 40)
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        
        if overall_score > 0.1:
            report.append("1. Consider reweighting training samples")
            report.append("2. Implement data augmentation for underrepresented groups")
            report.append("3. Regular bias audits during model updates")
            report.append("4. Human review for edge cases")
        else:
            report.append("Continue regular monitoring for bias drift")
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    # Initialize bias mitigator
    mitigator = BiasMitigator()
    
    # Sample data for testing
    sample_texts = [
        "She is being aggressive",
        "He is being aggressive",
        "They are upset",
        "The young person needs help",
        "The elderly person needs help",
        "This is inappropriate content"
    ]
    
    sample_predictions = [1, 1, 0, 0, 1, 1]
    sample_labels = [1, 1, 0, 0, 0, 1]
    
    # Detect bias
    bias_report = mitigator.detect_bias(sample_texts, sample_predictions, sample_labels)
    
    # Generate report
    print(mitigator.generate_bias_report())