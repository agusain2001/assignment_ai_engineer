"""
AI Safety Pipeline - Core processing engine for all safety models
"""

import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level enumeration"""
    SAFE = "safe"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class InterventionType(Enum):
    """Types of interventions"""
    NONE = "none"
    WARNING = "warning"
    CONTENT_HIDDEN = "content_hidden"
    ESCALATE_MODERATOR = "escalate_moderator"
    CRISIS_SUPPORT = "crisis_support"
    ACCOUNT_RESTRICTION = "account_restriction"


@dataclass
class SafetyAnalysisResult:
    """Result of safety analysis"""
    text: str
    timestamp: float
    risk_level: RiskLevel
    risk_score: float
    abuse_score: float
    escalation_score: float
    crisis_score: float
    content_filter_score: float
    interventions: List[InterventionType]
    explanations: Dict[str, str]
    processing_time: float
    age_appropriate: bool
    confidence: float
    
    def to_dict(self):
        """Convert to dictionary"""
        result = asdict(self)
        result['risk_level'] = self.risk_level.value
        result['interventions'] = [i.value for i in self.interventions]
        return result


class SafetyPipeline:
    """Main safety processing pipeline"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the safety pipeline"""
        self.config = self._load_config(config_path)
        self.models = {}
        self.tokenizer = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._initialize_models()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration"""
        default_config = {
            'thresholds': {
                'abuse': 0.7,
                'escalation': 0.6,
                'crisis': 0.8,
                'content_filter': 0.5
            },
            'model_weights': {
                'abuse': 0.3,
                'escalation': 0.25,
                'crisis': 0.35,
                'content_filter': 0.1
            },
            'age_categories': {
                'child': (0, 12),
                'teen': (13, 17),
                'adult': (18, 100)
            }
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    import yaml
                    loaded_config = yaml.safe_load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Could not load config: {e}")
        
        return default_config
    
    def _initialize_models(self):
        """Initialize all safety models"""
        logger.info("Initializing safety models...")
        
        # Initialize with lightweight models for POC
        try:
            # Abuse Detection Model
            self.models['abuse'] = self._create_abuse_detector()
            
            # Escalation Recognition Model
            self.models['escalation'] = self._create_escalation_detector()
            
            # Crisis Intervention Model
            self.models['crisis'] = self._create_crisis_detector()
            
            # Content Filter Model
            self.models['content_filter'] = self._create_content_filter()
            
            logger.info("All models initialized successfully")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            # Fall back to rule-based models
            self._initialize_fallback_models()
    
    def _create_abuse_detector(self):
        """Create abuse detection model"""
        try:
            return pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                device=-1  # CPU
            )
        except:
            return self._create_rule_based_abuse_detector()
    
    def _create_escalation_detector(self):
        """Create escalation pattern detector"""
        return EscalationDetector()
    
    def _create_crisis_detector(self):
        """Create crisis intervention detector"""
        return CrisisDetector()
    
    def _create_content_filter(self):
        """Create content filter for age-appropriate content"""
        return ContentFilter()
    
    def _create_rule_based_abuse_detector(self):
        """Fallback rule-based abuse detector"""
        class RuleBasedAbuseDetector:
            def __init__(self):
                self.keywords = {
                    'high': ['kill', 'die', 'hate', 'attack'],
                    'medium': ['stupid', 'idiot', 'ugly', 'dumb'],
                    'low': ['bad', 'wrong', 'terrible']
                }
            
            def __call__(self, text):
                text_lower = text.lower()
                score = 0
                for level, words in self.keywords.items():
                    for word in words:
                        if word in text_lower:
                            if level == 'high':
                                score += 0.9
                            elif level == 'medium':
                                score += 0.5
                            else:
                                score += 0.2
                
                return [{'label': 'TOXIC' if score > 0.5 else 'SAFE', 'score': min(score, 1.0)}]
        
        return RuleBasedAbuseDetector()
    
    def _initialize_fallback_models(self):
        """Initialize simple fallback models"""
        self.models['abuse'] = self._create_rule_based_abuse_detector()
        self.models['escalation'] = EscalationDetector()
        self.models['crisis'] = CrisisDetector()
        self.models['content_filter'] = ContentFilter()
    
    def analyze(
        self,
        text: str,
        user_age: Optional[int] = None,
        context_history: Optional[List[str]] = None,
        user_id: Optional[str] = None
    ) -> SafetyAnalysisResult:
        """
        Analyze text for safety concerns
        
        Args:
            text: Input text to analyze
            user_age: Age of user (for content filtering)
            context_history: Previous messages for context
            user_id: User identifier for tracking patterns
        
        Returns:
            SafetyAnalysisResult with risk assessment and interventions
        """
        start_time = time.time()
        
        # Parallel model inference
        futures = []
        with self.executor as executor:
            futures.append(executor.submit(self._detect_abuse, text))
            futures.append(executor.submit(self._detect_escalation, text, context_history))
            futures.append(executor.submit(self._detect_crisis, text))
            futures.append(executor.submit(self._filter_content, text, user_age))
        
        # Collect results
        abuse_score, abuse_explanation = futures[0].result()
        escalation_score, escalation_explanation = futures[1].result()
        crisis_score, crisis_explanation = futures[2].result()
        content_score, age_appropriate, content_explanation = futures[3].result()
        
        # Calculate combined risk score
        risk_score = self._calculate_risk_score(
            abuse_score, escalation_score, crisis_score, content_score
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(risk_score)
        
        # Generate interventions
        interventions = self._generate_interventions(
            risk_level, abuse_score, escalation_score, crisis_score, age_appropriate
        )
        
        # Compile explanations
        explanations = {
            'abuse': abuse_explanation,
            'escalation': escalation_explanation,
            'crisis': crisis_explanation,
            'content_filter': content_explanation,
            'overall': self._generate_overall_explanation(risk_level, interventions)
        }
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            abuse_score, escalation_score, crisis_score, content_score
        )
        
        processing_time = time.time() - start_time
        
        return SafetyAnalysisResult(
            text=text,
            timestamp=time.time(),
            risk_level=risk_level,
            risk_score=risk_score,
            abuse_score=abuse_score,
            escalation_score=escalation_score,
            crisis_score=crisis_score,
            content_filter_score=content_score,
            interventions=interventions,
            explanations=explanations,
            processing_time=processing_time,
            age_appropriate=age_appropriate,
            confidence=confidence
        )
    
    def _detect_abuse(self, text: str) -> Tuple[float, str]:
        """Detect abusive language"""
        try:
            result = self.models['abuse'](text)
            if result[0]['label'] == 'TOXIC':
                score = result[0]['score']
                explanation = f"Detected potentially abusive content (confidence: {score:.2f})"
            else:
                score = 1 - result[0]['score']
                explanation = "No abusive content detected"
            return score, explanation
        except Exception as e:
            logger.error(f"Abuse detection failed: {e}")
            return 0.0, "Abuse detection unavailable"
    
    def _detect_escalation(self, text: str, context: Optional[List[str]]) -> Tuple[float, str]:
        """Detect escalation patterns"""
        try:
            score = self.models['escalation'].detect(text, context)
            if score > 0.6:
                explanation = f"Conversation showing signs of escalation (score: {score:.2f})"
            else:
                explanation = "No escalation patterns detected"
            return score, explanation
        except Exception as e:
            logger.error(f"Escalation detection failed: {e}")
            return 0.0, "Escalation detection unavailable"
    
    def _detect_crisis(self, text: str) -> Tuple[float, str]:
        """Detect crisis situations"""
        try:
            score = self.models['crisis'].detect(text)
            if score > 0.7:
                explanation = f"Potential crisis situation detected (score: {score:.2f})"
            else:
                explanation = "No crisis indicators found"
            return score, explanation
        except Exception as e:
            logger.error(f"Crisis detection failed: {e}")
            return 0.0, "Crisis detection unavailable"
    
    def _filter_content(self, text: str, user_age: Optional[int]) -> Tuple[float, bool, str]:
        """Filter content for age-appropriateness"""
        try:
            score, appropriate = self.models['content_filter'].filter(text, user_age)
            if not appropriate:
                explanation = f"Content may not be age-appropriate (score: {score:.2f})"
            else:
                explanation = "Content appears age-appropriate"
            return score, appropriate, explanation
        except Exception as e:
            logger.error(f"Content filtering failed: {e}")
            return 0.0, True, "Content filtering unavailable"
    
    def _calculate_risk_score(self, abuse: float, escalation: float, 
                              crisis: float, content: float) -> float:
        """Calculate weighted risk score"""
        weights = self.config['model_weights']
        score = (
            abuse * weights['abuse'] +
            escalation * weights['escalation'] +
            crisis * weights['crisis'] +
            content * weights['content_filter']
        )
        return min(max(score, 0.0), 1.0)
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from score"""
        if risk_score < 0.2:
            return RiskLevel.SAFE
        elif risk_score < 0.4:
            return RiskLevel.LOW
        elif risk_score < 0.6:
            return RiskLevel.MODERATE
        elif risk_score < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _generate_interventions(self, risk_level: RiskLevel, abuse: float,
                                escalation: float, crisis: float, 
                                age_appropriate: bool) -> List[InterventionType]:
        """Generate appropriate interventions"""
        interventions = []
        
        if risk_level == RiskLevel.SAFE:
            interventions.append(InterventionType.NONE)
        else:
            if risk_level in [RiskLevel.LOW, RiskLevel.MODERATE]:
                interventions.append(InterventionType.WARNING)
            
            if abuse > self.config['thresholds']['abuse']:
                interventions.append(InterventionType.CONTENT_HIDDEN)
            
            if escalation > self.config['thresholds']['escalation']:
                interventions.append(InterventionType.ESCALATE_MODERATOR)
            
            if crisis > self.config['thresholds']['crisis']:
                interventions.append(InterventionType.CRISIS_SUPPORT)
            
            if not age_appropriate:
                interventions.append(InterventionType.CONTENT_HIDDEN)
            
            if risk_level == RiskLevel.CRITICAL:
                interventions.append(InterventionType.ACCOUNT_RESTRICTION)
        
        return interventions if interventions else [InterventionType.NONE]
    
    def _generate_overall_explanation(self, risk_level: RiskLevel, 
                                     interventions: List[InterventionType]) -> str:
        """Generate overall explanation"""
        if risk_level == RiskLevel.SAFE:
            return "Content appears safe with no concerns detected"
        
        explanation = f"Risk level: {risk_level.value}. "
        if InterventionType.CRISIS_SUPPORT in interventions:
            explanation += "Immediate support recommended. "
        if InterventionType.ESCALATE_MODERATOR in interventions:
            explanation += "Human moderator review suggested. "
        
        return explanation
    
    def _calculate_confidence(self, *scores: float) -> float:
        """Calculate confidence in the assessment"""
        # Higher variance = lower confidence
        variance = np.var(scores)
        confidence = 1.0 - min(variance * 2, 0.5)
        return confidence


class EscalationDetector:
    """Detect escalation patterns in conversations"""
    
    def __init__(self):
        self.escalation_indicators = [
            'getting worse', 'can\'t take', 'had enough', 
            'done with', 'give up', 'no point'
        ]
        self.intensity_modifiers = ['really', 'very', 'extremely', 'so']
    
    def detect(self, text: str, context: Optional[List[str]] = None) -> float:
        """Detect escalation patterns"""
        score = 0.0
        text_lower = text.lower()
        
        # Check for escalation indicators
        for indicator in self.escalation_indicators:
            if indicator in text_lower:
                score += 0.3
        
        # Check for intensity modifiers
        for modifier in self.intensity_modifiers:
            if modifier in text_lower:
                score += 0.1
        
        # Check for all caps (shouting)
        if text.isupper() and len(text) > 5:
            score += 0.2
        
        # Check context for pattern
        if context:
            negative_trend = sum(1 for msg in context[-3:] 
                                if any(ind in msg.lower() 
                                      for ind in self.escalation_indicators))
            score += negative_trend * 0.15
        
        return min(score, 1.0)


class CrisisDetector:
    """Detect crisis situations requiring intervention"""
    
    def __init__(self):
        self.crisis_keywords = [
            'suicide', 'kill myself', 'end it all', 'not worth living',
            'self harm', 'hurt myself', 'overdose', 'jump off'
        ]
        self.support_needed = [
            'help me', 'need someone', 'alone', 'no one cares',
            'cant go on', 'hopeless', 'worthless'
        ]
    
    def detect(self, text: str) -> float:
        """Detect crisis indicators"""
        score = 0.0
        text_lower = text.lower()
        
        # Critical keywords
        for keyword in self.crisis_keywords:
            if keyword in text_lower:
                score += 0.8
        
        # Support indicators
        for indicator in self.support_needed:
            if indicator in text_lower:
                score += 0.3
        
        return min(score, 1.0)


class ContentFilter:
    """Filter content for age-appropriateness"""
    
    def __init__(self):
        self.mature_topics = [
            'violence', 'drug', 'alcohol', 'sexual', 
            'explicit', 'gore', 'death'
        ]
        self.mild_concerns = [
            'dating', 'relationship', 'kiss', 'love'
        ]
    
    def filter(self, text: str, user_age: Optional[int]) -> Tuple[float, bool]:
        """Filter content based on age"""
        if user_age is None:
            return 0.0, True
        
        score = 0.0
        text_lower = text.lower()
        
        # Check mature topics
        for topic in self.mature_topics:
            if topic in text_lower:
                score += 0.4
        
        # Check mild concerns for younger users
        if user_age < 13:
            for topic in self.mild_concerns:
                if topic in text_lower:
                    score += 0.2
        
        # Determine appropriateness
        if user_age < 13:
            appropriate = score < 0.3
        elif user_age < 18:
            appropriate = score < 0.6
        else:
            appropriate = True
        
        return min(score, 1.0), appropriate