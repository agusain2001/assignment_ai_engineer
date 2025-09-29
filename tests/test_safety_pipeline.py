"""
Complete Testing Suite for AI Safety Models POC
"""

import unittest
import sys
import os
import json
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.safety_pipeline import (
    SafetyPipeline, RiskLevel, InterventionType,
    EscalationDetector, CrisisDetector, ContentFilter
)


class TestSafetyPipeline(unittest.TestCase):
    """Test cases for SafetyPipeline"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.pipeline = SafetyPipeline()
    
    def test_pipeline_initialization(self):
        """Test pipeline initializes correctly"""
        self.assertIsNotNone(self.pipeline)
        self.assertIsNotNone(self.pipeline.models)
        self.assertTrue(len(self.pipeline.models) > 0)
    
    def test_safe_message_analysis(self):
        """Test analysis of safe messages"""
        safe_messages = [
            "Hello, how are you today?",
            "Thanks for your help!",
            "Have a great day!"
        ]
        
        for message in safe_messages:
            result = self.pipeline.analyze(message)
            self.assertIsNotNone(result)
            self.assertLessEqual(result.risk_score, 0.3)
            self.assertIn(result.risk_level, [RiskLevel.SAFE, RiskLevel.LOW])
    
    def test_abusive_message_detection(self):
        """Test detection of abusive content"""
        abusive_messages = [
            "You're stupid and worthless",
            "I hate you",
            "You're an idiot"
        ]
        
        for message in abusive_messages:
            result = self.pipeline.analyze(message)
            self.assertIsNotNone(result)
            self.assertGreater(result.abuse_score, 0.5)
            self.assertNotEqual(result.risk_level, RiskLevel.SAFE)
    
    def test_escalation_detection(self):
        """Test escalation pattern detection"""
        escalating_messages = [
            "This is getting worse",
            "I CAN'T TAKE THIS ANYMORE",
            "STOP NOW!"
        ]
        
        for i, message in enumerate(escalating_messages):
            context = escalating_messages[:i] if i > 0 else None
            result = self.pipeline.analyze(message, context_history=context)
            
            if i > 1:  # Pattern should be detected after multiple messages
                self.assertGreater(result.escalation_score, 0.3)
    
    def test_crisis_detection(self):
        """Test crisis situation detection"""
        crisis_messages = [
            "I want to end it all",
            "Life isn't worth living",
            "Nobody would care if I disappeared"
        ]
        
        for message in crisis_messages:
            result = self.pipeline.analyze(message)
            self.assertGreater(result.crisis_score, 0.7)
            self.assertIn(InterventionType.CRISIS_SUPPORT, result.interventions)
    
    def test_content_filtering(self):
        """Test age-appropriate content filtering"""
        inappropriate_content = "This contains violent and explicit content"
        
        # Test for child
        result_child = self.pipeline.analyze(inappropriate_content, user_age=10)
        self.assertFalse(result_child.age_appropriate)
        
        # Test for adult
        result_adult = self.pipeline.analyze(inappropriate_content, user_age=25)
        # Adults should have less restriction
        self.assertLessEqual(result_adult.content_filter_score, result_child.content_filter_score)
    
    def test_intervention_generation(self):
        """Test appropriate intervention generation"""
        # Safe message - no interventions
        safe_result = self.pipeline.analyze("Have a nice day!")
        self.assertIn(InterventionType.NONE, safe_result.interventions)
        
        # High risk message - multiple interventions
        high_risk_result = self.pipeline.analyze("I hate you and want to hurt myself")
        self.assertGreater(len(high_risk_result.interventions), 1)
        self.assertNotIn(InterventionType.NONE, high_risk_result.interventions)
    
    def test_performance_metrics(self):
        """Test performance requirements"""
        message = "Test message for performance"
        
        # Measure latency
        start_time = time.time()
        result = self.pipeline.analyze(message)
        elapsed = time.time() - start_time
        
        # Should be under 100ms for single message
        self.assertLess(elapsed, 0.1)
        self.assertIsNotNone(result.processing_time)
        self.assertGreater(result.confidence, 0)
    
    def test_risk_level_determination(self):
        """Test risk level categorization"""
        test_cases = [
            (0.1, RiskLevel.SAFE),
            (0.3, RiskLevel.LOW),
            (0.5, RiskLevel.MODERATE),
            (0.7, RiskLevel.HIGH),
            (0.9, RiskLevel.CRITICAL)
        ]
        
        for score, expected_level in test_cases:
            level = self.pipeline._determine_risk_level(score)
            self.assertEqual(level, expected_level)
    
    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        # Similar scores = high confidence
        high_conf = self.pipeline._calculate_confidence(0.8, 0.8, 0.8, 0.8)
        self.assertGreater(high_conf, 0.8)
        
        # Varied scores = lower confidence
        low_conf = self.pipeline._calculate_confidence(0.1, 0.9, 0.5, 0.3)
        self.assertLess(low_conf, 0.7)


class TestEscalationDetector(unittest.TestCase):
    """Test cases for EscalationDetector"""
    
    def setUp(self):
        self.detector = EscalationDetector()
    
    def test_escalation_indicators(self):
        """Test detection of escalation indicators"""
        escalating = "I can't take this anymore"
        score = self.detector.detect(escalating)
        self.assertGreater(score, 0)
    
    def test_intensity_modifiers(self):
        """Test detection of intensity modifiers"""
        intense = "This is REALLY getting bad"
        score = self.detector.detect(intense)
        self.assertGreater(score, 0)
    
    def test_caps_detection(self):
        """Test all caps detection"""
        shouting = "STOP DOING THAT RIGHT NOW"
        score = self.detector.detect(shouting)
        self.assertGreater(score, 0.2)
    
    def test_context_pattern(self):
        """Test context-based escalation"""
        context = ["getting worse", "can't handle this", "had enough"]
        current = "I'm done with this"
        score = self.detector.detect(current, context)
        self.assertGreater(score, 0.5)


class TestCrisisDetector(unittest.TestCase):
    """Test cases for CrisisDetector"""
    
    def setUp(self):
        self.detector = CrisisDetector()
    
    def test_crisis_keywords(self):
        """Test detection of crisis keywords"""
        crisis = "I want to end it all"
        score = self.detector.detect(crisis)
        self.assertGreater(score, 0.7)
    
    def test_support_indicators(self):
        """Test detection of support need indicators"""
        support = "I feel so alone and hopeless"
        score = self.detector.detect(support)
        self.assertGreater(score, 0.3)
    
    def test_false_positives(self):
        """Test avoiding false positives"""
        safe = "I need help with my homework"
        score = self.detector.detect(safe)
        self.assertLess(score, 0.5)


class TestContentFilter(unittest.TestCase):
    """Test cases for ContentFilter"""
    
    def setUp(self):
        self.filter = ContentFilter()
    
    def test_mature_content_detection(self):
        """Test detection of mature content"""
        mature = "This contains violence and gore"
        score, appropriate = self.filter.filter(mature, user_age=10)
        self.assertGreater(score, 0)
        self.assertFalse(appropriate)
    
    def test_age_based_filtering(self):
        """Test age-appropriate filtering"""
        mild = "Let's talk about dating"
        
        # Should be inappropriate for young children
        score_child, appropriate_child = self.filter.filter(mild, user_age=10)
        self.assertFalse(appropriate_child)
        
        # Should be appropriate for teens
        score_teen, appropriate_teen = self.filter.filter(mild, user_age=16)
        self.assertTrue(appropriate_teen)
    
    def test_no_age_specified(self):
        """Test behavior when no age is specified"""
        content = "Random content"
        score, appropriate = self.filter.filter(content, user_age=None)
        self.assertEqual(score, 0.0)
        self.assertTrue(appropriate)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    @classmethod
    def setUpClass(cls):
        cls.pipeline = SafetyPipeline()
    
    def test_end_to_end_safe_flow(self):
        """Test end-to-end flow for safe content"""
        messages = [
            "Hello, how are you?",
            "I'm doing well, thanks!",
            "What's the weather like?"
        ]
        
        for msg in messages:
            result = self.pipeline.analyze(msg, user_age=15)
            self.assertIsNotNone(result)
            self.assertEqual(result.text, msg)
            self.assertLess(result.risk_score, 0.3)
            self.assertTrue(result.age_appropriate)
    
    def test_end_to_end_crisis_flow(self):
        """Test end-to-end flow for crisis situation"""
        crisis_msg = "I don't want to live anymore"
        result = self.pipeline.analyze(crisis_msg)
        
        self.assertGreater(result.risk_score, 0.7)
        self.assertIn(result.risk_level, [RiskLevel.HIGH, RiskLevel.CRITICAL])
        self.assertIn(InterventionType.CRISIS_SUPPORT, result.interventions)
        self.assertIn("crisis", result.explanations)
    
    def test_conversation_context_tracking(self):
        """Test conversation context influences analysis"""
        messages = [
            "Things are getting bad",
            "I can't handle this",
            "Everything is falling apart"
        ]
        
        context = []
        for msg in messages:
            result = self.pipeline.analyze(msg, context_history=context)
            context.append(msg)
            
            # Escalation should increase with context
            if len(context) > 2:
                self.assertGreater(result.escalation_score, 0.4)
    
    def test_multi_risk_detection(self):
        """Test detection of multiple risk types"""
        complex_msg = "You're an idiot and I want to hurt myself"
        result = self.pipeline.analyze(complex_msg)
        
        # Should detect both abuse and crisis
        self.assertGreater(result.abuse_score, 0.5)
        self.assertGreater(result.crisis_score, 0.5)
        self.assertGreater(len(result.interventions), 2)


class TestPerformance(unittest.TestCase):
    """Performance and load tests"""
    
    @classmethod
    def setUpClass(cls):
        cls.pipeline = SafetyPipeline()
    
    def test_latency_requirements(self):
        """Test latency meets requirements"""
        messages = ["Test message"] * 100
        
        latencies = []
        for msg in messages:
            start = time.time()
            self.pipeline.analyze(msg)
            latencies.append(time.time() - start)
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        # Average should be under 50ms
        self.assertLess(avg_latency, 0.05)
        # 95th percentile should be under 100ms
        self.assertLess(p95_latency, 0.1)
    
    def test_throughput(self):
        """Test system throughput"""
        messages = ["Test message"] * 1000
        
        start = time.time()
        for msg in messages:
            self.pipeline.analyze(msg)
        elapsed = time.time() - start
        
        throughput = len(messages) / elapsed
        
        # Should handle at least 20 messages per second
        self.assertGreater(throughput, 20)
    
    def test_memory_usage(self):
        """Test memory usage remains stable"""
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Initial memory
        gc.collect()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process many messages
        for i in range(1000):
            self.pipeline.analyze(f"Test message {i}")
        
        # Final memory
        gc.collect()
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory increase should be minimal (less than 100MB)
        mem_increase = mem_after - mem_before
        self.assertLess(mem_increase, 100)


def run_all_tests():
    """Run all test suites"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSafetyPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestEscalationDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestCrisisDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestContentFilter))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
