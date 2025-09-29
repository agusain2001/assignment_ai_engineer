"""
AI Safety POC - Complete Demo and Testing Script
Run this script to demonstrate all functionality
"""

import os
import sys
import time
import json
import random
from datetime import datetime
from typing import List, Dict
import argparse
from colorama import Fore, Back, Style, init

# Initialize colorama for colored output
init(autoreset=True)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import safety pipeline
from src.safety_pipeline import SafetyPipeline, RiskLevel, InterventionType


class SafetyDemo:
    """Interactive demo for AI Safety Models POC"""
    
    def __init__(self):
        self.pipeline = SafetyPipeline()
        self.conversation_history = []
        self.user_age = None
        self.demo_messages = self._load_demo_messages()
        
    def _load_demo_messages(self) -> Dict[str, List[str]]:
        """Load demo messages for testing"""
        return {
            'safe': [
                "Hello, how are you today?",
                "Can you help me with my homework?",
                "What's the weather like?",
                "I love learning new things!",
                "Thanks for your help!"
            ],
            'abuse': [
                "You're so stupid and useless",
                "I hate everything about this",
                "This is completely worthless",
                "You're an idiot if you believe that",
                "Nobody wants to hear from you"
            ],
            'escalation': [
                "This is getting worse and worse!",
                "I CAN'T TAKE THIS ANYMORE!",
                "Everything is falling apart",
                "I've had enough of this nonsense",
                "THIS IS OUT OF CONTROL NOW!"
            ],
            'crisis': [
                "I don't see the point anymore",
                "Nobody would care if I disappeared",
                "Life feels hopeless right now",
                "I can't go on like this",
                "Everything hurts so much"
            ],
            'inappropriate_content': [
                "Let's talk about adult topics",
                "Check out this violent content",
                "Want to see something inappropriate?",
                "This contains mature themes",
                "Not suitable for children"
            ]
        }
    
    def print_banner(self):
        """Print demo banner"""
        print(Fore.CYAN + "="*70)
        print(Fore.CYAN + " " * 20 + "AI SAFETY MODELS POC DEMO")
        print(Fore.CYAN + " " * 15 + "Real-time Safety Analysis System")
        print(Fore.CYAN + "="*70)
        print()
    
    def get_risk_color(self, risk_level: RiskLevel) -> str:
        """Get color based on risk level"""
        colors = {
            RiskLevel.SAFE: Fore.GREEN,
            RiskLevel.LOW: Fore.YELLOW,
            RiskLevel.MODERATE: Fore.LIGHTYELLOW_EX,
            RiskLevel.HIGH: Fore.LIGHTRED_EX,
            RiskLevel.CRITICAL: Fore.RED + Back.WHITE
        }
        return colors.get(risk_level, Fore.WHITE)
    
    def display_analysis(self, result):
        """Display analysis results with colors"""
        print("\n" + "="*50)
        print(Fore.WHITE + "ANALYSIS RESULTS")
        print("="*50)
        
        # Risk level with color
        risk_color = self.get_risk_color(result.risk_level)
        print(f"\n{Fore.WHITE}Risk Level: {risk_color}{result.risk_level.value.upper()}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Overall Risk Score: {self._get_score_color(result.risk_score)}{result.risk_score:.2f}{Style.RESET_ALL}")
        
        # Individual scores
        print(f"\n{Fore.WHITE}Component Scores:")
        print(f"  • Abuse Detection:     {self._get_score_color(result.abuse_score)}{result.abuse_score:.2f}{Style.RESET_ALL}")
        print(f"  • Escalation Pattern:  {self._get_score_color(result.escalation_score)}{result.escalation_score:.2f}{Style.RESET_ALL}")
        print(f"  • Crisis Indicators:   {self._get_score_color(result.crisis_score)}{result.crisis_score:.2f}{Style.RESET_ALL}")
        print(f"  • Content Filter:      {self._get_score_color(result.content_filter_score)}{result.content_filter_score:.2f}{Style.RESET_ALL}")
        
        # Age appropriateness
        if self.user_age:
            age_text = "✓ Age Appropriate" if result.age_appropriate else "✗ Not Age Appropriate"
            age_color = Fore.GREEN if result.age_appropriate else Fore.RED
            print(f"\n{Fore.WHITE}Age Check ({self.user_age} years): {age_color}{age_text}{Style.RESET_ALL}")
        
        # Interventions
        if result.interventions and result.interventions[0] != InterventionType.NONE:
            print(f"\n{Fore.WHITE}Recommended Interventions:")
            for intervention in result.interventions:
                icon, text = self._get_intervention_display(intervention)
                print(f"  {icon} {text}")
        else:
            print(f"\n{Fore.GREEN}✓ No interventions needed{Style.RESET_ALL}")
        
        # Confidence and processing time
        print(f"\n{Fore.WHITE}Confidence: {result.confidence:.1%}")
        print(f"Processing Time: {result.processing_time*1000:.1f}ms")
        
        # Explanations
        if result.explanations.get('overall'):
            print(f"\n{Fore.WHITE}Summary: {Fore.LIGHTWHITE_EX}{result.explanations['overall']}{Style.RESET_ALL}")
    
    def _get_score_color(self, score: float) -> str:
        """Get color based on score value"""
        if score < 0.3:
            return Fore.GREEN
        elif score < 0.6:
            return Fore.YELLOW
        elif score < 0.8:
            return Fore.LIGHTRED_EX
        else:
            return Fore.RED
    
    def _get_intervention_display(self, intervention: InterventionType) -> tuple:
        """Get display icon and text for intervention"""
        displays = {
            InterventionType.WARNING: ("⚠️ ", "Issue warning to user"),
            InterventionType.CONTENT_HIDDEN: ("🚫", "Hide inappropriate content"),
            InterventionType.ESCALATE_MODERATOR: ("👤", "Escalate to human moderator"),
            InterventionType.CRISIS_SUPPORT: ("🆘", "Activate crisis support resources"),
            InterventionType.ACCOUNT_RESTRICTION: ("🔒", "Apply temporary account restrictions")
        }
        return displays.get(intervention, ("📋", intervention.value))
    
    def run_interactive_mode(self):
        """Run interactive chat mode"""
        self.print_banner()
        
        print(Fore.YELLOW + "Interactive Mode - Type messages to analyze")
        print(Fore.YELLOW + "Commands: 'quit', 'age <number>', 'clear', 'demo'")
        print("-"*50)
        
        while True:
            try:
                # Get input
                message = input(f"\n{Fore.CYAN}Enter message> {Style.RESET_ALL}").strip()
                
                # Handle commands
                if message.lower() == 'quit':
                    print(Fore.GREEN + "Goodbye!")
                    break
                elif message.lower().startswith('age'):
                    try:
                        self.user_age = int(message.split()[1])
                        print(f"{Fore.GREEN}User age set to {self.user_age}")
                    except:
                        print(f"{Fore.RED}Invalid age format. Use: age <number>")
                    continue
                elif message.lower() == 'clear':
                    self.conversation_history = []
                    print(f"{Fore.GREEN}Conversation history cleared")
                    continue
                elif message.lower() == 'demo':
                    self.run_demo_scenarios()
                    continue
                elif not message:
                    continue
                
                # Analyze message
                print(f"\n{Fore.CYAN}Analyzing message...{Style.RESET_ALL}")
                result = self.pipeline.analyze(
                    text=message,
                    user_age=self.user_age,
                    context_history=self.conversation_history[-5:]
                )
                
                # Add to history
                self.conversation_history.append(message)
                
                # Display results
                self.display_analysis(result)
                
            except KeyboardInterrupt:
                print(f"\n{Fore.GREEN}Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
    
    def run_demo_scenarios(self):
        """Run predefined demo scenarios"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}RUNNING DEMO SCENARIOS")
        print(f"{Fore.CYAN}{'='*60}")
        
        scenarios = [
            ("Safe Conversation", 'safe', None),
            ("Abuse Detection", 'abuse', None),
            ("Escalation Pattern", 'escalation', None),
            ("Crisis Situation", 'crisis', None),
            ("Child User (Age 10)", 'inappropriate_content', 10),
            ("Teen User (Age 15)", 'inappropriate_content', 15),
            ("Adult User (Age 25)", 'inappropriate_content', 25)
        ]
        
        for scenario_name, category, age in scenarios:
            print(f"\n{Fore.YELLOW}Scenario: {scenario_name}")
            print(f"{Fore.YELLOW}{'-'*40}")
            
            if age:
                self.user_age = age
                print(f"User Age: {age}")
            
            # Get random message from category
            message = random.choice(self.demo_messages[category])
            print(f"Message: \"{message}\"")
            
            # Analyze
            result = self.pipeline.analyze(
                text=message,
                user_age=self.user_age,
                context_history=self.conversation_history[-3:] if self.conversation_history else None
            )
            
            # Show brief results
            risk_color = self.get_risk_color(result.risk_level)
            print(f"Result: {risk_color}{result.risk_level.value.upper()}{Style.RESET_ALL} (Score: {result.risk_score:.2f})")
            
            if result.interventions and result.interventions[0] != InterventionType.NONE:
                print("Interventions:", ", ".join([i.value for i in result.interventions]))
            
            time.sleep(0.5)  # Brief pause between scenarios
        
        print(f"\n{Fore.GREEN}Demo scenarios completed!{Style.RESET_ALL}")
    
    def run_batch_analysis(self, messages: List[str]):
        """Run batch analysis on multiple messages"""
        print(f"\n{Fore.CYAN}Batch Analysis - Processing {len(messages)} messages")
        print("="*50)
        
        results = []
        start_time = time.time()
        
        for i, message in enumerate(messages, 1):
            print(f"Processing message {i}/{len(messages)}...", end='\r')
            result = self.pipeline.analyze(text=message, user_age=self.user_age)
            results.append(result)
        
        total_time = time.time() - start_time
        
        # Statistics
        risk_counts = {level: 0 for level in RiskLevel}
        for result in results:
            risk_counts[result.risk_level] += 1
        
        print(f"\n\n{Fore.WHITE}Batch Analysis Complete!")
        print(f"Total Messages: {len(messages)}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Avg Time/Message: {total_time/len(messages)*1000:.1f}ms")
        
        print(f"\n{Fore.WHITE}Risk Distribution:")
        for level, count in risk_counts.items():
            if count > 0:
                percentage = count/len(messages)*100
                color = self.get_risk_color(level)
                print(f"  {color}{level.value.upper()}: {count} ({percentage:.1f}%){Style.RESET_ALL}")
        
        # Find highest risk messages
        high_risk = [r for r in results if r.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        if high_risk:
            print(f"\n{Fore.RED}⚠️  {len(high_risk)} high-risk messages detected{Style.RESET_ALL}")
    
    def run_performance_test(self):
        """Run performance benchmarks"""
        print(f"\n{Fore.CYAN}PERFORMANCE BENCHMARKING")
        print("="*50)
        
        test_sizes = [10, 50, 100]
        
        for size in test_sizes:
            messages = []
            for category in self.demo_messages.values():
                messages.extend(category * (size // len(self.demo_messages)))
            
            messages = messages[:size]
            
            print(f"\n{Fore.WHITE}Testing with {size} messages...")
            
            start_time = time.time()
            for msg in messages:
                self.pipeline.analyze(msg)
            
            elapsed = time.time() - start_time
            avg_time = elapsed / size * 1000
            throughput = size / elapsed
            
            print(f"  Total Time: {elapsed:.2f}s")
            print(f"  Avg Latency: {avg_time:.1f}ms")
            print(f"  Throughput: {throughput:.1f} msg/s")
        
        print(f"\n{Fore.GREEN}✓ Performance test complete{Style.RESET_ALL}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='AI Safety Models POC Demo')
    parser.add_argument('--mode', choices=['interactive', 'demo', 'batch', 'performance'], 
                       default='interactive', help='Demo mode to run')
    parser.add_argument('--input', type=str, help='Single message to analyze')
    parser.add_argument('--age', type=int, help='User age for content filtering')
    parser.add_argument('--file', type=str, help='File with messages for batch analysis')
    
    args = parser.parse_args()
    
    demo = SafetyDemo()
    
    if args.age:
        demo.user_age = args.age
    
    if args.input:
        # Single message analysis
        print(f"\n{Fore.CYAN}Analyzing single message...")
        result = demo.pipeline.analyze(text=args.input, user_age=args.age)
        demo.display_analysis(result)
    elif args.file:
        # Batch file analysis
        with open(args.file, 'r') as f:
            messages = [line.strip() for line in f if line.strip()]
        demo.run_batch_analysis(messages)
    elif args.mode == 'demo':
        demo.run_demo_scenarios()
    elif args.mode == 'batch':
        # Use demo messages for batch
        all_messages = []
        for msgs in demo.demo_messages.values():
            all_messages.extend(msgs)
        demo.run_batch_analysis(all_messages)
    elif args.mode == 'performance':
        demo.run_performance_test()
    else:
        demo.run_interactive_mode()
    
    print(f"\n{Fore.CYAN}Thank you for using AI Safety Models POC!{Style.RESET_ALL}")


if __name__ == '__main__':
    main()
