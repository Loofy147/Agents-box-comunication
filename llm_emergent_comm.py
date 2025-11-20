"""
Production-Ready LLM Emergent Communication System
===================================================
Enables multiple LLM agents to develop efficient communication protocols
through reinforcement learning with bandwidth costs.

Requirements:
    pip install anthropic numpy pandas matplotlib tqdm
    
Set environment variable:
    export ANTHROPIC_API_KEY='your-key-here'
"""

import os
import json
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION & DATA MODELS
# ==========================================

@dataclass
class CommunicationTask:
    """A task that requires agent coordination"""
    task_id: str
    concept: str  # What needs to be communicated
    context: Dict  # Additional context
    urgency: float  # 0.0 to 1.0
    complexity: float  # 0.0 to 1.0
    
@dataclass
class Message:
    """A message sent between agents"""
    sender_id: str
    receiver_id: str
    content: str  # The actual signal
    tokens_used: int
    timestamp: float
    task_id: str
    
@dataclass
class CommunicationResult:
    """Result of a communication attempt"""
    task_id: str
    success: bool
    sender_message: str
    receiver_interpretation: str
    tokens_used: int
    latency: float
    reward: float
    timestamp: float
    
@dataclass
class ProtocolEntry:
    """Learned protocol mapping"""
    concept: str
    signal: str
    success_count: int
    failure_count: int
    avg_tokens: float
    last_used: float

class Config:
    """System configuration"""
    # LLM Settings
    MODEL = "claude-3-5-sonnet-20240620"
    MAX_TOKENS = 1000
    TEMPERATURE = 0.7
    
    # Economic Settings
    TOKEN_COST = 0.001  # $ per token
    LATENCY_PENALTY = 0.1  # $ per second
    SUCCESS_REWARD = 10.0  # $
    FAILURE_PENALTY = -1.0  # $
    
    # Learning Settings
    EPSILON_START = 0.5  # Exploration rate
    EPSILON_DECAY = 0.995
    EPSILON_MIN = 0.05
    LEARNING_RATE = 0.1
    
    # Protocol Settings
    MAX_SIGNAL_TOKENS = 50  # Enforce brevity
    PROTOCOL_MEMORY_SIZE = 100
    
    # Task Settings
    CONCEPTS = [
        "CRITICAL_BUG_FOUND",
        "DEPLOY_READY",
        "CODE_REVIEW_NEEDED",
        "SECURITY_VULNERABILITY",
        "PERFORMANCE_DEGRADATION",
        "DEPENDENCY_UPDATE",
        "MERGE_CONFLICT",
        "TEST_FAILURE"
    ]

# ==========================================
# 2. LLM AGENT WITH PROTOCOL LEARNING
# ==========================================

class LLMAgent:
    """
    An LLM-powered agent that learns efficient communication protocols
    """
    def __init__(self, agent_id: str, role: str, api_key: str):
        self.agent_id = agent_id
        self.role = role  # 'sender' or 'receiver'
        self.api_key = api_key
        
        # Learning state
        self.epsilon = Config.EPSILON_START
        self.protocol_memory: Dict[str, List[ProtocolEntry]] = defaultdict(list)
        self.performance_history = deque(maxlen=100)
        
        # Economics
        self.total_cost = 0.0
        self.total_reward = 0.0
        
        # API setup (using Anthropic Claude)
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> Tuple[str, int]:
        """
        Call the LLM API and return response + token count
        """
        try:
            response = self.client.messages.create(
                model=Config.MODEL,
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            content = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens
            
            return content, tokens
            
        except Exception as e:
            print(f"LLM API Error: {e}")
            return f"[ERROR: {str(e)}]", 0
    
    def generate_signal(self, task: CommunicationTask) -> Tuple[str, int]:
        """
        Sender: Generate a signal to communicate a concept
        """
        # Check if we should explore or exploit
        if np.random.random() < self.epsilon:
            mode = "EXPLORE"
        else:
            mode = "EXPLOIT"
        
        # Build context from learned protocols
        protocol_context = self._get_protocol_context(task.concept)
        
        system_prompt = f"""You are Agent {self.agent_id}, a SENDER in a multi-agent system.

Your goal: Communicate the concept '{task.concept}' to another agent as EFFICIENTLY as possible.

CONSTRAINTS:
- Maximum {Config.MAX_SIGNAL_TOKENS} tokens
- Each token costs ${Config.TOKEN_COST}
- You will be rewarded for successful communication
- You will be penalized for verbosity

MODE: {mode}
- EXPLORE: Try new, creative encodings
- EXPLOIT: Use proven efficient patterns

LEARNED PROTOCOLS:
{protocol_context}

TASK CONTEXT:
- Urgency: {task.urgency}
- Complexity: {task.complexity}
- Additional: {json.dumps(task.context, indent=2)}

OUTPUT: A concise signal that conveys '{task.concept}'. Be creative and efficient."""

        user_prompt = f"Generate signal for: {task.concept}"
        
        signal, tokens = self._call_llm(system_prompt, user_prompt)
        
        # Track cost
        self.total_cost += tokens * Config.TOKEN_COST
        
        return signal.strip(), tokens
    
    def interpret_signal(self, signal: str, task: CommunicationTask) -> Tuple[str, int]:
        """
        Receiver: Interpret a signal and guess the concept
        """
        protocol_context = self._get_protocol_context(None)  # All protocols
        
        system_prompt = f"""You are Agent {self.agent_id}, a RECEIVER in a multi-agent system.

Your goal: Interpret the signal you receive and identify which concept is being communicated.

POSSIBLE CONCEPTS:
{json.dumps(Config.CONCEPTS, indent=2)}

LEARNED SIGNAL PATTERNS:
{protocol_context}

TASK CONTEXT:
- Urgency: {task.urgency}
- Complexity: {task.complexity}

OUTPUT: Only the concept name (e.g., "CRITICAL_BUG_FOUND"). Nothing else."""

        user_prompt = f"Interpret this signal: {signal}"
        
        interpretation, tokens = self._call_llm(system_prompt, user_prompt)
        
        # Track cost
        self.total_cost += tokens * Config.TOKEN_COST
        
        return interpretation.strip(), tokens
    
    def _get_protocol_context(self, concept: Optional[str]) -> str:
        """Build context string from learned protocols"""
        if concept and concept in self.protocol_memory:
            entries = sorted(
                self.protocol_memory[concept],
                key=lambda x: x.success_count / max(x.failure_count, 1),
                reverse=True
            )[:5]
            
            if entries:
                return "\n".join([
                    f"- Signal: '{e.signal[:50]}...' | Success: {e.success_count} | Tokens: {e.avg_tokens:.1f}"
                    for e in entries
                ])
        
        # Return all protocols for receiver
        context_lines = []
        for c, entries in list(self.protocol_memory.items())[:5]:
            best = max(entries, key=lambda x: x.success_count) if entries else None
            if best:
                context_lines.append(f"- {c}: '{best.signal[:30]}...'")
        
        return "\n".join(context_lines) if context_lines else "No protocols learned yet."
    
    def update_protocol(self, concept: str, signal: str, success: bool, tokens: int):
        """
        Update learned protocols based on outcome
        """
        # Find existing entry or create new
        existing = None
        for entry in self.protocol_memory[concept]:
            if entry.signal == signal:
                existing = entry
                break
        
        if existing:
            if success:
                existing.success_count += 1
            else:
                existing.failure_count += 1
            # Update rolling average
            existing.avg_tokens = (existing.avg_tokens * 0.9) + (tokens * 0.1)
            existing.last_used = time.time()
        else:
            # Create new protocol entry
            entry = ProtocolEntry(
                concept=concept,
                signal=signal,
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                avg_tokens=float(tokens),
                last_used=time.time()
            )
            self.protocol_memory[concept].append(entry)
        
        # Prune old protocols
        if len(self.protocol_memory[concept]) > Config.PROTOCOL_MEMORY_SIZE:
            self.protocol_memory[concept] = sorted(
                self.protocol_memory[concept],
                key=lambda x: x.success_count / max(x.failure_count, 1),
                reverse=True
            )[:Config.PROTOCOL_MEMORY_SIZE]
        
        # Decay exploration
        self.epsilon = max(Config.EPSILON_MIN, self.epsilon * Config.EPSILON_DECAY)
    
    def record_performance(self, result: CommunicationResult):
        """Track performance metrics"""
        self.performance_history.append({
            'success': result.success,
            'tokens': result.tokens_used,
            'reward': result.reward,
            'timestamp': result.timestamp
        })
        
        self.total_reward += result.reward

# ==========================================
# 3. COMMUNICATION ENVIRONMENT
# ==========================================

class CommunicationEnvironment:
    """
    Manages multi-agent communication tasks and learning
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        
        # Initialize agents
        self.sender = LLMAgent("Sender_001", "sender", api_key)
        self.receiver = LLMAgent("Receiver_001", "receiver", api_key)
        
        # Tracking
        self.results: List[CommunicationResult] = []
        self.episode = 0
        
    def generate_task(self) -> CommunicationTask:
        """Generate a random communication task"""
        concept = np.random.choice(Config.CONCEPTS)
        
        return CommunicationTask(
            task_id=f"task_{self.episode}",
            concept=concept,
            context={
                "priority": np.random.choice(["low", "medium", "high"]),
                "affected_systems": np.random.randint(1, 5)
            },
            urgency=np.random.random(),
            complexity=np.random.random()
        )
    
    def run_episode(self) -> CommunicationResult:
        """
        Execute one communication episode
        """
        # 1. Generate task
        task = self.generate_task()
        start_time = time.time()
        
        # 2. Sender generates signal
        signal, sender_tokens = self.sender.generate_signal(task)
        
        # 3. Receiver interprets signal
        interpretation, receiver_tokens = self.receiver.interpret_signal(signal, task)
        
        # 4. Calculate outcome
        latency = time.time() - start_time
        total_tokens = sender_tokens + receiver_tokens
        
        # Check if interpretation matches concept (fuzzy match)
        success = task.concept.lower() in interpretation.lower()
        
        # 5. Calculate reward
        reward = self._calculate_reward(success, total_tokens, latency)
        
        # 6. Update both agents' protocols
        self.sender.update_protocol(task.concept, signal, success, sender_tokens)
        self.receiver.update_protocol(task.concept, signal, success, receiver_tokens)
        
        # 7. Record results
        result = CommunicationResult(
            task_id=task.task_id,
            success=success,
            sender_message=signal,
            receiver_interpretation=interpretation,
            tokens_used=total_tokens,
            latency=latency,
            reward=reward,
            timestamp=time.time()
        )
        
        self.sender.record_performance(result)
        self.receiver.record_performance(result)
        self.results.append(result)
        
        self.episode += 1
        
        return result
    
    def _calculate_reward(self, success: bool, tokens: int, latency: float) -> float:
        """
        Calculate reward based on success and efficiency
        """
        if success:
            reward = Config.SUCCESS_REWARD
            # Bonus for efficiency
            reward -= (tokens * Config.TOKEN_COST)
            reward -= (latency * Config.LATENCY_PENALTY)
        else:
            reward = Config.FAILURE_PENALTY
        
        return reward
    
    def run_training(self, num_episodes: int = 100, verbose: bool = True):
        """
        Run full training loop
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"EMERGENT COMMUNICATION TRAINING")
            print(f"{'='*60}\n")
            print(f"Model: {Config.MODEL}")
            print(f"Episodes: {num_episodes}")
            print(f"Concepts: {len(Config.CONCEPTS)}")
            print(f"\nStarting training...\n")
        
        pbar = tqdm(range(num_episodes), desc="Training") if verbose else range(num_episodes)
        
        for i in pbar:
            result = self.run_episode()
            
            if verbose and i % 10 == 0:
                recent_success = np.mean([
                    r.success for r in self.results[-10:]
                ]) * 100
                
                avg_tokens = np.mean([
                    r.tokens_used for r in self.results[-10:]
                ])
                
                pbar.set_postfix({
                    'Success': f'{recent_success:.1f}%',
                    'Tokens': f'{avg_tokens:.0f}',
                    'Epsilon': f'{self.sender.epsilon:.3f}'
                })
        
        if verbose:
            self._print_summary()
    
    def _print_summary(self):
        """Print training summary"""
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}\n")
        
        total_success = np.mean([r.success for r in self.results]) * 100
        total_tokens = np.sum([r.tokens_used for r in self.results])
        total_cost = self.sender.total_cost + self.receiver.total_cost
        
        print(f"Overall Success Rate: {total_success:.1f}%")
        print(f"Total Tokens Used: {total_tokens:,}")
        print(f"Total Cost: ${total_cost:.2f}")
        print(f"Sender ROI: ${self.sender.total_reward - self.sender.total_cost:.2f}")
        print(f"Receiver ROI: ${self.receiver.total_reward - self.receiver.total_cost:.2f}")
        
        # Show learned protocols
        print(f"\n{'='*60}")
        print(f"EMERGENT PROTOCOLS")
        print(f"{'='*60}\n")
        
        for concept in Config.CONCEPTS[:5]:  # Show top 5
            if concept in self.sender.protocol_memory:
                best = max(
                    self.sender.protocol_memory[concept],
                    key=lambda x: x.success_count,
                    default=None
                )
                if best and best.success_count > 0:
                    print(f"{concept}:")
                    print(f"  Signal: '{best.signal[:60]}...'")
                    print(f"  Success: {best.success_count}/{best.success_count + best.failure_count}")
                    print(f"  Avg Tokens: {best.avg_tokens:.1f}\n")
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Visualize training results
        """
        if not self.results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('LLM Emergent Communication Training Results', fontsize=16)
        
        # Moving average helper
        def moving_avg(data, window=10):
            return pd.Series(data).rolling(window=window, min_periods=1).mean()
        
        # 1. Success Rate
        success_data = [1 if r.success else 0 for r in self.results]
        axes[0, 0].plot(moving_avg(success_data), color='green', linewidth=2)
        axes[0, 0].set_title('Success Rate Over Time')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Token Efficiency
        token_data = [r.tokens_used for r in self.results]
        axes[0, 1].plot(moving_avg(token_data), color='blue', linewidth=2)
        axes[0, 1].set_title('Tokens Used Per Communication')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Tokens')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Reward Accumulation
        reward_data = [r.reward for r in self.results]
        cumulative_reward = np.cumsum(reward_data)
        axes[1, 0].plot(cumulative_reward, color='purple', linewidth=2)
        axes[1, 0].set_title('Cumulative Reward')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Total Reward ($)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Exploration Rate
        epsilon_data = [Config.EPSILON_START * (Config.EPSILON_DECAY ** i) 
                       for i in range(len(self.results))]
        axes[1, 1].plot(epsilon_data, color='orange', linewidth=2)
        axes[1, 1].set_title('Exploration Rate (Epsilon)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def export_protocols(self, filepath: str):
        """Export learned protocols to JSON"""
        protocols = {
            'sender': {},
            'receiver': {},
            'metadata': {
                'episodes': self.episode,
                'timestamp': datetime.now().isoformat(),
                'model': Config.MODEL
            }
        }
        
        for concept, entries in self.sender.protocol_memory.items():
            protocols['sender'][concept] = [
                {
                    'signal': e.signal,
                    'success_rate': e.success_count / max(e.success_count + e.failure_count, 1),
                    'avg_tokens': e.avg_tokens
                }
                for e in sorted(entries, key=lambda x: x.success_count, reverse=True)[:3]
            ]
        
        with open(filepath, 'w') as f:
            json.dump(protocols, f, indent=2)
        
        print(f"Protocols exported to: {filepath}")

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def main():
    """
    Main execution function
    """
    # Get API key from environment
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        print("export ANTHROPIC_API_KEY='your-key-here'")
        return
    
    # Initialize environment
    env = CommunicationEnvironment(api_key)
    
    # Run training
    env.run_training(num_episodes=50, verbose=True)
    
    # Visualize results
    env.plot_results(save_path='emergent_comm_results.png')
    
    # Export learned protocols
    env.export_protocols('learned_protocols.json')
    
    print("\n✅ Training complete!")
    print("📊 Check 'emergent_comm_results.png' for visualizations")
    print("📝 Check 'learned_protocols.json' for learned protocols")

if __name__ == "__main__":
    main()
