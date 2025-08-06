from typing import Dict, Any, Callable, Optional, List, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import json
import time
from enum import Enum


class ExecutionStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    RETRY = "retry"
    SKIPPED = "skipped"


@dataclass
class PromptNode:
    """Represents a prompt node in the chain"""
    prompt_id: str
    template: str
    output_parser: Callable[[str], Any]
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class Edge:
    """Represents an edge between prompt nodes"""
    from_prompt_id: str
    to_prompt_id: str
    condition_func: Optional[Callable[[Any], bool]] = None


@dataclass
class ExecutionResult:
    """Result of executing a prompt"""
    prompt_id: str
    status: ExecutionStatus
    output: Any
    error: Optional[str] = None
    attempts: int = 1


class PromptChainOrchestrator:
    """Orchestrates the execution of connected prompts in a directed graph"""

    def __init__(self, llm_simulator: Optional[Callable] = None):
        self.prompts: Dict[str, PromptNode] = {}
        self.edges: Dict[str, List[Edge]] = defaultdict(list)
        self.execution_history: List[ExecutionResult] = []
        self.llm_simulator = llm_simulator or self._default_llm_simulator

    def add_prompt(self, prompt_id: str, template: str,
                   output_parser: Callable[[str], Any],
                   max_retries: int = 3, retry_delay: float = 1.0) -> None:
        """Add a prompt node to the chain"""
        if prompt_id in self.prompts:
            raise ValueError(f"Prompt with id '{prompt_id}' already exists")

        self.prompts[prompt_id] = PromptNode(
            prompt_id=prompt_id,
            template=template,
            output_parser=output_parser,
            max_retries=max_retries,
            retry_delay=retry_delay
        )

    def add_edge(self, from_prompt_id: str, to_prompt_id: str,
                 condition_func: Optional[Callable[[Any], bool]] = None) -> None:
        """Connect two prompts with optional conditional logic"""
        if from_prompt_id not in self.prompts:
            raise ValueError(f"Source prompt '{from_prompt_id}' not found")
        if to_prompt_id not in self.prompts:
            raise ValueError(f"Target prompt '{to_prompt_id}' not found")

        edge = Edge(from_prompt_id, to_prompt_id, condition_func)
        self.edges[from_prompt_id].append(edge)

    def execute_chain(self, input_data: Dict[str, Any],
                      start_prompt_id: str) -> Dict[str, Any]:
        """Execute the prompt chain starting from specified prompt"""
        if start_prompt_id not in self.prompts:
            raise ValueError(f"Start prompt '{start_prompt_id}' not found")

        self.execution_history = []
        context = {"initial_input": input_data}
        executed = set()
        execution_queue = []

        # Track which prompts are ready to execute
        ready_prompts = {start_prompt_id}
        pending_prompts = set()

        while ready_prompts or execution_queue:
            # Move ready prompts to execution queue
            execution_queue.extend(ready_prompts)
            ready_prompts.clear()

            # Execute prompts in queue
            while execution_queue:
                current_prompt_id = execution_queue.pop(0)

                if current_prompt_id in executed:
                    continue

                # Check if all dependencies are satisfied
                if not self._dependencies_satisfied(current_prompt_id, executed, context):
                    pending_prompts.add(current_prompt_id)
                    continue

                # Execute current prompt
                result = self._execute_prompt(current_prompt_id, context)
                self.execution_history.append(result)
                executed.add(current_prompt_id)

                if result.status == ExecutionStatus.SUCCESS:
                    # Update context with result
                    context[current_prompt_id] = result.output

                    # Find next prompts that should be executed
                    next_prompts = self._get_next_prompts(current_prompt_id, result.output)

                    # Add next prompts to ready set
                    for next_prompt in next_prompts:
                        if next_prompt not in executed:
                            ready_prompts.add(next_prompt)

                    # Check if any pending prompts are now ready
                    newly_ready = []
                    for pending in pending_prompts:
                        if self._dependencies_satisfied(pending, executed, context):
                            newly_ready.append(pending)

                    for prompt in newly_ready:
                        pending_prompts.remove(prompt)
                        ready_prompts.add(prompt)

                elif result.status == ExecutionStatus.FAILED:
                    print(f"Prompt '{current_prompt_id}' failed after {result.attempts} attempts")
                    # Continue with other branches if possible

        return {
            "final_context": context,
            "execution_history": self.execution_history,
            "executed_prompts": list(executed),
            "execution_order": [r.prompt_id for r in self.execution_history if r.status == ExecutionStatus.SUCCESS]
        }

    def _dependencies_satisfied(self, prompt_id: str, executed: Set[str],
                                context: Dict[str, Any]) -> bool:
        """Check if all dependencies for a prompt are satisfied"""
        # Find all prompts that have edges leading to this prompt
        dependencies = []
        for from_id, edges in self.edges.items():
            for edge in edges:
                if edge.to_prompt_id == prompt_id:
                    # Check if the condition would be satisfied
                    if edge.condition_func is None:
                        dependencies.append(from_id)
                    elif from_id in context:
                        if edge.condition_func(context[from_id]):
                            dependencies.append(from_id)

        # All dependencies must be executed
        return all(dep in executed for dep in dependencies)

    def _execute_prompt(self, prompt_id: str, context: Dict[str, Any]) -> ExecutionResult:
        """Execute a single prompt with retry logic"""
        prompt_node = self.prompts[prompt_id]
        attempts = 0
        last_error = None

        while attempts < prompt_node.max_retries:
            attempts += 1

            try:
                # Format the prompt template with context
                formatted_prompt = self._format_template(prompt_node.template, context)

                # Simulate LLM call
                raw_response = self.llm_simulator(formatted_prompt, prompt_id)

                # Parse the response
                parsed_output = prompt_node.output_parser(raw_response)

                return ExecutionResult(
                    prompt_id=prompt_id,
                    status=ExecutionStatus.SUCCESS,
                    output=parsed_output,
                    attempts=attempts
                )

            except Exception as e:
                last_error = str(e)
                if attempts < prompt_node.max_retries:
                    time.sleep(prompt_node.retry_delay)

        return ExecutionResult(
            prompt_id=prompt_id,
            status=ExecutionStatus.FAILED,
            output=None,
            error=last_error,
            attempts=attempts
        )

    def _get_next_prompts(self, current_prompt_id: str,
                          current_output: Any) -> List[str]:
        """Determine which prompts to execute next based on conditions"""
        next_prompts = []

        for edge in self.edges.get(current_prompt_id, []):
            # Check if condition is met (if any)
            if edge.condition_func is None or edge.condition_func(current_output):
                next_prompts.append(edge.to_prompt_id)

        return next_prompts

    def _format_template(self, template: str, context: Dict[str, Any]) -> str:
        """Format template with context variables"""
        try:
            # Only include available context keys
            available_context = {k: v for k, v in context.items() if v is not None}
            return template.format(**available_context)
        except KeyError as e:
            # Return template as-is if keys are missing
            return template

    def _default_llm_simulator(self, prompt: str, prompt_id: str) -> str:
        """Default LLM simulator for testing"""
        # Simple rule-based responses for demonstration
        responses = {
            "classify": "technical" if "error" in prompt.lower() else "general",
            "analyze": {"severity": "high", "category": "system"},
            "generate": "Based on the analysis, here is the solution...",
            "summarize": "Summary: Task completed successfully",
            "validate": "valid"
        }

        for key, response in responses.items():
            if key in prompt_id.lower():
                return json.dumps(response) if isinstance(response, dict) else response

        return "Default response"

    def visualize_chain(self) -> str:
        """Generate a simple text visualization of the prompt chain"""
        lines = ["Prompt Chain Structure:"]
        lines.append("-" * 50)

        for prompt_id in self.prompts:
            lines.append(f"[{prompt_id}]")
            for edge in self.edges.get(prompt_id, []):
                condition = " (conditional)" if edge.condition_func else ""
                lines.append(f"  └─> [{edge.to_prompt_id}]{condition}")

        return "\n".join(lines)


# Example usage and test scenarios
def example_usage():
    """Demonstrate the PromptChainOrchestrator with a practical example"""

    # Create orchestrator
    orchestrator = PromptChainOrchestrator()

    # Define output parsers
    def parse_classification(response: str) -> str:
        return response.strip().lower()

    def parse_analysis(response: str) -> Dict[str, Any]:
        try:
            return json.loads(response)
        except:
            return {"severity": "unknown", "category": "unknown"}

    def parse_generation(response: str) -> str:
        return response.strip()

    def parse_validation(response: str) -> bool:
        return response.strip().lower() == "valid"

    # Add prompts to the chain
    orchestrator.add_prompt(
        "classify_input",
        "Classify the following input: {initial_input}",
        parse_classification
    )

    orchestrator.add_prompt(
        "technical_analysis",
        "Perform technical analysis on: {initial_input}",
        parse_analysis
    )

    orchestrator.add_prompt(
        "general_analysis",
        "Perform general analysis on: {initial_input}",
        parse_analysis
    )

    orchestrator.add_prompt(
        "generate_response",
        "Generate response based on analysis: {technical_analysis} or {general_analysis}",
        parse_generation
    )

    orchestrator.add_prompt(
        "validate_response",
        "Validate the generated response: {generate_response}",
        parse_validation
    )

    orchestrator.add_prompt(
        "final_summary",
        "Create final summary of the process",
        parse_generation
    )

    # Define conditions
    def is_technical(classification: str) -> bool:
        return classification == "technical"

    def is_general(classification: str) -> bool:
        return classification == "general"

    def is_valid(validation: bool) -> bool:
        return validation

    # Connect prompts with edges
    orchestrator.add_edge("classify_input", "technical_analysis", is_technical)
    orchestrator.add_edge("classify_input", "general_analysis", is_general)
    orchestrator.add_edge("technical_analysis", "generate_response")
    orchestrator.add_edge("general_analysis", "generate_response")
    orchestrator.add_edge("generate_response", "validate_response")
    orchestrator.add_edge("validate_response", "final_summary", is_valid)

    # Visualize the chain
    print(orchestrator.visualize_chain())
    print("\n" + "=" * 50 + "\n")

    # Execute the chain
    result = orchestrator.execute_chain(
        {"user_query": "System error in module X"},
        "classify_input"
    )

    # Display results
    print("Execution Results:")
    print("-" * 50)
    for execution in result["execution_history"]:
        print(f"Prompt: {execution.prompt_id}")
        print(f"Status: {execution.status.value}")
        print(f"Output: {execution.output}")
        print(f"Attempts: {execution.attempts}")
        print("-" * 30)

    print(f"\nExecution order: {result['execution_order']}")
    print(f"All executed prompts: {result['executed_prompts']}")

    return orchestrator


# Advanced example with error handling
def advanced_example():
    """Demonstrate advanced features including error handling and complex conditions"""

    # Custom LLM simulator with occasional failures
    def unreliable_llm_simulator(prompt: str, prompt_id: str) -> str:
        import random

        # Simulate occasional failures
        if random.random() < 0.2:  # 20% failure rate
            raise Exception("LLM API timeout")

        # Return appropriate responses
        if "risk_assessment" in prompt_id:
            return json.dumps({"risk_level": random.choice(["low", "medium", "high"])})
        elif "mitigation" in prompt_id:
            return "Implement security patches and monitoring"
        else:
            return "Processed successfully"

    orchestrator = PromptChainOrchestrator(llm_simulator=unreliable_llm_simulator)

    # Add prompts with different retry configurations
    orchestrator.add_prompt(
        "risk_assessment",
        "Assess risk level for: {initial_input}",
        lambda x: json.loads(x),
        max_retries=5,
        retry_delay=0.5
    )

    orchestrator.add_prompt(
        "high_risk_mitigation",
        "Generate mitigation plan for high risk scenario",
        lambda x: x,
        max_retries=3
    )

    orchestrator.add_prompt(
        "standard_procedure",
        "Apply standard security procedures",
        lambda x: x,
        max_retries=2
    )

    orchestrator.add_prompt(
        "final_report",
        "Generate final security report",
        lambda x: x
    )

    # Complex condition based on risk level
    def is_high_risk(assessment: Dict[str, Any]) -> bool:
        return assessment.get("risk_level") == "high"

    def is_not_high_risk(assessment: Dict[str, Any]) -> bool:
        return assessment.get("risk_level") != "high"

    # Build the chain
    orchestrator.add_edge("risk_assessment", "high_risk_mitigation", is_high_risk)
    orchestrator.add_edge("risk_assessment", "standard_procedure", is_not_high_risk)
    orchestrator.add_edge("high_risk_mitigation", "final_report")
    orchestrator.add_edge("standard_procedure", "final_report")

    print("Advanced Chain Structure:")
    print(orchestrator.visualize_chain())
    print("\n" + "=" * 50 + "\n")

    # Execute with error handling demonstration
    result = orchestrator.execute_chain(
        {"security_event": "Unauthorized access attempt"},
        "risk_assessment"
    )

    print("Advanced Execution Results:")
    print("-" * 50)
    for execution in result["execution_history"]:
        print(f"Prompt: {execution.prompt_id}")
        print(f"Status: {execution.status.value}")
        print(f"Output: {execution.output}")
        print(f"Attempts: {execution.attempts}")
        if execution.error:
            print(f"Error: {execution.error}")
        print("-" * 30)

    print(f"\nExecution order: {result['execution_order']}")

    return result


if __name__ == "__main__":
    # Run the example
    orchestrator = example_usage()

    print("\n" + "=" * 50 + "\n")
    print("Advanced Example with Error Handling:")
    print("=" * 50)
    advanced_result = advanced_example()
