
from typing import List, Dict, Any, Optional
from tenacity import retry, wait_exponential, stop_after_attempt
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

from agent.utils import (
    _parse_json_response,
    merge_module_tasks
)


class Clarify:
    def __init__(self, client: OpenAI, base_prompt: str, model_name: str, semantic_model_dir, threshold):
        self._base_prompt = base_prompt
        self._model_name = model_name
        self._semantic_model = SentenceTransformer(semantic_model_dir)
        self.threshold = threshold
        self.PROMPT = "Represent this sentence for semantic similarity comparison: "

        # Zero-width/invisible character pool
        self._ZERO_WIDTH_CHARS = [
            "\u200B",  # Zero Width Space
            "\u200C",  # Zero Width Non-Joiner
            "\u200D",  # Zero Width Joiner
            "\u2060",  # Word Joiner
            "\uFEFF",  # Zero Width No-Break Space (BOM)
        ]
        self._CONTROL_CHARS = [
            "\u0000",  # Null
            "\u0001",  # Start of Header
            "\u0002",  # Start of Text
            "\u0003",  # End of Text
        ]
        self._client = client

    def _query_llm(self, prompt: str, _temperature, num_outputs: int = 1,
                   base_prompt="You are a helpful coding assistant.",
                   use_perturbation: bool = True):

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=5)
        )
        def _single_query(call_index):
            if use_perturbation:
                # Fixed high perturbation sampling parameters (seed changes with call_index)
                temperature = 1.5  # High temperature → more randomness
                top_p = 0.75  # Low top_p → more extreme sampling
                seed_value = 4242 + call_index  # Deterministic but different each time

                # Deterministic zero-width suffix (based on call_index)
                num_zw = 5
                # Cycle through predefined zero-width characters for reproducibility
                zw_suffix = "".join(
                    self._ZERO_WIDTH_CHARS[i % len(self._ZERO_WIDTH_CHARS)]
                    for i in range(num_zw)
                )

                # Deterministic control character prefix (100% added, fixed content)
                num_ctrl = 3
                ctrl_prefix = "".join(
                    self._CONTROL_CHARS[i % len(self._CONTROL_CHARS)]
                    for i in range(num_ctrl)
                )

                # Fixed trailing whitespace
                trailing_spaces = " " * 3
                trailing_newlines = "\n" * 2

                perturbed_prompt = ctrl_prefix + prompt + zw_suffix + trailing_spaces + trailing_newlines

            else:
                # No perturbation: use original parameters
                temperature = _temperature
                top_p = 1.0
                seed_value = 42 + call_index if _temperature > 0 else 42
                perturbed_prompt = prompt

            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": base_prompt},
                    {"role": "user", "content": perturbed_prompt}
                ],
                extra_body={
                    "enable_thinking": False,
                    "thinking": {"type": "disabled"},
                    "return_usage": True
                },
                temperature=temperature,
                top_p=top_p,
                seed=seed_value
            )
            return response.choices[0].message.content

        results = []
        for i in range(num_outputs):
            try:
                result = _single_query(i)
                results.append(result)
            except Exception as e:
                print(f"Query {i + 1} failed after maximum retries: {str(e)}")
        return results

    def _compare_commands(self, task1, task2):
        """
        Compare semantic similarity of two task lists, treating each command as a whole without # chunking.
        """
        if len(task1) != len(task2):
            return 0.0

        modules1 = [item.get('module', '') for item in task1]
        modules2 = [item.get('module', '') for item in task2]

        if modules1 != modules2:
            return 0.0

        total_score = 0.0
        num_modules = len(task1)

        for item1, item2 in zip(task1, task2):
            cmd1 = item1.get('command', '').strip() or "[Empty command]"
            cmd2 = item2.get('command', '').strip() or "[Empty command]"

            # Treat the entire command as a single sentence without chunking
            sentences = [self.PROMPT + cmd1, self.PROMPT + cmd2]
            embeddings = self._semantic_model.encode(sentences, normalize_embeddings=True)
            raw_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

            total_score += raw_sim

        avg_score = total_score / num_modules if num_modules > 0 else 0.0

        return avg_score

    def _run_semantic_model(self, task_lists):
        """Run semantic model to check consistency"""
        if len(task_lists) < 2:
            return True  # Single solution is considered consistent by default

        tasks = []
        for task in task_lists:
            content = extract_task_steps(task)
            content = _parse_json_response(content)
            content = merge_module_tasks(content)
            tasks.append(content)

        n = len(tasks)
        if n < 2:
            return True

        scores = []
        for i in range(n):
            for j in range(i + 1, n):
                score = self._compare_commands(tasks[i], tasks[j])
                scores.append(score)

        min_score = min(scores) if scores else 1.0
        inconsistency_score = 1 - min_score
        return inconsistency_score <= self.threshold

    def _analyze_semantic_consistency(self, current_query: str, solutions: List[str]) -> Dict[str, Any]:
        """Analyze semantic consistency and generate user-oriented clarification questions"""
        # Step 1: Quick consistency check using semantic model
        if self._run_semantic_model(solutions):
            # Consistent: No need to call LLM, return standard consistent result directly
            return {
                "consistent": True,
                "summary": "All solutions are highly consistent in task understanding.",
                "differences": [],
                "clarification_question": ""
            }

        analyze_prompt = f"""
            Please analyze whether the following {len(solutions)} solutions are consistent in terms of user intent and task understanding.

            Focus on:
            - Are different assumptions being made about unspecified user requirements?
            - Are there fundamental disagreements?

            Requirements:
            - If there are discrepancies in understanding, generate a **concise, natural, user-oriented** clarification question based on these discrepancies to help the user clarify ambiguous needs.
            - The clarification question should focus on the one or two most critical ambiguous points and avoid being verbose.

            Output strictly in JSON format with the following fields:
            {{
                "consistent": true/false,
                "summary": "The core intent commonly expressed by all solutions (one sentence)",
                "differences": ["Understanding difference 1 caused by ambiguous user requirements", ...],
                "clarification_question": "[Specific question]?"
            }}

            User's original task: {current_query}

            Proposed solutions:
            """ + "\n\n".join(f"Solution {i + 1}:\n{s}" for i, s in enumerate(solutions)) + f"\n{'-' * 50}"

        response = self._query_llm(
            prompt=analyze_prompt,
            num_outputs=1,
            base_prompt="",
            _temperature=0,
            use_perturbation=False
        )

        return self._parse_analysis_response(response[0] if response else "", "semantic_consistency")

    def _analyze_violations(self, current_query: str, solutions: List[str]) -> Dict[str, Any]:
        """Check compliance issues in multiple solutions and generate user-oriented clarification questions"""
        solution_list = []
        for solution in solutions:
            steps = extract_task_steps(solution)
            task = _parse_json_response(steps)
            if task:
                task = merge_module_tasks(task)
                solution_list.append(str(task))

        solutions = solution_list

        rules_context = """
        Microscope Operation and System Module Notes:
            3D samples must have Z-stack scanning parameters set (e.g., organoids, cell clusters).
            There is no independent hardware-based autofocus; focusing must be implemented via software based on image feedback.
            Focusing should be performed using the final exposure and light source brightness settings. If imaging parameters are already determined before a focusing operation, re-adjustment is unnecessary; exposure, brightness, and focusing can be executed independently and do not need to be performed consecutively or as a bundled operation.
            Parameter settings:
                Fluorescence: filter must match the channel, brightness=0, high exposure;
                Brightfield: filter=brightfield, low exposure, auto-brightness enabled.
            Focusing strategy:
                Switching fluorescence channels does not require refocusing;
                When switching between brightfield and fluorescence modes, refocusing must be performed.
                After changing objectives, repositioning, brightness adjustment, and refocusing are required.
        """

        analyze_prompt = f"""You are a microscope operation compliance analysis assistant.

        Please review the high-level plan below for ambiguity, omissions, or non-compliant operations according to the following notes:
        {rules_context}

        Requirements:
        Focus on the following:
        - Whether the plan can fulfill the task requirements.
        - If historical information exists, incorporate it. Any changes to objective magnification or fluorescence must be explicitly stated, specifying the exact fluorescence channel/color or objective magnification (e.g., 4x, 10x, 20x, 40x, 60x).
        - If all proposed plans comply with the rules, return has_violation=false and an empty string for clarification_question.
        - Analyze whether the parameter settings in the plan make unwarranted assumptions about the user's original request.
        - Generate a concise, natural, user-oriented clarification question.

        Important: Your response must be a **pure JSON object**. Do not include any additional content, explanations, Markdown (e.g., ```json), line breaks, or extra text. Output only the following JSON format:
        {{"has_violation": true/false, "clarification_question": "Your question"}}
        User's original task: {current_query}
        User-provided plans:
        {'-' * 50}
        """ + "\n\n".join(f"Plan {i + 1}:\n{s}" for i, s in enumerate(solutions)) + f"\n{'-' * 50}"

        response = self._query_llm(
            prompt=analyze_prompt,
            num_outputs=1,
            base_prompt="",
            _temperature=0,
            use_perturbation=False
        )

        return self._parse_violation_response(response[0] if response else "")

    def _parse_analysis_response(self, response_text: str, analysis_type: str) -> Dict[str, Any]:
        """Parse analysis response with unified error handling"""
        try:
            result = _parse_json_response(response_text)
            required_fields = ["consistent", "summary", "differences", "clarification_question"]

            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing field: {field}")

            assert isinstance(result["consistent"], bool)
            assert isinstance(result["summary"], str)
            assert isinstance(result["differences"], list)
            assert isinstance(result["clarification_question"], str)

            return result
        except Exception as e:

            return self._get_default_analysis_response(analysis_type)

    def _parse_violation_response(self, response_text: str) -> Dict[str, Any]:
        """Parse violation analysis response"""
        try:
            result = _parse_json_response(response_text)
            has_violation = result.get("has_violation", False)
            clarification_question = result.get("clarification_question", "").strip() if has_violation else ""
            return {
                "has_violation": has_violation,
                "clarification_question": clarification_question
            }
        except Exception as e:
            return {
                "has_violation": True,
                "clarification_question": "To ensure the feasibility of the plan, please clarify: Do you wish to perform any operations beyond the functionality of the standard microscope module? For example, manual focusing, custom Z-scanning, etc.?"
            }

    def _get_default_analysis_response(self, analysis_type: str) -> Dict[str, Any]:
        """Get default analysis response"""
        if analysis_type == "semantic_consistency":
            return {
                "consistent": False,
                "summary": "Failed to parse consistency result",
                "differences": ["LLM returned invalid format or missing required fields"],
                "clarification_question": "To better assist you, please provide detailed information about your experimental requirements, such as: sample type (2D/3D), fluorescence channels used, objective magnification, etc."
            }
        return {}

    @staticmethod
    def _get_user_feedback(suggested_question: str) -> str:
        """Get user feedback"""
        print(f"\n[System Question] {suggested_question}")
        print(f"\n[Robot] {suggested_question}")
        req = input("Your response: ").strip()
        print(f'[User]{req}')
        return req

    def generate_code_solutions(self, user_input: str, num_solutions: int = 3,
                                max_clarification_rounds: int = 10) -> str:
        """Main workflow for generating code solutions"""
        current_query = user_input
        _query = user_input
        round_count = 0

        while round_count < max_clarification_rounds:
            round_count += 1

            exploration_prompt = (
                f"{current_query}\n\n"
                "Note: The user's request may contain unspecified parameters.\n"
                "Please proactively make different reasonable assumptions about these ambiguous points to generate diverse plans.\n"
                "Maintain the required output format and do not add any content beyond the specified format.\n"
            )

            solutions = self._query_llm(
                prompt=exploration_prompt,
                num_outputs=num_solutions,
                base_prompt=self._base_prompt,
                _temperature=1.0,
                use_perturbation=True)

            if not solutions:
                return ""

            # Step 1: Analyze semantic consistency
            consistency_result = self._analyze_semantic_consistency(_query, solutions)

            # Step 2: Check operational compliance
            violation_result = self._analyze_violations(_query, solutions)

            # Scenario 1: Semantically consistent and no violations → Success, return result
            if consistency_result["consistent"] and not violation_result["has_violation"]:
                return solutions[0]

            # Scenario 2: Semantically inconsistent → Must clarify with user
            if not consistency_result["consistent"]:

                base_diff = consistency_result['differences'][0] if consistency_result[
                    'differences'] else "Please further explain your experimental setup"
                suggested_question = f"To ensure we accurately understand your request, please clarify: {base_diff}?"
                additional_input = self._get_user_feedback(suggested_question)

                # Update context (only for next round of semantic understanding)
                _query = (
                    f"{_query}\n\n"
                    f"System clarification question: {suggested_question}\n"
                    f"User's additional input: {additional_input}"
                )
                current_query = (
                    f"{_query}\n"
                    "Please generate a solution based on the clarified requirements above."
                )
                continue  # Proceed to next round

            # Scenario 3: Semantically consistent but with violations → Auto-integrate suggestions and retry
            if consistency_result["consistent"] and violation_result["has_violation"]:
                suggestion_text = violation_result["clarification_question"]

                # Construct clarification question: Ask user to confirm adjustment or stick to original intent
                clarification_question = (
                    f"{suggestion_text}\n"
                )
                additional_input = self._get_user_feedback(clarification_question)

                # Add user feedback to context and regenerate in next round
                _query = (
                    f"{_query}\n\n"
                    f"System noted a compliance issue: {suggestion_text}\n"
                    f"User response: {additional_input}"
                )
                current_query = (
                    f"{_query}\n"
                    "Please generate a more accurate solution based on the user's clarification and feedback above."
                )

                continue

        return solutions[0] if solutions else "Failed to generate valid solution, please retry or provide more details."

def extract_task_steps(input_text: str) -> str:
    match = re.search(r'<Task steps>(.*?)</Task steps>', input_text, re.DOTALL)
    return match.group(1).strip() if match else ""