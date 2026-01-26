import requests
import time
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Configure extensive logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import LLM judge for semantic verification
try:
    from tests.llm_judge import LLMJudge
except ImportError:
    from llm_judge import LLMJudge


@dataclass
class TestResult:
    success: bool
    model: str
    test_case: str
    response_time: float
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    passed: bool = False


@dataclass
class ModelTestResults:
    model: str
    model_id: str
    total_tests: int
    passed: int
    failed: int
    average_response_time: float
    supports_tool_calling: bool
    results: List[TestResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class LLMToolCallingTester:
    def __init__(
        self,
        base_url: str = "http://localhost:8087/v1",
        use_llm_judge: bool = True,
        judge_model: str = "gemini-2.5-flash",
        db_manager: Optional[Any] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.models_url = f"{self.base_url}/models"
        self.chat_url = f"{self.base_url}/chat/completions"
        self.models: List[Dict[str, Any]] = []
        self.metrics: Dict[str, ModelTestResults] = {}
        self.db_manager = db_manager

        # Initialize LLM judge for semantic verification
        self.use_llm_judge = use_llm_judge
        if use_llm_judge:
            self.judge = LLMJudge(base_url=base_url, judge_model=judge_model)
            logger.info(f"✓ LLM Judge initialized: {judge_model}")
        else:
            self.judge = None
            logger.info("✓ Using exact matching (LLM judge disabled)")

        if self.db_manager:
            logger.info("✓ Database tracking enabled")

    def _fetch_models(self) -> List[Dict[str, Any]]:
        logger.info("=" * 80)
        logger.info("FETCHING AVAILABLE MODELS")
        logger.info("=" * 80)
        logger.info(f"Request URL: {self.models_url}")

        try:
            response = requests.get(self.models_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            self.models = data.get("data", [])

            logger.info(f"✓ Successfully fetched {len(self.models)} models")
            logger.info("-" * 80)
            for i, model in enumerate(self.models, 1):
                model_id = model.get("id", "unknown")
                owned_by = model.get("owned_by", "unknown")
                logger.info(f"  {i:2d}. {model_id:40s} (by {owned_by})")
            logger.info("=" * 80)

            return self.models
        except requests.RequestException as e:
            logger.error(f"✗ Failed to fetch models: {e}")
            raise Exception(f"Failed to fetch models: {e}")

    def test_model_tool_calling(
        self, model: str, test_cases: List[Dict[str, Any]]
    ) -> ModelTestResults:
        logger.info("")
        logger.info("╔" + "=" * 78 + "╗")
        logger.info(f"║ {'TESTING MODEL: ' + model + '':^76} ║")
        logger.info("╚" + "=" * 78 + "╝")
        logger.info(f"Total test cases: {len(test_cases)}")
        logger.info("")

        results = []
        total_response_time = 0
        passed = 0
        failed = 0
        errors = []
        supports_tool_calling = False

        for i, test_case in enumerate(test_cases, 1):
            logger.info(
                f"┌─ TEST {i}/{len(test_cases)}: {test_case.get('name', 'unnamed')}"
            )
            logger.info("│")

            test_result = self._execute_test(model, test_case)
            results.append(test_result)
            total_response_time += test_result.response_time

            if test_result.success:
                supports_tool_calling = True
                if test_result.passed:
                    passed += 1
                    logger.info(
                        f"│ ✓ PASSED (response time: {test_result.response_time:.3f}s)"
                    )
                else:
                    failed += 1
                    logger.info(f"│ ✗ FAILED - Tool calls didn't match expected")
            else:
                failed += 1
                if test_result.error:
                    errors.append(test_result.error)
                    logger.info(f"│ ✗ ERROR: {test_result.error}")

            logger.info("└" + "─" * 78)
            logger.info("")

        average_response_time = (
            total_response_time / len(test_cases) if test_cases else 0
        )

        model_results = ModelTestResults(
            model=model,
            model_id=model,
            total_tests=len(test_cases),
            passed=passed,
            failed=failed,
            average_response_time=average_response_time,
            supports_tool_calling=supports_tool_calling,
            results=results,
            errors=errors,
        )

        self.metrics[model] = model_results

        # Print summary for this model
        logger.info(f"{'=' * 80}")
        logger.info(f"MODEL {model} SUMMARY:")
        logger.info(f"  Total Tests: {len(test_cases)}")
        logger.info(f"  Passed: {passed}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Success Rate: {(passed / len(test_cases) * 100):.1f}%")
        logger.info(f"  Avg Response Time: {average_response_time:.3f}s")
        logger.info(
            f"  Supports Tool Calling: {'✓ YES' if supports_tool_calling else '✗ NO'}"
        )
        logger.info(f"{'=' * 80}")
        logger.info("")

        return model_results

    def _execute_test(self, model: str, test_case: Dict[str, Any]) -> TestResult:
        start_time = time.time()
        tool_calls = []  # Initialize to avoid unbound variable
        test_run_id = None

        # Get or create test case in database
        test_case_id = self._get_or_create_test_case(test_case)
        run_number = self._get_next_run_number(test_case_id) if test_case_id else 1

        # Create test run record
        if test_case_id and self.db_manager:
            try:
                test_run_id = self.db_manager.create_test_run(
                    test_case_id=test_case_id,
                    run_number=run_number,
                    status="running",
                )
                logger.info(
                    f"│  📊 Test run created (ID: {test_run_id}, Run #{run_number})"
                )
            except Exception as e:
                logger.warning(f"│  ⚠ Failed to create test run: {e}")

        logger.info("│  📤 SENDING REQUEST TO LLM")
        logger.info(f"│  Model: {model}")
        logger.info(f"│  Endpoint: {self.chat_url}")
        logger.info(f"│  Tool Choice: {test_case.get('tool_choice', 'auto')}")
        logger.info(f"│  Temperature: {test_case.get('temperature', 0.7)}")
        logger.info(f"│  Max Tokens: {test_case.get('max_tokens', 1000)}")

        # Log messages
        messages = test_case.get("messages", [])
        logger.info(f"│  Messages ({len(messages)}):")
        for i, msg in enumerate(messages, 1):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            logger.info(f"│    [{i}] Role: {role}")
            if len(content) > 100:
                logger.info(f"│    Content: {content[:100]}... (truncated)")
            else:
                logger.info(f"│    Content: {content}")

        # Log tools
        tools = test_case.get("tools", [])
        logger.info(f"│  Tools Available ({len(tools)}):")
        for i, tool in enumerate(tools, 1):
            tool_name = tool.get("function", {}).get("name", "unknown")
            tool_desc = tool.get("function", {}).get("description", "")
            logger.info(f"│    [{i}] {tool_name}: {tool_desc[:60]}")

        logger.info("│")

        try:
            payload = {
                "model": model,
                "messages": test_case.get("messages", []),
                "tools": test_case.get("tools", []),
                "tool_choice": test_case.get("tool_choice", "auto"),
                "temperature": test_case.get("temperature", 0.7),
                "max_tokens": test_case.get("max_tokens", 1000),
            }

            response = requests.post(self.chat_url, json=payload, timeout=60)
            end_time = time.time()
            response_time = end_time - start_time

            logger.info(f"│  📥 RECEIVED RESPONSE (Status: {response.status_code})")
            logger.info(f"│  Response Time: {response_time:.3f}s")
            logger.info("│")

            response.raise_for_status()
            data = response.json()

            # Log response details
            logger.info("│  RESPONSE DETAILS:")
            logger.info(f"│    ID: {data.get('id', 'N/A')}")
            logger.info(f"│    Model: {data.get('model', 'N/A')}")
            logger.info(f"│    Object: {data.get('object', 'N/A')}")

            # Log usage
            usage = data.get("usage", {})
            logger.info(f"│    Usage:")
            logger.info(f"│      Prompt Tokens: {usage.get('prompt_tokens', 0)}")
            logger.info(
                f"│      Completion Tokens: {usage.get('completion_tokens', 0)}"
            )
            logger.info(f"│      Total Tokens: {usage.get('total_tokens', 0)}")
            logger.info("│")

            # Log choice details
            choices = data.get("choices", [])
            if choices:
                choice = choices[0]
                finish_reason = choice.get("finish_reason", "unknown")
                logger.info(f"│    Finish Reason: {finish_reason}")

                message = choice.get("message", {})

                # Log content if present
                content = message.get("content", "")
                if content:
                    logger.info(f"│    Content:")
                    logger.info(f"│      {content}")
                    logger.info("│")

                # Log reasoning_content if present
                reasoning = message.get("reasoning_content", "")
                if reasoning:
                    logger.info(f"│    Reasoning:")
                    if len(reasoning) > 200:
                        logger.info(f"│      {reasoning[:200]}... (truncated)")
                    else:
                        logger.info(f"│      {reasoning}")
                    logger.info("│")

                # Log tool calls
                tool_calls = message.get("tool_calls", [])
                logger.info(f"│    Tool Calls: {len(tool_calls)}")

                if tool_calls:
                    for i, tool_call in enumerate(tool_calls, 1):
                        logger.info(f"│      Tool Call #{i}:")
                        logger.info(f"│        ID: {tool_call.get('id', 'N/A')}")
                        logger.info(f"│        Type: {tool_call.get('type', 'N/A')}")

                        function = tool_call.get("function", {})
                        func_name = function.get("name", "N/A")
                        func_args = function.get("arguments", "{}")

                        logger.info(f"│        Function: {func_name}")
                        logger.info(f"│        Arguments:")
                        try:
                            args_dict = (
                                json.loads(func_args)
                                if isinstance(func_args, str)
                                else func_args
                            )
                            for key, value in args_dict.items():
                                logger.info(f"│          {key}: {value}")
                        except:
                            logger.info(f"│          {func_args}")
                else:
                    logger.info(f"│      (No tool calls made)")

                logger.info("│")

            # Verify tool calls
            passed = self._verify_tool_calls(tool_calls, test_case)

            logger.info(f"│  VERIFICATION: {'✓ PASSED' if passed else '✗ FAILED'}")

            if test_case.get("expected_tools"):
                logger.info(
                    f"│  Expected Tools: {[t.get('name', t.get('function', {}).get('name', 'N/A')) for t in test_case.get('expected_tools', [])]}"
                )
                logger.info(
                    f"│  Actual Tools: {[tc.get('function', {}).get('name', 'N/A') for tc in tool_calls]}"
                )

            # Update test run with results
            if test_run_id and self.db_manager:
                try:
                    self.db_manager.update_test_run(
                        test_run_id,
                        status="completed",
                        completed_at=datetime.now(),
                        total_iterations=1,
                        total_tool_calls=len(tool_calls),
                        final_status="success" if passed else "failed",
                    )

                    # Store performance metrics
                    self.db_manager.add_performance_metric(
                        session_id=test_run_id,
                        metric_name="response_time",
                        metric_value=response_time * 1000,
                        metric_unit="ms",
                    )
                    self.db_manager.add_performance_metric(
                        session_id=test_run_id,
                        metric_name="token_usage",
                        metric_value=usage.get("total_tokens", 0),
                        metric_unit="tokens",
                    )
                except Exception as e:
                    logger.warning(f"│  ⚠ Failed to update test run: {e}")

            return TestResult(
                success=True,
                model=model,
                test_case=test_case.get("name", "unnamed"),
                response_time=response_time,
                tool_calls=tool_calls,
                passed=passed,
            )

        except requests.RequestException as e:
            end_time = time.time()
            response_time = end_time - start_time
            logger.error(f"│  ✗ REQUEST ERROR: {e}")
            logger.error(f"│  Response Time: {response_time:.3f}s")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"│  Response Status: {e.response.status_code}")
                try:
                    logger.error(f"│  Response Body: {e.response.text[:500]}")
                except:
                    pass

            # Update test run with error
            if test_run_id and self.db_manager:
                try:
                    self.db_manager.update_test_run(
                        test_run_id,
                        status="error",
                        completed_at=datetime.now(),
                        error_message=str(e),
                        final_status="error",
                    )
                except Exception as db_error:
                    logger.warning(
                        f"│  ⚠ Failed to update test run with error: {db_error}"
                    )

            return TestResult(
                success=False,
                model=model,
                test_case=test_case.get("name", "unnamed"),
                response_time=response_time,
                error=str(e),
                passed=False,
            )
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            logger.error(f"│  ✗ UNEXPECTED ERROR: {e}")
            logger.error(f"│  Error Type: {type(e).__name__}")
            import traceback

            logger.error(f"│  Traceback:")
            for line in traceback.format_exc().split("\n"):
                if line.strip():
                    logger.error(f"│    {line}")

            # Update test run with error
            if test_run_id and self.db_manager:
                try:
                    self.db_manager.update_test_run(
                        test_run_id,
                        status="error",
                        completed_at=datetime.now(),
                        error_message=str(e),
                        final_status="error",
                    )
                except Exception as db_error:
                    logger.warning(
                        f"│  ⚠ Failed to update test run with error: {db_error}"
                    )

            return TestResult(
                success=False,
                model=model,
                test_case=test_case.get("name", "unnamed"),
                response_time=response_time,
                error=str(e),
                passed=False,
            )

    def _verify_tool_calls(
        self, tool_calls: List[Dict[str, Any]], test_case: Dict[str, Any]
    ) -> bool:
        """Verify tool calls using LLM judge or exact matching."""

        if self.use_llm_judge and self.judge:
            # Use LLM judge for semantic verification
            logger.info("│  🤖 Using LLM Judge for semantic verification")

            judgment = self.judge.verify_tool_call(test_case, tool_calls)

            passed = judgment.get("passed", False)
            reasoning = judgment.get("reasoning", "")
            confidence = judgment.get("confidence", 0.5)

            logger.info(f"│  🤖 LLM Judgment: {'✓ PASSED' if passed else '✗ FAILED'}")
            logger.info(f"│  🤖 Reasoning: {reasoning}")
            logger.info(f"│  🤖 Confidence: {confidence:.2f}")

            if judgment.get("details"):
                details = judgment["details"]
                if details.get("correct_tools"):
                    logger.info(
                        f"│  🤖 Correct tools: {', '.join(details['correct_tools'])}"
                    )
                if details.get("incorrect_tools"):
                    logger.info(
                        f"│  🤖 Incorrect tools: {', '.join(details['incorrect_tools'])}"
                    )
                if details.get("parameter_issues"):
                    logger.info(
                        f"│  🤖 Parameter issues: {', '.join(details['parameter_issues'])}"
                    )

            return passed
        else:
            # Use exact matching (old behavior)
            logger.info("│  🔍 Using exact matching verification")

            expected_tools = test_case.get("expected_tools", [])

            if not expected_tools:
                return len(tool_calls) == 0

            if len(tool_calls) != len(expected_tools):
                logger.info(
                    f"│  ✗ Wrong number of tools: expected {len(expected_tools)}, got {len(tool_calls)}"
                )
                return False

            for tool_call, expected_tool in zip(tool_calls, expected_tools):
                function = tool_call.get("function", {})
                function_name = function.get("name", "")
                expected_name = expected_tool.get("name", "")

                if function_name != expected_name:
                    logger.info(
                        f"│  ✗ Wrong tool: expected {expected_name}, got {function_name}"
                    )
                    return False

                expected_params = expected_tool.get("parameters", {})
                actual_params = function.get("arguments", {})

                if isinstance(actual_params, str):
                    try:
                        actual_params = json.loads(actual_params)
                    except json.JSONDecodeError:
                        logger.info(f"│  ✗ Failed to parse parameters")
                        return False

                for key, value in expected_params.items():
                    if key not in actual_params:
                        logger.info(f"│  ✗ Missing parameter: {key}")
                        return False
                    if actual_params[key] != value:
                        logger.info(
                            f"│  ✗ Parameter mismatch: {key} expected {value}, got {actual_params[key]}"
                        )
                        return False

            return True

    def run_test_case(
        self,
        model: str,
        test_case: Dict[str, Any],
        session_id: Optional[int] = None,
    ) -> TestResult:
        """Run a single test case with database tracking.

        Args:
            model: Model to test
            test_case: Test case definition
            session_id: Optional session ID from orchestrator

        Returns:
            TestResult with execution details
        """
        test_run_id = None

        # Get or create test case in database
        test_case_id = self._get_or_create_test_case(test_case)
        run_number = self._get_next_run_number(test_case_id) if test_case_id else 1

        # Create test run record
        if test_case_id and self.db_manager:
            try:
                test_run_id = self.db_manager.create_test_run(
                    test_case_id=test_case_id,
                    run_number=run_number,
                    session_id=session_id,
                    status="running",
                )
                logger.info(
                    f"📊 Test run created (ID: {test_run_id}, Run #{run_number})"
                )
            except Exception as e:
                logger.warning(f"⚠ Failed to create test run: {e}")

        # Execute the test
        result = self._execute_test(model, test_case)

        # Update test run with results
        if test_run_id and self.db_manager:
            try:
                self.db_manager.update_test_run(
                    test_run_id,
                    status="completed",
                    completed_at=datetime.now(),
                    total_iterations=1,
                    total_tool_calls=len(result.tool_calls),
                    final_status="success"
                    if result.success and result.passed
                    else "failed",
                )

                # Store judge evaluation if LLM judge was used
                if self.use_llm_judge and self.judge:
                    judgment = self.judge.verify_tool_call(test_case, result.tool_calls)
                    self.store_judge_evaluation(test_run_id, judgment)

                # Store performance metrics
                self.db_manager.add_performance_metric(
                    session_id=test_run_id,
                    metric_name="response_time",
                    metric_value=result.response_time * 1000,
                    metric_unit="ms",
                )
            except Exception as e:
                logger.warning(f"⚠ Failed to update test run: {e}")

        return result

    def test_all_models(
        self, test_cases: List[Dict[str, Any]]
    ) -> Dict[str, ModelTestResults]:
        self._fetch_models()

        for model_info in self.models:
            model_id = model_info.get("id", model_info.get("name", ""))
            if not model_id:
                continue

            print(f"Testing model: {model_id}")
            results = self.test_model_tool_calling(model_id, test_cases)
            self.metrics[model_id] = results

        return self.metrics

    def get_summary(self) -> Dict[str, Any]:
        total_models = len(self.metrics)
        total_tests = sum(m.total_tests for m in self.metrics.values())
        total_passed = sum(m.passed for m in self.metrics.values())
        total_failed = sum(m.failed for m in self.metrics.values())
        avg_response_time = (
            sum(m.average_response_time for m in self.metrics.values()) / total_models
            if total_models
            else 0
        )
        tool_calling_models = sum(
            1 for m in self.metrics.values() if m.supports_tool_calling
        )

        return {
            "total_models": total_models,
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "success_rate": (total_passed / total_tests * 100) if total_tests else 0,
            "average_response_time": avg_response_time,
            "models_with_tool_calling": tool_calling_models,
            "models": self.metrics,
        }

    def _get_or_create_test_case(self, test_case: Dict[str, Any]) -> Optional[int]:
        if not self.db_manager:
            return None

        try:
            test_case_name = test_case.get("name", "unnamed")

            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id FROM test_cases WHERE name = ?",
                    (test_case_name,),
                )
                row = cursor.fetchone()
                if row:
                    logger.info(f"│  📊 Found existing test case: {test_case_name}")
                    return row["id"]

            category = test_case.get("category", "tool_calling")
            description = test_case.get("description", "")
            messages = test_case.get("messages", [])
            tools = test_case.get("tools", [])
            expected_tools = test_case.get("expected_tools", None)
            expected_parameters = test_case.get("expected_parameters", None)
            expect_no_tools = test_case.get("expect_no_tools", False)

            test_case_id = self.db_manager.create_test_case(
                name=test_case_name,
                category=category,
                description=description,
                messages=messages,
                tools=tools,
                expected_tools=expected_tools,
                expected_parameters=expected_parameters,
                expect_no_tools=expect_no_tools,
            )
            logger.info(
                f"│  📊 Created new test case: {test_case_name} (ID: {test_case_id})"
            )
            return test_case_id

        except Exception as e:
            logger.warning(f"│  ⚠ Failed to get/create test case: {e}")
            return None

    def _get_next_run_number(self, test_case_id: int) -> int:
        if not self.db_manager:
            return 1

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COALESCE(MAX(run_number), 0) + 1 as next_run FROM test_runs WHERE test_case_id = ?",
                    (test_case_id,),
                )
                row = cursor.fetchone()
                return row["next_run"] if row else 1

        except Exception as e:
            logger.warning(f"│  ⚠ Failed to get next run number: {e}")
            return 1

    def store_judge_evaluation(
        self,
        test_run_id: int,
        judgment: Dict[str, Any],
        tool_call_id: Optional[int] = None,
    ) -> Optional[int]:
        if not self.db_manager:
            return None

        try:
            judge_model = judgment.get("judge_model", "unknown")
            passed = judgment.get("passed", False)
            confidence = judgment.get("confidence", 0.5)
            reasoning = judgment.get("reasoning", "")
            details = judgment.get("details", None)
            criteria = judgment.get("criteria", None)
            evaluation_time_ms = judgment.get("evaluation_time_ms", None)

            evaluation_id = self.db_manager.add_judge_evaluation(
                test_run_id=test_run_id,
                judge_model=judge_model,
                passed=passed,
                confidence=confidence,
                reasoning=reasoning,
                evaluation_type="semantic",
                criteria=criteria,
                details=details,
                evaluation_time_ms=evaluation_time_ms,
                tool_call_id=tool_call_id,
            )
            logger.info(f"│  📊 Stored judge evaluation (ID: {evaluation_id})")
            return evaluation_id

        except Exception as e:
            logger.warning(f"│  ⚠ Failed to store judge evaluation: {e}")
            return None

    def get_test_history(
        self, test_case_name: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        if not self.db_manager:
            return []

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT tr.*, tc.name as test_case_name
                    FROM test_runs tr
                    JOIN test_cases tc ON tr.test_case_id = tc.id
                    WHERE tc.name = ?
                    ORDER BY tr.run_number DESC
                    LIMIT ?
                    """,
                    (test_case_name, limit),
                )
                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.warning(f"⚠ Failed to get test history: {e}")
            return []

    def get_performance_trends(self, days: int = 30) -> Dict[str, Any]:
        if not self.db_manager:
            return {}

        try:
            performance = self.db_manager.get_test_performance(days)
            tool_stats = self.db_manager.get_tool_success_rate()
            model_stats = self.db_manager.get_model_comparison()

            return {
                "test_performance": performance,
                "tool_success_rate": tool_stats,
                "model_comparison": model_stats,
                "total_test_cases": len(performance),
                "total_tools": len(tool_stats),
                "total_models": len(model_stats),
            }

        except Exception as e:
            logger.warning(f"⚠ Failed to get performance trends: {e}")
            return {}
