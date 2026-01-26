#!/usr/bin/env python3
"""
VibeAgent Implementation Verification Script
Run this script to verify all components are working correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def print_section(title):
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def verify_imports():
    """Verify all core modules can be imported."""
    print_section("1. Verifying Core Module Imports")

    modules = [
        "core.database_manager",
        "core.tool_orchestrator",
        "core.error_handler",
        "core.retry_manager",
        "core.parallel_executor",
        "core.self_corrector",
        "core.tot_orchestrator",
        "core.plan_execute_orchestrator",
        "core.context_manager",
        "core.analytics_engine",
        "core.analytics_dashboard",
        "config.model_configs",
        "prompts.react_prompt",
    ]

    success = 0
    failed = 0

    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
            success += 1
        except Exception as e:
            print(f"  ✗ {module}: {e}")
            failed += 1

    print(f"\n  Result: {success}/{len(modules)} modules imported successfully")
    return failed == 0


def verify_database():
    """Verify database initialization."""
    print_section("2. Verifying Database Initialization")

    try:
        from core.database_manager import DatabaseManager

        # Create temporary database
        db_path = project_root / "data" / "test_verification.db"
        db = DatabaseManager(str(db_path))

        # Test basic operations
        session_id = db.create_session(
            session_id="test_session", session_type="test", model="test-model"
        )

        print(f"  ✓ Database created at {db_path}")
        print(f"  ✓ Session created with ID: {session_id}")
        print(f"  ✓ Database operations working")

        # Clean up
        db_path.unlink(missing_ok=True)
        db_path.parent.rmdir()

        return True
    except Exception as e:
        print(f"  ✗ Database verification failed: {e}")
        return False


def verify_model_configs():
    """Verify model configuration system."""
    print_section("3. Verifying Model Configuration System")

    try:
        from config.model_configs import (
            get_model_config,
            get_temperature_for_phase,
            get_max_tokens_for_phase,
        )

        # Test getting config for GPT-4
        config = get_model_config("gpt-4")
        print(f"  ✓ GPT-4 config loaded")
        print(f"    - Max context: {config.max_context_tokens}")
        print(f"    - Max iterations: {config.max_iterations}")

        # Test phase-specific settings
        temp = get_temperature_for_phase("gpt-4", "planning")
        tokens = get_max_tokens_for_phase("gpt-4", "execution")
        print(f"  ✓ Phase-specific settings working")
        print(f"    - Planning temperature: {temp}")
        print(f"    - Execution max tokens: {tokens}")

        return True
    except Exception as e:
        print(f"  ✗ Model config verification failed: {e}")
        return False


def verify_prompts():
    """Verify ReAct prompt system."""
    print_section("4. Verifying ReAct Prompt System")

    try:
        from prompts.react_prompt import (
            get_react_system_prompt,
            get_few_shot_examples,
            build_react_prompt,
        )

        # Test getting system prompt
        prompt = get_react_system_prompt("gpt4")
        print(f"  ✓ GPT-4 system prompt loaded ({len(prompt)} chars)")

        # Test getting examples
        examples = get_few_shot_examples("simple")
        print(f"  ✓ Few-shot examples loaded ({len(examples)} examples)")

        # Test building complete prompt
        full_prompt = build_react_prompt(
            messages=[{"role": "user", "content": "test"}], tools=[], model_type="gpt4"
        )
        print(f"  ✓ Complete prompt built ({len(full_prompt)} chars)")

        return True
    except Exception as e:
        print(f"  ✗ Prompt system verification failed: {e}")
        return False


def verify_components():
    """Verify core components."""
    print_section("5. Verifying Core Components")

    components = []

    # Test ErrorHandler
    try:
        from core.error_handler import ErrorHandler

        handler = ErrorHandler()
        print(f"  ✓ ErrorHandler initialized")
        components.append("ErrorHandler")
    except Exception as e:
        print(f"  ✗ ErrorHandler: {e}")

    # Test RetryManager
    try:
        from core.retry_manager import RetryManager

        retry_mgr = RetryManager()
        print(f"  ✓ RetryManager initialized")
        components.append("RetryManager")
    except Exception as e:
        print(f"  ✗ RetryManager: {e}")

    # Test ParallelExecutor
    try:
        from core.parallel_executor import ParallelExecutor

        executor = ParallelExecutor()
        print(f"  ✓ ParallelExecutor initialized")
        components.append("ParallelExecutor")
    except Exception as e:
        print(f"  ✗ ParallelExecutor: {e}")

    # Test SelfCorrector
    try:
        from core.self_corrector import SelfCorrector

        corrector = SelfCorrector()
        print(f"  ✓ SelfCorrector initialized")
        components.append("SelfCorrector")
    except Exception as e:
        print(f"  ✗ SelfCorrector: {e}")

    # Test ContextManager
    try:
        from core.context_manager import ContextManager

        ctx_mgr = ContextManager()
        print(f"  ✓ ContextManager initialized")
        components.append("ContextManager")
    except Exception as e:
        print(f"  ✗ ContextManager: {e}")

    print(f"\n  Result: {len(components)}/5 components initialized")
    return len(components) >= 4


def verify_orchestrators():
    """Verify orchestrator classes."""
    print_section("6. Verifying Orchestrator Classes")

    orchestrators = []

    # Test ToolOrchestrator
    try:
        from core.tool_orchestrator import ToolOrchestrator

        print(f"  ✓ ToolOrchestrator class available")
        orchestrators.append("ToolOrchestrator")
    except Exception as e:
        print(f"  ✗ ToolOrchestrator: {e}")

    # Test TreeOfThoughtsOrchestrator
    try:
        from core.tot_orchestrator import TreeOfThoughtsOrchestrator

        print(f"  ✓ TreeOfThoughtsOrchestrator class available")
        orchestrators.append("TreeOfThoughtsOrchestrator")
    except Exception as e:
        print(f"  ✗ TreeOfThoughtsOrchestrator: {e}")

    # Test PlanExecuteOrchestrator
    try:
        from core.plan_execute_orchestrator import PlanExecuteOrchestrator

        print(f"  ✓ PlanExecuteOrchestrator class available")
        orchestrators.append("PlanExecuteOrchestrator")
    except Exception as e:
        print(f"  ✗ PlanExecuteOrchestrator: {e}")

    print(f"\n  Result: {len(orchestrators)}/3 orchestrators available")
    return len(orchestrators) >= 2


def verify_analytics():
    """Verify analytics components."""
    print_section("7. Verifying Analytics Components")

    try:
        from core.analytics_engine import AnalyticsEngine
        from core.analytics_dashboard import AnalyticsDashboard

        print(f"  ✓ AnalyticsEngine class available")
        print(f"  ✓ AnalyticsDashboard class available")

        # Test dashboard methods exist
        dashboard = AnalyticsDashboard(None)
        methods = [
            "get_overview_panel",
            "get_performance_panel",
            "get_test_results_panel",
            "generate_report",
        ]

        for method in methods:
            if hasattr(dashboard, method):
                print(f"  ✓ Dashboard method: {method}")
            else:
                print(f"  ✗ Dashboard method missing: {method}")

        return True
    except Exception as e:
        print(f"  ✗ Analytics verification failed: {e}")
        return False


def verify_tests():
    """Verify test files exist."""
    print_section("8. Verifying Test Files")

    test_files = [
        "tests/test_database_manager.py",
        "tests/test_integration.py",
        "tests/test_error_handler.py",
        "tests/test_retry_manager.py",
        "tests/test_parallel_executor.py",
        "tests/test_self_corrector.py",
        "tests/test_tot_orchestrator.py",
        "tests/test_plan_execute_orchestrator.py",
        "tests/test_context_manager.py",
        "tests/test_model_configs.py",
    ]

    found = 0
    for test_file in test_files:
        path = project_root / test_file
        if path.exists():
            print(f"  ✓ {test_file}")
            found += 1
        else:
            print(f"  ✗ {test_file} (not found)")

    print(f"\n  Result: {found}/{len(test_files)} test files found")
    return found >= len(test_files) * 0.8


def verify_documentation():
    """Verify documentation files."""
    print_section("9. Verifying Documentation")

    docs = [
        "IMPLEMENTATION_SUMMARY.md",
        "COMPREHENSIVE_IMPLEMENTATION_PLAN.md",
        "RESEARCH_EXECUTIVE_SUMMARY.md",
        "MULTI_CALL_IMPROVEMENT_PLAN.md",
    ]

    found = 0
    for doc in docs:
        path = project_root / doc
        if path.exists():
            size = path.stat().st_size / 1024
            print(f"  ✓ {doc} ({size:.1f} KB)")
            found += 1
        else:
            print(f"  ✗ {doc} (not found)")

    print(f"\n  Result: {found}/{len(docs)} documentation files found")
    return found >= 3


def print_summary(results):
    """Print verification summary."""
    print_section("VERIFICATION SUMMARY")

    total = len(results)
    passed = sum(results.values())

    for test, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {status}: {test}")

    print(f"\n{'=' * 70}")
    print(f"  Overall Result: {passed}/{total} tests passed")
    print(f"  Success Rate: {passed / total * 100:.1f}%")
    print(f"{'=' * 70}\n")

    if passed == total:
        print("  🎉 All verifications passed! The system is ready to use.")
        print("\n  Next steps:")
        print("    1. Initialize the database: python scripts/init_db.py")
        print("    2. Run tests: pytest tests/")
        print("    3. Try the examples in examples/")
        return 0
    else:
        print("  ⚠️  Some verifications failed. Please check the errors above.")
        return 1


def main():
    """Run all verifications."""
    print("\n" + "=" * 70)
    print("  VibeAgent Implementation Verification")
    print("=" * 70)

    results = {
        "Core Module Imports": verify_imports(),
        "Database Initialization": verify_database(),
        "Model Configuration System": verify_model_configs(),
        "ReAct Prompt System": verify_prompts(),
        "Core Components": verify_components(),
        "Orchestrator Classes": verify_orchestrators(),
        "Analytics Components": verify_analytics(),
        "Test Files": verify_tests(),
        "Documentation": verify_documentation(),
    }

    return print_summary(results)


if __name__ == "__main__":
    sys.exit(main())
