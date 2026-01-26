"""OpenAI-compatible API benchmark tool."""

import time
import requests
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, median


@dataclass
class BenchmarkResult:
    """Result from a single benchmark test."""
    model: str
    success: bool
    latency_ms: float
    tokens_per_second: float
    error: str = None
    response_time: float = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIBenchmark:
    """Benchmark tool for OpenAI-compatible APIs."""

    def __init__(self, base_url: str = "http://localhost:8087/v1"):
        self.base_url = base_url.rstrip('/')
        self.chat_url = f"{self.base_url}/chat/completions"
        self.models_url = f"{self.base_url}/models"

    def get_models(self) -> List[str]:
        """Get list of available models."""
        try:
            response = requests.get(self.models_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
        except Exception as e:
            print(f"Failed to get models: {e}")
            return []

    def test_compatibility(self, model: str) -> Dict[str, Any]:
        """Test OpenAI API compatibility for a model."""
        print(f"\n🧪 Testing compatibility: {model}")

        tests = {
            "basic_chat": self._test_basic_chat(model),
            "system_prompt": self._test_system_prompt(model),
            "temperature": self._test_temperature(model),
            "max_tokens": self._test_max_tokens(model),
            "streaming": self._test_streaming(model),
            "json_mode": self._test_json_mode(model),
        }

        passed = sum(1 for test in tests.values() if test["success"])
        total = len(tests)

        print(f"   Compatibility: {passed}/{total} tests passed")

        return {
            "model": model,
            "tests": tests,
            "passed": passed,
            "total": total,
            "compatibility_score": (passed / total) * 100
        }

    def benchmark_speed(self, model: str, iterations: int = 5) -> BenchmarkResult:
        """Benchmark model speed."""
        print(f"\n⚡ Benchmarking speed: {model}")

        prompt = "Write a brief summary of machine learning in 3 sentences."

        results = []
        for i in range(iterations):
            try:
                start_time = time.time()
                response = requests.post(
                    self.chat_url,
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 100
                    },
                    timeout=30
                )
                end_time = time.time()

                if response.status_code == 200:
                    data = response.json()
                    usage = data.get("usage", {})
                    latency = (end_time - start_time) * 1000

                    completion_tokens = usage.get("completion_tokens", 0)
                    tokens_per_sec = completion_tokens / (end_time - start_time) if completion_tokens > 0 else 0

                    results.append({
                        "latency_ms": latency,
                        "tokens_per_second": tokens_per_sec,
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": completion_tokens,
                        "total_tokens": usage.get("total_tokens", 0)
                    })
                    print(f"   [{i+1}/{iterations}] {latency:.0f}ms, {tokens_per_sec:.1f} tok/s")
                else:
                    print(f"   [{i+1}/{iterations}] Failed: {response.status_code}")
            except Exception as e:
                print(f"   [{i+1}/{iterations}] Error: {e}")

        if results:
            avg_latency = mean(r["latency_ms"] for r in results)
            avg_tokens_per_sec = mean(r["tokens_per_second"] for r in results)
            median_latency = median(r["latency_ms"] for r in results)

            return BenchmarkResult(
                model=model,
                success=True,
                latency_ms=avg_latency,
                tokens_per_second=avg_tokens_per_sec,
                response_time=median_latency,
                prompt_tokens=int(mean(r["prompt_tokens"] for r in results)),
                completion_tokens=int(mean(r["completion_tokens"] for r in results)),
                total_tokens=int(mean(r["total_tokens"] for r in results))
            )
        else:
            return BenchmarkResult(
                model=model,
                success=False,
                latency_ms=0,
                tokens_per_second=0,
                error="All requests failed"
            )

    def test_rate_limit(self, model: str, requests_per_second: int = 10) -> Dict[str, Any]:
        """Test rate limiting."""
        print(f"\n🚦 Testing rate limit: {model} ({requests_per_second} req/s)")

        prompt = "Say hello"
        interval = 1.0 / requests_per_second

        successful = 0
        rate_limited = 0
        errors = 0
        latencies = []

        for i in range(20):  # Test 20 requests
            try:
                start_time = time.time()
                response = requests.post(
                    self.chat_url,
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 10
                    },
                    timeout=10
                )
                latency = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    successful += 1
                    latencies.append(latency)
                elif response.status_code == 429:
                    rate_limited += 1
                    print(f"   [{i+1}] Rate limited")
                else:
                    errors += 1
                    print(f"   [{i+1}] Error: {response.status_code}")
            except Exception as e:
                errors += 1
                print(f"   [{i+1}] Exception: {e}")

            time.sleep(interval)

        avg_latency = mean(latencies) if latencies else 0

        return {
            "model": model,
            "successful": successful,
            "rate_limited": rate_limited,
            "errors": errors,
            "avg_latency_ms": avg_latency,
            "requests_per_second": requests_per_second
        }

    def _test_basic_chat(self, model: str) -> Dict:
        """Test basic chat completion."""
        try:
            response = requests.post(
                self.chat_url,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "max_tokens": 10
                },
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return {"success": True, "response": data.get("choices", [{}])[0].get("message", {}).get("content", "")}
            return {"success": False, "error": f"Status {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _test_system_prompt(self, model: str) -> Dict:
        """Test system prompt support."""
        try:
            response = requests.post(
                self.chat_url,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "What is your role?"}
                    ],
                    "max_tokens": 20
                },
                timeout=10
            )
            if response.status_code == 200:
                return {"success": True}
            return {"success": False, "error": f"Status {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _test_temperature(self, model: str) -> Dict:
        """Test temperature parameter."""
        try:
            response = requests.post(
                self.chat_url,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "temperature": 0.1,
                    "max_tokens": 10
                },
                timeout=10
            )
            if response.status_code == 200:
                return {"success": True}
            return {"success": False, "error": f"Status {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _test_max_tokens(self, model: str) -> Dict:
        """Test max_tokens parameter."""
        try:
            response = requests.post(
                self.chat_url,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Count to 100"}],
                    "max_tokens": 10
                },
                timeout=10
            )
            if response.status_code == 200:
                return {"success": True}
            return {"success": False, "error": f"Status {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _test_streaming(self, model: str) -> Dict:
        """Test streaming support."""
        try:
            response = requests.post(
                self.chat_url,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "stream": True,
                    "max_tokens": 10
                },
                timeout=10,
                stream=True
            )
            if response.status_code == 200:
                # Read a few chunks to verify streaming
                chunks = []
                for line in response.iter_lines():
                    if line:
                        chunks.append(line)
                        if len(chunks) >= 3:
                            break
                return {"success": len(chunks) > 0}
            return {"success": False, "error": f"Status {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _test_json_mode(self, model: str) -> Dict:
        """Test JSON mode support."""
        try:
            response = requests.post(
                self.chat_url,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Return a JSON object with key 'hello' and value 'world'"}],
                    "response_format": {"type": "json_object"},
                    "max_tokens": 50
                },
                timeout=10
            )
            if response.status_code == 200:
                return {"success": True}
            elif response.status_code == 400:
                return {"success": False, "error": "JSON mode not supported"}
            return {"success": False, "error": f"Status {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def run_full_benchmark(self, models: List[str] = None):
        """Run full benchmark suite."""
        print("🚀 Starting OpenAI API Benchmark Suite")
        print(f"📍 API: {self.base_url}")
        print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if models is None:
            models = self.get_models()

        if not models:
            print("❌ No models found")
            return

        print(f"\n📊 Found {len(models)} models to benchmark\n")

        results = []

        for model in models:
            print(f"\n{'='*60}")
            print(f"🔍 BENCHMARKING: {model}")
            print(f"{'='*60}")

            # Compatibility tests
            compatibility = self.test_compatibility(model)

            # Speed benchmark
            speed = self.benchmark_speed(model, iterations=5)

            # Rate limit test
            rate_limit = self.test_rate_limit(model, requests_per_second=5)

            results.append({
                "model": model,
                "compatibility": compatibility,
                "speed": speed,
                "rate_limit": rate_limit
            })

        # Print summary
        print(f"\n{'='*60}")
        print("📋 BENCHMARK SUMMARY")
        print(f"{'='*60}\n")

        for result in results:
            comp = result["compatibility"]
            speed = result["speed"]
            rate = result["rate_limit"]

            print(f"🤖 {result['model']}")
            print(f"   Compatibility: {comp['compatibility_score']:.0f}% ({comp['passed']}/{comp['total']})")
            if speed.success:
                print(f"   Speed: {speed.latency_ms:.0f}ms avg, {speed.tokens_per_second:.1f} tok/s")
            print(f"   Rate Limit: {rate['successful']}/{20} successful, {rate['rate_limited']} rate limited")
            print()

        return results


if __name__ == "__main__":
    import sys

    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8087/v1"
    models = sys.argv[2:] if len(sys.argv) > 2 else None

    benchmark = OpenAIBenchmark(base_url)
    benchmark.run_full_benchmark(models)