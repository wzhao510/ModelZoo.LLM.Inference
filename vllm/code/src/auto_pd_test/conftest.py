def pytest_configure(config):
    config.addinivalue_line("markers", "pd: prefill/decode disaggregation functional test")
    config.addinivalue_line("markers", "kvoffload: single-instance KV cache offload test")
    config.addinivalue_line("markers", "kvshare: cross-instance KV cache sharing test")
    config.addinivalue_line(
        "markers", "backend(name): which connector/backend a test exercises (lmcache, mooncake, nixl, vllm_native)"
    )
    config.addinivalue_line(
        "markers", "slow: long-running test (multi-instance cluster boot, large prompts)"
    )
