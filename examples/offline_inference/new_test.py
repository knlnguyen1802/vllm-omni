import importlib, time

print("Before import")
t0 = time.time()
try:
    from vllm.entrypoints.openai.api_server import init_app_state
    print("After import", time.time() - t0)
except Exception as e:
    import traceback; traceback.print_exc()
    print("FAILED after", time.time() - t0)