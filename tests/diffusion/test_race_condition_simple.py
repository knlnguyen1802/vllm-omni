"""
Simple demonstration of the race condition bug.

This test shows how concurrent calls to add_req and collective_rpc
can interfere with each other because they both access scheduler.mq 
without synchronization.

To run this test:
    python tests/diffusion/test_race_condition_simple.py
    
Or with pytest:
    pytest tests/diffusion/test_race_condition_simple.py -v -s
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor


def demonstrate_race_condition():
    """
    Simplified demonstration showing the problematic code pattern.
    """
    print("=== Demonstrating Race Condition ===\n")
    
    # Simulate the shared queue (like scheduler.mq)
    class MockQueue:
        def __init__(self):
            self.queue = []
            self.access_log = []
            
        def enqueue(self, msg):
            # No lock here - this is the bug!
            thread_name = threading.current_thread().name
            self.access_log.append(f"{thread_name} entering enqueue")
            
            # Simulate some processing
            time.sleep(0.001)
            self.queue.append(msg)
            
            self.access_log.append(f"{thread_name} exiting enqueue")
    
    class MockResultQueue:
        def __init__(self):
            self.responses = []
            self.dequeue_calls = []
            
        def enqueue(self, response):
            self.responses.append(response)
            
        def dequeue(self, timeout=None):
            thread_name = threading.current_thread().name
            self.dequeue_calls.append(thread_name)
            
            # In real code, this returns responses that may not match the caller!
            if self.responses:
                return self.responses.pop(0)
            return {"status": "ok"}
    
    # Simulate the scheduler
    class MockScheduler:
        def __init__(self):
            self.mq = MockQueue()
            self.result_mq = MockResultQueue()
            
        def add_req(self, request):
            """This is like the add_req in scheduler.py"""
            rpc_request = {
                "type": "rpc",
                "method": "generate",
                "request_id": request,
            }
            # BUG: No synchronization before accessing mq!
            self.mq.enqueue(rpc_request)
            return self.result_mq.dequeue()
    
    class MockExecutor:
        def __init__(self, scheduler):
            self.scheduler = scheduler
            self.od_config = type('obj', (object,), {'num_gpus': 2})()
            
        def collective_rpc(self, method):
            """This is like collective_rpc in multiproc_executor.py"""
            rpc_request = {
                "type": "rpc",
                "method": method,
            }
            # BUG: No synchronization before accessing mq!
            self.scheduler.mq.enqueue(rpc_request)
            
            num_responses = 1
            responses = []
            for _ in range(num_responses):
                response = self.scheduler.result_mq.dequeue()
                responses.append(response)
            return responses[0]
        
        def add_req(self, request):
            return self.scheduler.add_req(request)
    
    # Create the mock system
    scheduler = MockScheduler()
    executor = MockExecutor(scheduler)
    
    # Run concurrent operations
    def worker_add_req(i):
        result = executor.add_req(f"request_{i}")
        print(f"  add_req({i}) completed")
        return result
    
    def worker_rpc(i):
        result = executor.collective_rpc(f"method_{i}")
        print(f"  collective_rpc({i}) completed")
        return result
    
    print("Starting concurrent calls...")
    print("(Without proper synchronization, these will interleave)\n")
    
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = []
        
        # Submit mixed add_req and collective_rpc calls
        for i in range(5):
            futures.append(pool.submit(worker_add_req, i))
            futures.append(pool.submit(worker_rpc, i))
        
        # Wait for completion
        for f in futures:
            f.result()
    
    # Show the evidence
    print("\n=== Evidence of Race Condition ===\n")
    
    print("Access pattern to shared queue (scheduler.mq):")
    for log in scheduler.mq.access_log[:20]:  # Show first 20
        print(f"  {log}")
    
    # Check for interleaving
    interleaves = 0
    for i in range(1, len(scheduler.mq.access_log)):
        if "entering" in scheduler.mq.access_log[i] and "entering" in scheduler.mq.access_log[i-1]:
            interleaves += 1
        elif "exiting" in scheduler.mq.access_log[i] and "entering" in scheduler.mq.access_log[i-1]:
            if scheduler.mq.access_log[i].split()[0] != scheduler.mq.access_log[i-1].split()[0]:
                interleaves += 1
    
    print(f"\nDetected {interleaves} thread interleaves")
    print("\n=== Explanation ===")
    print("The issue:")
    print("  1. collective_rpc (line 161 in multiproc_executor.py) does:")
    print("     self.scheduler.mq.enqueue(rpc_request)")
    print()
    print("  2. add_req (line 132 in multiproc_executor.py -> scheduler.py) does:")
    print("     self.scheduler.add_req(request)")
    print("     which internally calls: self.mq.enqueue(rpc_request)")
    print()
    print("  3. Both access the SAME queue (scheduler.mq) without any lock!")
    print()
    print("  4. When called concurrently, this causes:")
    print("     - Message queue corruption")
    print("     - Responses going to wrong caller (both dequeue from result_mq)")
    print("     - Race condition on queue state")
    print()
    print("Solution: Add a lock around mq.enqueue() and result_mq.dequeue() calls")
    print("          OR ensure only one thread can access scheduler at a time")
    

def test_race_condition_with_pytest():
    """Pytest version of the test."""
    demonstrate_race_condition()
    # If we got here without crashes, the test passes
    # (In real scenarios, the race condition causes intermittent failures)


if __name__ == "__main__":
    demonstrate_race_condition()
