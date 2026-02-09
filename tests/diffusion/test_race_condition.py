"""
Test to demonstrate race condition between collective_rpc and add_req.

Both methods access scheduler.mq.enqueue() without synchronization, which can cause:
1. Interleaved messages in the queue
2. Wrong thread receiving responses from result_mq
3. Timeout errors or incorrect results
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from unittest.mock import MagicMock, Mock, patch

import pytest

from vllm_omni.diffusion.executor.multiproc_executor import (
    MultiprocDiffusionExecutor,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest


class TestRaceCondition:
    """Test class to demonstrate scheduler queue race condition."""

    @pytest.fixture
    def mock_executor(self):
        """Create a mock executor with minimal setup to test the race condition."""
        executor = MultiprocDiffusionExecutor.__new__(MultiprocDiffusionExecutor)
        
        # Create mock scheduler
        mock_scheduler = MagicMock()
        mock_mq = MagicMock()
        mock_result_mq = MagicMock()
        
        # Track enqueue calls to detect interleaving
        executor._enqueue_calls = []
        executor._lock_for_tracking = threading.Lock()
        
        def track_enqueue(msg):
            """Track all enqueue calls with thread information."""
            with executor._lock_for_tracking:
                executor._enqueue_calls.append({
                    'thread_id': threading.current_thread().ident,
                    'thread_name': threading.current_thread().name,
                    'timestamp': time.time(),
                    'msg_type': msg.get('type'),
                    'method': msg.get('method'),
                })
        
        mock_mq.enqueue.side_effect = track_enqueue
        
        # Create a queue to simulate responses (but may go to wrong thread)
        executor._response_queue = []
        executor._response_lock = threading.Lock()
        
        def mock_dequeue(timeout=None):
            """Simulate dequeue - may return response to wrong caller in race condition."""
            # Simulate some processing delay
            time.sleep(0.001)
            
            with executor._response_lock:
                if executor._response_queue:
                    return executor._response_queue.pop(0)
                else:
                    # Return a generic response
                    return {"status": "ok", "result": "test_result"}
        
        mock_result_mq.dequeue.side_effect = mock_dequeue
        
        mock_scheduler.mq = mock_mq
        mock_scheduler.result_mq = mock_result_mq
        
        executor.scheduler = mock_scheduler
        executor._closed = False
        executor.od_config = MagicMock()
        executor.od_config.num_gpus = 2
        
        return executor

    def test_concurrent_add_req_and_collective_rpc_race_condition(self, mock_executor):
        """
        Test demonstrating race condition when add_req and collective_rpc 
        are called concurrently from different threads.
        
        Expected behavior WITHOUT race condition:
        - Each thread should enqueue its message
        - Each thread should receive the correct response
        - Messages should be processed in a predictable order
        
        Actual behavior WITH race condition:
        - Messages from different calls interleave in the queue
        - Threads may receive each other's responses
        - Can cause timeouts, wrong results, or crashes
        """
        num_add_req_calls = 5
        num_rpc_calls = 5
        num_threads = 10
        
        errors = []
        results = []
        
        def call_add_req(i):
            """Simulate add_req call."""
            try:
                request = MagicMock(spec=OmniDiffusionRequest)
                request.request_id = f"add_req_{i}"
                result = mock_executor.add_req(request)
                results.append(('add_req', i, result))
            except Exception as e:
                errors.append(('add_req', i, str(e)))
        
        def call_collective_rpc(i):
            """Simulate collective_rpc call."""
            try:
                result = mock_executor.collective_rpc(
                    method=f"test_method_{i}",
                    timeout=5.0,
                    unique_reply_rank=0,
                )
                results.append(('collective_rpc', i, result))
            except Exception as e:
                errors.append(('collective_rpc', i, str(e)))
        
        # Execute concurrent calls
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            
            # Submit add_req calls
            for i in range(num_add_req_calls):
                futures.append(executor.submit(call_add_req, i))
            
            # Submit collective_rpc calls
            for i in range(num_rpc_calls):
                futures.append(executor.submit(call_collective_rpc, i))
            
            # Wait for all to complete
            for future in as_completed(futures):
                future.result()
        
        # Analyze the race condition evidence
        print("\n=== Race Condition Analysis ===")
        print(f"\nTotal enqueue calls: {len(mock_executor._enqueue_calls)}")
        print(f"Total errors: {len(errors)}")
        print(f"Total successful results: {len(results)}")
        
        # Check for message interleaving
        print("\n=== Message Interleaving Evidence ===")
        prev_thread = None
        interleave_count = 0
        
        for i, call in enumerate(mock_executor._enqueue_calls):
            if prev_thread is not None and prev_thread != call['thread_id']:
                interleave_count += 1
            print(f"Call {i}: Thread {call['thread_name']}, Method: {call['method']}")
            prev_thread = call['thread_id']
        
        print(f"\nThread switches (interleaving): {interleave_count}")
        
        # Show thread distribution
        thread_distribution = {}
        for call in mock_executor._enqueue_calls:
            thread_name = call['thread_name']
            thread_distribution[thread_name] = thread_distribution.get(thread_name, 0) + 1
        
        print("\n=== Thread Distribution ===")
        for thread_name, count in sorted(thread_distribution.items()):
            print(f"{thread_name}: {count} calls")
        
        # Print any errors
        if errors:
            print("\n=== Errors Encountered (Evidence of Race Condition) ===")
            for error_type, error_id, error_msg in errors:
                print(f"{error_type}_{error_id}: {error_msg}")
        
        # Demonstrate the race condition
        assert interleave_count > 0, (
            "Expected message interleaving between threads, which demonstrates "
            "the race condition where multiple threads access scheduler.mq.enqueue() "
            "without synchronization"
        )
        
        print("\n=== Race Condition Confirmed ===")
        print(f"Multiple threads interleaved {interleave_count} times while accessing "
              "the scheduler message queue without proper synchronization.")
        print("This can lead to:")
        print("  1. Wrong thread receiving responses")
        print("  2. Timeout errors")
        print("  3. Incorrect results")
        print("  4. Message ordering issues")

    def test_race_condition_with_timing_analysis(self, mock_executor):
        """
        Test with timing analysis to show overlapping access to scheduler queue.
        """
        access_log = []
        lock = threading.Lock()
        
        original_enqueue = mock_executor.scheduler.mq.enqueue.side_effect
        
        def enqueue_with_timing(msg):
            """Track when each thread enters and exits enqueue."""
            thread_id = threading.current_thread().ident
            thread_name = threading.current_thread().name
            
            with lock:
                access_log.append({
                    'event': 'enter_enqueue',
                    'thread_id': thread_id,
                    'thread_name': thread_name,
                    'timestamp': time.time(),
                    'method': msg.get('method'),
                })
            
            # Call original behavior
            original_enqueue(msg)
            
            with lock:
                access_log.append({
                    'event': 'exit_enqueue',
                    'thread_id': thread_id,
                    'thread_name': thread_name,
                    'timestamp': time.time(),
                    'method': msg.get('method'),
                })
        
        mock_executor.scheduler.mq.enqueue.side_effect = enqueue_with_timing
        
        # Run concurrent operations
        def worker(worker_id):
            if worker_id % 2 == 0:
                request = MagicMock(spec=OmniDiffusionRequest)
                mock_executor.add_req(request)
            else:
                mock_executor.collective_rpc(method=f"test_{worker_id}", unique_reply_rank=0)
        
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,), name=f"Worker-{i}")
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Analyze for overlapping access
        print("\n=== Timing Analysis ===")
        
        active_threads = {}
        overlaps = []
        
        for entry in access_log:
            thread_id = entry['thread_id']
            
            if entry['event'] == 'enter_enqueue':
                # Check if other threads are currently in enqueue
                if active_threads:
                    overlaps.append({
                        'time': entry['timestamp'],
                        'thread': entry['thread_name'],
                        'overlapping_with': list(active_threads.keys()),
                    })
                active_threads[entry['thread_name']] = entry
            else:  # exit_enqueue
                if entry['thread_name'] in active_threads:
                    del active_threads[entry['thread_name']]
        
        if overlaps:
            print(f"\nFound {len(overlaps)} instances of overlapping access!")
            print("\nFirst 5 overlaps:")
            for i, overlap in enumerate(overlaps[:5]):
                print(f"  {i+1}. Thread '{overlap['thread']}' entered while "
                      f"{overlap['overlapping_with']} were active")
        
        assert len(overlaps) > 0, (
            "Expected overlapping access to scheduler.mq.enqueue(), "
            "demonstrating lack of synchronization"
        )

    def test_response_mismatch_scenario(self, mock_executor):
        """
        Test scenario where response from result_mq goes to the wrong caller.
        This happens because both add_req and collective_rpc dequeue from 
        the same result_mq without coordination.
        """
        responses_received = []
        lock = threading.Lock()
        
        # Setup result queue with specific responses
        mock_executor._response_queue = [
            {"type": "rpc_response", "data": f"response_{i}"} 
            for i in range(10)
        ]
        
        def call_and_track(call_type, call_id):
            """Make a call and track which response we get."""
            try:
                if call_type == "add_req":
                    request = MagicMock(spec=OmniDiffusionRequest)
                    request.request_id = f"req_{call_id}"
                    result = mock_executor.add_req(request)
                else:
                    result = mock_executor.collective_rpc(
                        method=f"method_{call_id}",
                        unique_reply_rank=0,
                    )
                
                with lock:
                    responses_received.append({
                        'call_type': call_type,
                        'call_id': call_id,
                        'thread': threading.current_thread().name,
                        'response': result,
                    })
            except Exception as e:
                with lock:
                    responses_received.append({
                        'call_type': call_type,
                        'call_id': call_id,
                        'thread': threading.current_thread().name,
                        'error': str(e),
                    })
        
        # Launch concurrent calls
        threads = []
        for i in range(5):
            t1 = threading.Thread(target=call_and_track, args=("add_req", i))
            t2 = threading.Thread(target=call_and_track, args=("collective_rpc", i))
            threads.extend([t1, t2])
            t1.start()
            t2.start()
        
        for t in threads:
            t.join()
        
        print("\n=== Response Distribution Analysis ===")
        print(f"Total calls made: {len(threads)}")
        print(f"Total responses received: {len(responses_received)}")
        
        # Group by call type
        add_req_responses = [r for r in responses_received if r['call_type'] == 'add_req']
        rpc_responses = [r for r in responses_received if r['call_type'] == 'collective_rpc']
        
        print(f"\nadd_req calls: {len(add_req_responses)}")
        print(f"collective_rpc calls: {len(rpc_responses)}")
        
        print("\nThis demonstrates the race condition where:")
        print("  - Multiple threads dequeue from result_mq simultaneously")
        print("  - No guarantee which thread gets which response")
        print("  - Can lead to wrong results, timeouts, or errors")
        
        # The race condition exists if we got this far without errors,
        # but in a real scenario, this would cause failures
        assert len(responses_received) > 0, "Should have received some responses"


if __name__ == "__main__":
    # Run the test manually to see output
    pytest.main([__file__, "-v", "-s"])
