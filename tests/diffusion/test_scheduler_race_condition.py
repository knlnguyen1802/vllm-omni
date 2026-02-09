"""
Test demonstrating race condition in scheduler using real classes with mocked queues.

This test uses the actual Scheduler and MultiprocDiffusionExecutor implementations
but mocks the MessageQueue to avoid multiprocessing complexity while still
demonstrating the real race condition bug.

The bug: Both collective_rpc() and add_req() access scheduler.mq without
synchronization, causing race conditions when called from multiple threads.

To run:
    pytest tests/diffusion/test_scheduler_race_condition.py -v -s
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.executor.multiproc_executor import (
    MultiprocDiffusionExecutor,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.scheduler import Scheduler


class MockMessageQueue:
    """
    Mock MessageQueue that tracks concurrent access to demonstrate race condition.
    """
    
    def __init__(self):
        self.messages = []
        self.responses = []
        self.access_log = []
        self.currently_accessing = {}  # Track which threads are currently in enqueue/dequeue
        self._log_lock = threading.Lock()
        
    def enqueue(self, msg):
        """Enqueue without synchronization - this is the bug!"""
        thread_id = threading.current_thread().ident
        thread_name = threading.current_thread().name
        
        with self._log_lock:
            self.access_log.append({
                'action': 'enqueue_start',
                'thread_id': thread_id,
                'thread_name': thread_name,
                'timestamp': time.time(),
                'msg_type': msg.get('type'),
                'method': msg.get('method'),
            })
            self.currently_accessing[thread_id] = thread_name
        
        # Simulate some processing time where race can occur
        time.sleep(0.001)
        
        # The actual enqueue - this can interleave with other threads
        self.messages.append({
            'msg': msg,
            'thread': thread_name,
            'thread_id': thread_id,
        })
        
        with self._log_lock:
            self.access_log.append({
                'action': 'enqueue_end',
                'thread_id': thread_id,
                'thread_name': thread_name,
                'timestamp': time.time(),
            })
            if thread_id in self.currently_accessing:
                del self.currently_accessing[thread_id]
    
    def dequeue(self, timeout=None):
        """Dequeue - can return wrong response to wrong thread!"""
        thread_id = threading.current_thread().ident
        thread_name = threading.current_thread().name
        
        with self._log_lock:
            self.access_log.append({
                'action': 'dequeue_start',
                'thread_id': thread_id,
                'thread_name': thread_name,
                'timestamp': time.time(),
            })
        
        # Simulate processing
        time.sleep(0.001)
        
        # Return response - but may not be for this caller!
        if self.responses:
            response = self.responses.pop(0)
        else:
            response = {'status': 'ok', 'result': f'response_for_{thread_name}'}
        
        with self._log_lock:
            self.access_log.append({
                'action': 'dequeue_end',
                'thread_id': thread_id,
                'thread_name': thread_name,
                'timestamp': time.time(),
                'got_response': str(response)[:50],
            })
        
        return response
    
    def export_handle(self):
        """Mock export_handle"""
        return "mock_handle"
    
    @staticmethod
    def create_from_handle(handle, rank):
        """Mock create_from_handle"""
        return MockMessageQueue()
    
    @property
    def closed(self):
        return False


class TestSchedulerRaceCondition:
    """Test race condition using real Scheduler and Executor classes."""
    
    @pytest.fixture
    def setup_executor(self, monkeypatch):
        """Setup executor with real classes but mocked message queues."""
        
        # Mock MessageQueue class where it's used (in scheduler module)
        monkeypatch.setattr(
            'vllm_omni.diffusion.scheduler.MessageQueue',
            MockMessageQueue
        )
        
        # Create real scheduler
        scheduler = Scheduler()
        od_config = MagicMock(spec=OmniDiffusionConfig)
        od_config.num_gpus = 2
        scheduler.initialize(od_config)
        
        # Setup result queue
        mock_result_mq = MockMessageQueue()
        scheduler.result_mq = mock_result_mq
        
        # Create executor and inject scheduler (skip multiproc setup)
        executor = MultiprocDiffusionExecutor.__new__(MultiprocDiffusionExecutor)
        executor.scheduler = scheduler
        executor.od_config = od_config
        executor._closed = False
        executor._processes = []
        
        return executor, scheduler
    
    def test_concurrent_add_req_and_collective_rpc(self, setup_executor):
        """
        Test showing how concurrent add_req and collective_rpc cause race condition.
        
        Both methods call scheduler.mq.enqueue() without synchronization.
        This causes:
        1. Message interleaving in the queue
        2. Wrong thread receiving responses from result_mq.dequeue()
        3. Potential timeouts or incorrect results
        """
        executor, scheduler = setup_executor
        
        num_iterations = 10
        results = []
        errors = []
        lock = threading.Lock()
        
        def call_add_req(i):
            """Call add_req - uses scheduler.add_req() which calls mq.enqueue()"""
            try:
                request = MagicMock(spec=OmniDiffusionRequest)
                request.request_id = f"add_req_{i}"
                
                # Pre-populate response for this request
                scheduler.result_mq.responses.append({
                    'status': 'ok',
                    'request_id': f"add_req_{i}",
                })
                
                result = executor.add_req(request)
                
                with lock:
                    results.append(('add_req', i, result))
            except Exception as e:
                with lock:
                    errors.append(('add_req', i, str(e)))
        
        def call_collective_rpc(i):
            """Call collective_rpc - directly calls mq.enqueue()"""
            try:
                # Pre-populate response for this RPC
                scheduler.result_mq.responses.append({
                    'status': 'ok',
                    'method': f"test_method_{i}",
                })
                
                result = executor.collective_rpc(
                    method=f"test_method_{i}",
                    timeout=5.0,
                    unique_reply_rank=0,
                )
                
                with lock:
                    results.append(('collective_rpc', i, result))
            except Exception as e:
                with lock:
                    errors.append(('collective_rpc', i, str(e)))
        
        print("\n" + "="*60)
        print("Testing concurrent add_req and collective_rpc calls")
        print("="*60)
        
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = []
            for i in range(num_iterations):
                futures.append(pool.submit(call_add_req, i))
                futures.append(pool.submit(call_collective_rpc, i))
            
            for f in futures:
                f.result()
        
        # Analyze the race condition
        print(f"\nResults: {len(results)} successful, {len(errors)} errors")
        
        # Check message queue access log
        mq_log = scheduler.mq.access_log
        result_mq_log = scheduler.result_mq.access_log
        
        print(f"\nMessage queue accesses: {len(mq_log)}")
        print(f"Result queue accesses: {len(result_mq_log)}")
        
        # Find overlapping accesses (proof of race condition)
        overlaps = self._find_overlapping_accesses(mq_log)
        
        print(f"\n{'='*60}")
        print(f"RACE CONDITION DETECTED: {len(overlaps)} overlapping accesses")
        print(f"{'='*60}")
        
        if overlaps:
            print("\nFirst 5 overlapping accesses:")
            for i, overlap in enumerate(overlaps[:5]):
                print(f"\n{i+1}. {overlap['thread1']} and {overlap['thread2']}")
                print(f"   Both accessing mq.enqueue() at the same time")
                print(f"   Thread1: {overlap['method1']}, Thread2: {overlap['method2']}")
        
        # Show message ordering
        print(f"\n{'='*60}")
        print("Message enqueueing order (showing interleaving):")
        print(f"{'='*60}")
        
        enqueues = [log for log in mq_log if log['action'] == 'enqueue_start']
        for i, log in enumerate(enqueues[:10]):
            print(f"{i+1}. {log['thread_name']:20s} - {log['method']}")
        
        # Check dequeue mismatches
        dequeues = [log for log in result_mq_log if log['action'] == 'dequeue_end']
        print(f"\n{'='*60}")
        print("Response dequeuing (can go to wrong thread):")
        print(f"{'='*60}")
        
        for i, log in enumerate(dequeues[:10]):
            print(f"{i+1}. {log['thread_name']:20s} got: {log.get('got_response', 'N/A')}")
        
        # Assert that race condition exists
        assert len(overlaps) > 0, (
            "Expected concurrent access to scheduler.mq.enqueue(), "
            "demonstrating the race condition"
        )
        
        print(f"\n{'='*60}")
        print("CONCLUSION: Race Condition Confirmed!")
        print(f"{'='*60}")
        print("\nThe bug:")
        print("  File: vllm_omni/diffusion/executor/multiproc_executor.py")
        print("  - Line 132: add_req() -> scheduler.add_req() -> mq.enqueue()")
        print("  - Line 161: collective_rpc() -> mq.enqueue()")
        print("  Both access scheduler.mq WITHOUT synchronization!")
        print("\n  File: vllm_omni/diffusion/scheduler.py")  
        print("  - Line 53: add_req() calls mq.enqueue()")
        print("  - Line 62: add_req() calls result_mq.dequeue()")
        print("\nFix needed:")
        print("  Add a lock to protect mq.enqueue() and result_mq.dequeue() calls")
        print("  OR ensure single-threaded access to scheduler methods")
    
    def _find_overlapping_accesses(self, access_log: List[Dict]) -> List[Dict]:
        """Find instances where multiple threads are in enqueue simultaneously."""
        overlaps = []
        active = {}  # thread_id -> log entry
        
        for log in access_log:
            thread_id = log['thread_id']
            
            if log['action'] == 'enqueue_start':
                # Check if other threads are currently in enqueue
                for other_id, other_log in active.items():
                    if other_id != thread_id:
                        overlaps.append({
                            'thread1': log['thread_name'],
                            'thread2': other_log['thread_name'],
                            'method1': log.get('method'),
                            'method2': other_log.get('method'),
                            'timestamp': log['timestamp'],
                        })
                active[thread_id] = log
            
            elif log['action'] == 'enqueue_end':
                if thread_id in active:
                    del active[thread_id]
        
        return overlaps
    
    def test_response_mismatch_demonstration(self, setup_executor):
        """
        Demonstrate how responses can go to the wrong caller.
        """
        executor, scheduler = setup_executor
        
        # Setup specific responses
        responses_sent = [
            {'id': 'response_A', 'data': 'for_add_req_0'},
            {'id': 'response_B', 'data': 'for_rpc_0'},
            {'id': 'response_C', 'data': 'for_add_req_1'},
            {'id': 'response_D', 'data': 'for_rpc_1'},
        ]
        
        scheduler.result_mq.responses = responses_sent.copy()
        
        received_responses = []
        lock = threading.Lock()
        
        def worker(call_type, call_id):
            if call_type == 'add_req':
                request = MagicMock(spec=OmniDiffusionRequest)
                result = executor.add_req(request)
            else:
                result = executor.collective_rpc(method=f"test_{call_id}", unique_reply_rank=0)
            
            with lock:
                received_responses.append({
                    'caller': f"{call_type}_{call_id}",
                    'thread': threading.current_thread().name,
                    'response': result,
                })
        
        # Launch concurrent calls
        with ThreadPoolExecutor(max_workers=4) as pool:
            pool.submit(worker, 'add_req', 0)
            pool.submit(worker, 'collective_rpc', 0)
            pool.submit(worker, 'add_req', 1)
            pool.submit(worker, 'collective_rpc', 1)
        
        print(f"\n{'='*60}")
        print("Response Mismatch Analysis")
        print(f"{'='*60}")
        print(f"\nSent {len(responses_sent)} responses:")
        for r in responses_sent:
            print(f"  - {r['id']}: {r['data']}")
        
        print(f"\nReceived by callers:")
        for r in received_responses:
            response_id = r['response'].get('id', 'unknown')
            print(f"  - {r['caller']:20s} got: {response_id}")
        
        print("\nWithout proper synchronization, responses can be received")
        print("by the wrong caller, causing incorrect results or timeouts!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
