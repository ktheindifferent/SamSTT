#!/usr/bin/env python3
"""
Detailed test to understand the race condition behavior
"""

import threading
import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class MockEngineDetailed:
    """Mock engine with detailed tracking"""
    instances_created = 0
    instances_lock = threading.Lock()
    creation_log = []
    
    def __init__(self, config=None):
        self.config = config or {}
        self.name = "mock"
        self.is_available = True
        
        # Track instance creation with detailed info
        with MockEngineDetailed.instances_lock:
            MockEngineDetailed.instances_created += 1
            instance_num = MockEngineDetailed.instances_created
            instance_id = id(self)
            thread_name = threading.current_thread().name
            timestamp = time.time()
            
            log_entry = {
                'instance_num': instance_num,
                'instance_id': instance_id,
                'thread': thread_name,
                'timestamp': timestamp
            }
            MockEngineDetailed.creation_log.append(log_entry)
            print(f"[{timestamp:.6f}] Thread {thread_name}: Created instance #{instance_num} (id: {instance_id})")
        
        # Simulate initialization work
        time.sleep(0.01)
        self.instance_id = instance_id
        self.instance_num = instance_num
    
    @classmethod
    def reset_tracking(cls):
        with cls.instances_lock:
            cls.instances_created = 0
            cls.creation_log.clear()


def test_detailed():
    """Test with detailed logging"""
    from stts.engine_manager import STTEngineManager
    
    MockEngineDetailed.reset_tracking()
    
    # Patch the ENGINES registry
    original_engines = STTEngineManager.ENGINES.copy()
    STTEngineManager.ENGINES = {'mock': MockEngineDetailed}
    
    try:
        print("=" * 70)
        print("DETAILED RACE CONDITION ANALYSIS")
        print("=" * 70)
        
        # Create manager but don't let it auto-initialize
        print("\n1. Creating manager...")
        manager = STTEngineManager(default_engine='mock', config={})
        
        print(f"\n2. Manager created. Engines in manager: {list(manager.engines.keys())}")
        print(f"   Instances created so far: {MockEngineDetailed.instances_created}")
        
        # Clear engines to simulate fresh state
        print("\n3. Clearing manager.engines to simulate fresh start...")
        manager.engines.clear()
        print(f"   Engines in manager after clear: {list(manager.engines.keys())}")
        
        # Test concurrent access
        print("\n4. Testing concurrent access with 20 threads...")
        
        num_threads = 20
        barrier = threading.Barrier(num_threads)
        results = []
        
        def get_engine_detailed(thread_id):
            thread = threading.current_thread()
            thread.name = f"Worker-{thread_id:02d}"
            
            # Log before barrier
            print(f"   [{time.time():.6f}] Thread {thread.name}: Ready at barrier")
            
            # Wait for all threads
            barrier.wait()
            
            # Log after barrier
            print(f"   [{time.time():.6f}] Thread {thread.name}: Passed barrier, calling get_engine()")
            
            try:
                engine = manager.get_engine('mock')
                print(f"   [{time.time():.6f}] Thread {thread.name}: Got engine instance #{engine.instance_num}")
                return (thread_id, id(engine), engine.instance_num)
            except Exception as e:
                print(f"   [{time.time():.6f}] Thread {thread.name}: ERROR: {e}")
                return (thread_id, None, str(e))
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(get_engine_detailed, i) for i in range(num_threads)]
            
            for future in as_completed(futures):
                results.append(future.result())
        
        elapsed = time.time() - start_time
        
        print(f"\n5. All threads completed in {elapsed:.3f} seconds")
        
        # Analyze results
        print("\n6. ANALYSIS:")
        print(f"   Total instances created: {MockEngineDetailed.instances_created}")
        print(f"   Creation log:")
        for entry in MockEngineDetailed.creation_log:
            print(f"      Instance #{entry['instance_num']}: Thread {entry['thread']} at {entry['timestamp']:.6f}")
        
        unique_ids = set(r[1] for r in results if r[1] is not None)
        unique_nums = set(r[2] for r in results if isinstance(r[2], int))
        
        print(f"\n   Unique instance IDs returned: {len(unique_ids)}")
        print(f"   Unique instance numbers returned: {unique_nums}")
        
        if MockEngineDetailed.instances_created > 1:
            # Find time gap between creations
            if len(MockEngineDetailed.creation_log) > 1:
                time_gaps = []
                for i in range(1, len(MockEngineDetailed.creation_log)):
                    gap = MockEngineDetailed.creation_log[i]['timestamp'] - MockEngineDetailed.creation_log[i-1]['timestamp']
                    time_gaps.append(gap)
                print(f"\n   Time gaps between instance creations: {[f'{gap:.6f}s' for gap in time_gaps]}")
        
        print("\n7. RESULT:")
        if MockEngineDetailed.instances_created == 1:
            print("   ✓ SUCCESS: Only 1 instance created - No race condition!")
        else:
            print(f"   ✗ FAILURE: {MockEngineDetailed.instances_created} instances created - Race condition present!")
            
            # Identify which threads created instances
            creating_threads = [entry['thread'] for entry in MockEngineDetailed.creation_log]
            print(f"   Threads that created instances: {creating_threads}")
        
        return MockEngineDetailed.instances_created == 1
        
    finally:
        STTEngineManager.ENGINES = original_engines


if __name__ == "__main__":
    success = test_detailed()
    sys.exit(0 if success else 1)