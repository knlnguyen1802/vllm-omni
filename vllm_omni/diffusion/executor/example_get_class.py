"""
Example: All Ways to Specify Custom Executors

This demonstrates all the different ways you can specify an executor
using the new DiffusionExecutor.get_class() pattern.
"""

from vllm_omni.diffusion import DiffusionEngine
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.executor import DiffusionExecutor


def example_1_default():
    """Example 1: Default executor (None)"""
    print("\n" + "=" * 60)
    print("Example 1: Default Executor")
    print("=" * 60)
    
    config = OmniDiffusionConfig(
        model_class_name="Qwen3Omni",
        # executor_class is None by default
    )
    
    # Get the executor class
    executor_class = DiffusionExecutor.get_class(config)
    print(f"Executor class: {executor_class.__name__}")
    print("Expected: MultiProcDiffusionExecutor")


def example_2_shorthand_multiproc():
    """Example 2: Using 'multiproc' shorthand"""
    print("\n" + "=" * 60)
    print("Example 2: Shorthand 'multiproc'")
    print("=" * 60)
    
    config = OmniDiffusionConfig(
        model_class_name="Qwen3Omni",
        executor_class="multiproc",  # Shorthand
    )
    
    executor_class = DiffusionExecutor.get_class(config)
    print(f"Executor class: {executor_class.__name__}")
    print("Expected: MultiProcDiffusionExecutor")


def example_3_shorthand_mp():
    """Example 3: Using 'mp' shorthand"""
    print("\n" + "=" * 60)
    print("Example 3: Shorthand 'mp'")
    print("=" * 60)
    
    config = OmniDiffusionConfig(
        model_class_name="Qwen3Omni",
        executor_class="mp",  # Shorthand alias
    )
    
    executor_class = DiffusionExecutor.get_class(config)
    print(f"Executor class: {executor_class.__name__}")
    print("Expected: MultiProcDiffusionExecutor")


def example_4_shorthand_external():
    """Example 4: Using 'external' shorthand"""
    print("\n" + "=" * 60)
    print("Example 4: Shorthand 'external'")
    print("=" * 60)
    
    config = OmniDiffusionConfig(
        model_class_name="Qwen3Omni",
        executor_class="external",  # Shorthand for template
    )
    
    executor_class = DiffusionExecutor.get_class(config)
    print(f"Executor class: {executor_class.__name__}")
    print("Expected: ExternalDiffusionExecutor")


def example_5_shorthand_http():
    """Example 5: Using 'http' shorthand"""
    print("\n" + "=" * 60)
    print("Example 5: Shorthand 'http'")
    print("=" * 60)
    
    config = OmniDiffusionConfig(
        model_class_name="Qwen3Omni",
        executor_class="http",  # Shorthand for HTTP executor
    )
    
    executor_class = DiffusionExecutor.get_class(config)
    print(f"Executor class: {executor_class.__name__}")
    print("Expected: HTTPDiffusionExecutor")


def example_6_full_path():
    """Example 6: Using full qualified path"""
    print("\n" + "=" * 60)
    print("Example 6: Full Qualified Path")
    print("=" * 60)
    
    config = OmniDiffusionConfig(
        model_class_name="Qwen3Omni",
        executor_class="vllm_omni.diffusion.executor.multiproc_executor.MultiProcDiffusionExecutor",
    )
    
    executor_class = DiffusionExecutor.get_class(config)
    print(f"Executor class: {executor_class.__name__}")
    print("Expected: MultiProcDiffusionExecutor")


def example_7_class_type():
    """Example 7: Using direct class reference"""
    print("\n" + "=" * 60)
    print("Example 7: Direct Class Reference")
    print("=" * 60)
    
    from vllm_omni.diffusion.executor import MultiProcDiffusionExecutor
    
    config = OmniDiffusionConfig(
        model_class_name="Qwen3Omni",
        executor_class=MultiProcDiffusionExecutor,  # Direct class
    )
    
    executor_class = DiffusionExecutor.get_class(config)
    print(f"Executor class: {executor_class.__name__}")
    print("Expected: MultiProcDiffusionExecutor")


def example_8_error_handling():
    """Example 8: Error handling"""
    print("\n" + "=" * 60)
    print("Example 8: Error Handling")
    print("=" * 60)
    
    # Test invalid type
    try:
        config = OmniDiffusionConfig(
            model_class_name="Qwen3Omni",
            executor_class=123,  # Invalid type
        )
        DiffusionExecutor.get_class(config)
        print("ERROR: Should have raised TypeError")
    except TypeError as e:
        print(f"✓ Caught expected TypeError: {e}")
    
    # Test invalid class (not a subclass)
    try:
        class NotAnExecutor:
            pass
        
        config = OmniDiffusionConfig(
            model_class_name="Qwen3Omni",
            executor_class=NotAnExecutor,  # Not a DiffusionExecutor
        )
        DiffusionExecutor.get_class(config)
        print("ERROR: Should have raised TypeError")
    except TypeError as e:
        print(f"✓ Caught expected TypeError: {e}")
    
    # Test invalid string path
    try:
        config = OmniDiffusionConfig(
            model_class_name="Qwen3Omni",
            executor_class="nonexistent.module.BadClass",
        )
        DiffusionExecutor.get_class(config)
        print("ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Caught expected ValueError: {str(e)[:100]}...")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("DiffusionExecutor.get_class() Examples")
    print("=" * 60)
    
    examples = [
        example_1_default,
        example_2_shorthand_multiproc,
        example_3_shorthand_mp,
        example_4_shorthand_external,
        example_5_shorthand_http,
        example_6_full_path,
        example_7_class_type,
        example_8_error_handling,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"ERROR in {example.__name__}: {e}")
    
    print("\n" + "=" * 60)
    print("Summary: Supported executor_class values")
    print("=" * 60)
    print("1. None (default)           → MultiProcDiffusionExecutor")
    print("2. 'multiproc' or 'mp'      → MultiProcDiffusionExecutor")
    print("3. 'external'               → ExternalDiffusionExecutor")
    print("4. 'http'                   → HTTPDiffusionExecutor")
    print("5. 'path.to.CustomExecutor' → Resolved via import")
    print("6. CustomExecutor           → Direct class reference")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
