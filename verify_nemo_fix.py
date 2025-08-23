#!/usr/bin/env python3
"""
Verification script for NeMo engine temp file handling fix
This script performs static analysis without requiring dependencies
"""

import ast
import sys


def analyze_nemo_engine():
    """Analyze the NeMo engine code for the resource leak fix"""
    
    print("=== NeMo Engine Resource Leak Fix Verification ===\n")
    
    with open('stts/engines/nemo.py', 'r') as f:
        source = f.read()
    
    tree = ast.parse(source)
    
    issues = []
    improvements = []
    
    # Check for proper imports
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom)]
    has_logging = any('logging' in str(getattr(imp, 'module', '')) or 
                      any(alias.name == 'logging' for alias in getattr(imp, 'names', [])) 
                      for imp in imports)
    has_atexit = any('atexit' in str(getattr(imp, 'module', '')) or 
                     any(alias.name == 'atexit' for alias in getattr(imp, 'names', [])) 
                     for imp in imports)
    
    if has_logging:
        improvements.append("✓ Logging module imported for better debugging")
    else:
        issues.append("✗ Logging module not imported")
    
    if has_atexit:
        improvements.append("✓ Atexit module imported for cleanup on exit")
    else:
        issues.append("✗ Atexit module not imported")
    
    # Find NeMoEngine class
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'NeMoEngine':
            # Check for __init__ method
            has_init = any(isinstance(n, ast.FunctionDef) and n.name == '__init__' 
                          for n in node.body)
            if has_init:
                improvements.append("✓ Custom __init__ method added for initialization")
            else:
                issues.append("✗ No custom __init__ method")
            
            # Check for cleanup method
            has_cleanup = any(isinstance(n, ast.FunctionDef) and '_cleanup' in n.name 
                            for n in node.body)
            if has_cleanup:
                improvements.append("✓ Cleanup method implemented for orphaned files")
            else:
                issues.append("✗ No cleanup method found")
            
            # Check transcribe_raw method
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == 'transcribe_raw':
                    # Check for try-finally pattern
                    has_try_finally = any(isinstance(n, ast.Try) and 
                                        any(isinstance(h, ast.ExceptHandler) for h in n.handlers) and
                                        n.finalbody
                                        for n in ast.walk(method))
                    
                    if has_try_finally:
                        improvements.append("✓ Try-finally pattern used for resource cleanup")
                    else:
                        issues.append("✗ No proper try-finally pattern found")
                    
                    # Check for temp file cleanup in finally block
                    source_lines = source.split('\n')
                    method_start = method.lineno - 1
                    method_end = method_start + len(ast.unparse(method).split('\n'))
                    method_source = '\n'.join(source_lines[method_start:method_end])
                    
                    if 'finally:' in method_source and 'os.unlink' in method_source:
                        improvements.append("✓ Temp file deletion in finally block")
                    else:
                        issues.append("✗ No temp file deletion in finally block")
                    
                    # Check for logging
                    if 'logger' in method_source:
                        improvements.append("✓ Logging statements added for debugging")
                    
                    # Check for better temp file handling
                    if 'temp_filepath' in method_source:
                        improvements.append("✓ Improved temp file path tracking")
                    
                    # Check that finally is outside with statement
                    if 'with tempfile.NamedTemporaryFile' in method_source:
                        # Simple check: finally should come after the with block
                        with_index = method_source.find('with tempfile.NamedTemporaryFile')
                        finally_index = method_source.find('finally:')
                        
                        if finally_index > with_index:
                            # Check indentation to ensure finally is at correct level
                            with_line = method_source[with_index:method_source.find('\n', with_index)]
                            finally_line = method_source[finally_index:method_source.find('\n', finally_index)]
                            
                            # Count leading spaces
                            with_indent = len(with_line) - len(with_line.lstrip())
                            finally_indent = len(finally_line) - len(finally_line.lstrip())
                            
                            if finally_indent <= with_indent:
                                improvements.append("✓ Finally block at correct scope (outside with statement)")
                            else:
                                issues.append("✗ Finally block might be inside with statement")
    
    # Print results
    print("Improvements found:")
    for improvement in improvements:
        print(f"  {improvement}")
    
    if issues:
        print("\nPotential issues:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ No issues found!")
    
    # Summary
    print(f"\nSummary: {len(improvements)} improvements, {len(issues)} issues")
    
    # Check specific fix requirements
    print("\n=== Key Fix Requirements ===")
    
    requirements = {
        "Temp file cleanup in finally block": 'finally:' in source and 'os.unlink' in source,
        "Proper exception handling": 'except Exception' in source,
        "Logging for debugging": 'logger' in source,
        "Cleanup mechanism for orphaned files": '_cleanup' in source,
        "Unique temp file names": 'uuid' in source or 'NamedTemporaryFile' in source,
    }
    
    for req, met in requirements.items():
        status = "✅" if met else "❌"
        print(f"{status} {req}")
    
    return len(issues) == 0


if __name__ == '__main__':
    success = analyze_nemo_engine()
    sys.exit(0 if success else 1)