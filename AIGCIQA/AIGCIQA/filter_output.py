#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re

# 需要过滤的警告模式
WARNING_PATTERNS = [
    r'UserWarning:',
    r'FutureWarning:',
    r'DeprecationWarning:',
    r'Warning:',
    r'warnings.warn',
]

def should_keep_line(line):
    """判断是否应该保留这一行输出"""
    # 检查是否包含需要过滤的警告
    for pattern in WARNING_PATTERNS:
        if re.search(pattern, line):
            return False
    return True

def main():
    """主函数：读取标准输入并过滤输出"""
    for line in sys.stdin:
        if should_keep_line(line):
            sys.stdout.write(line)
            sys.stdout.flush()

if __name__ == '__main__':
    main() 