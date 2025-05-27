#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
쓰레기 분류 게임 모듈 (레거시 버전)

이 파일은 모듈화된 버전으로 리다이렉트합니다.
기존 코드와의 호환성을 위해 유지됩니다.
"""

import sys
import os

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 메인 모듈 임포트
from main import main

# 모듈화된 버전으로 리다이렉트
if __name__ == "__main__":
    main() 