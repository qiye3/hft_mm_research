"""Linux任务虚拟分配上限和匿名内存监控阈值独立配置。"""

import json
import os
import sys

import pytest

from mm_research.overnight import GIB, start_task


# 【非核心】真实子进程继承独立分配上限，默认配置维持原资源约束。
@pytest.mark.skipif(sys.platform != 'linux', reason='Linux服务器资源控制')
@pytest.mark.parametrize('allocation_limit,expected', [(None, 1), (2, 2)])
def test_allocation_limit_is_independent(tmp_path, allocation_limit, expected):
    """子进程输出实际RLIMIT_DATA，父任务的匿名内存阈值始终为1GiB。"""
    task = {
        'id': 'limits', 'threads': 1, 'memory_limit_gib': 1,
        'command': ['/usr/bin/python3', '-c',
                    'import resource\nimport json\nprint(json.dumps(resource.getrlimit(resource.RLIMIT_DATA)))'],
    }
    if allocation_limit is not None:
        task['allocation_limit_gib'] = allocation_limit
    record = start_task(tmp_path, tmp_path, task)
    os.waitpid(record['pid'], 0)
    limits = json.loads((tmp_path / 'limits/run.log').read_text().splitlines()[0])
    assert limits == [expected * GIB, expected * GIB]
    assert task['memory_limit_gib'] == 1
    assert (tmp_path / 'limits/exit_code.txt').read_text().strip() == '0'
