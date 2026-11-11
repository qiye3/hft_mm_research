"""夜间守护器的有限恢复、独立任务保留及截止凭证清理验证。"""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

specification = importlib.util.spec_from_file_location(
    'research_watchdog', Path(__file__).resolve().parents[1] / 'scripts/watch_research_night.py')
watchdog = importlib.util.module_from_spec(specification)
specification.loader.exec_module(watchdog)


class ResearchWatchdogTests(unittest.TestCase):
    # 【非核心】训练日仍有未完成机会时保留缓存，全部完成才允许下载队列释放。
    def test_release_waits_for_all_cases(self):
        """消费标记依赖完整预登记案例集合，不能因部分回放结束提前删除事件。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'e4').mkdir()
            cases = [{'id': name, 'date': '2025-10-01'} for name in ['first', 'second']]
            (root / 'e4/opportunity_case_plan.json').write_text(json.dumps(cases))
            for name in ['first', 'second']:
                directory = root / 'e4/runs' / name
                directory.mkdir(parents=True)
            (root / 'e4/runs/first/execution_metrics.json').write_text('{}')
            marker = root / 'l2/consumed/2025-10-01.json'
            watchdog.release_completed_opportunity_days(root)
            self.assertFalse(marker.exists())
            (root / 'e4/runs/second/execution_metrics.json').write_text('{}')
            watchdog.release_completed_opportunity_days(root)
            self.assertEqual(json.loads(marker.read_text())['completed_cases'], 2)

    # 【非核心】存活协调器不重复启动，异常退出次数达到三次后不再重启。
    def test_existing_process_and_restart_budget(self):
        """以替身进程验证恢复边界，测试不会启动下载或训练。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'night').mkdir()
            (root / 'night/supervisor.pid').write_text('123')
            restarts = {'research': 0, 'archive': 0}
            with patch.object(watchdog, 'owned_process', return_value=True), patch.object(watchdog.subprocess, 'Popen') as launch:
                watchdog.ensure_coordinator(root, root, root / 'config.yaml', root / 'private', 'research', restarts)
                launch.assert_not_called()
            restarts['research'] = 3
            with patch.object(watchdog, 'owned_process', return_value=False), patch.object(watchdog.subprocess, 'Popen') as launch:
                watchdog.ensure_coordinator(root, root, root / 'config.yaml', root / 'private', 'research', restarts)
                launch.assert_not_called()

    # 【非核心】即使下载清理出现错误，截止处理也删除独立临时凭证。
    def test_deadline_always_removes_credential(self):
        """模拟已过期实验及清理错误，私有凭证必须消失且错误保持可见。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            credential = root / 'private'
            credential.write_text('test-only')
            config = {'stop_at': '2000-01-01T08:00:00+08:00'}
            with patch.object(watchdog, 'context', return_value=(root, config, root)), \
                    patch.object(watchdog, 'close_archive_session', side_effect=ValueError('ownership mismatch')):
                with self.assertRaisesRegex(ValueError, 'ownership mismatch'):
                    watchdog.watch_night(root / 'config.yaml', credential)
            self.assertFalse(credential.exists())


if __name__ == '__main__':
    unittest.main()
