"""检查正式回放与提前准备任务同时请求同一天时的发布行为。"""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from mm_research.strategies.cost_native import prepare_native_date


class NativePreparationTests(unittest.TestCase):
    # 【非核心】两个调用方竞争同一天时，只执行一次转换并共享正式标记。
    def test_concurrent_preparation_converts_once(self):
        """模拟耗时转换，确认等待文件锁的调用方复用先发布的日期。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'output'
            (output / 'days').mkdir(parents=True)
            date = '2026-06-30'
            raw = output / 'raw' / date / 'book.csv.zst'
            raw.parent.mkdir(parents=True)
            raw.write_bytes(b'book')
            config = {'source': str(root / 'source')}
            inventory = {date: {'server_filename': 'BTCUSDT.csv.zst', 'size': 4}}
            ready = threading.Barrier(2)

            # 【非核心】等待期间另一路已进入准备函数，放大重复转换的竞争窗口。
            def finish_conversion(*arguments, **options):
                """发布模拟事件清单，真实日期标记仍由生产函数写入。"""
                time.sleep(0.1)
                destination = output / 'event_cache' / date / 'events'
                destination.mkdir(parents=True)
                (destination / 'manifest.json').write_text('{}\n')

            # 【非核心】使两个调用方在日期锁之前同时开始。
            def prepare_date():
                """使用生产函数处理同一日期和输出路径。"""
                ready.wait()
                prepare_native_date(root / 'config.yaml', config, output, inventory, date)

            with patch('mm_research.strategies.cost_native.subprocess.run', side_effect=finish_conversion) as conversion:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [executor.submit(prepare_date) for _ in range(2)]
                    for future in futures:
                        future.result()
            self.assertEqual(conversion.call_count, 1)
            marker = json.loads((output / 'days' / f'{date}.json').read_text())
            self.assertTrue(marker['created_here'])
            self.assertTrue(Path(marker['manifest']).is_file())


if __name__ == '__main__':
    unittest.main()
