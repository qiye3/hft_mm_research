"""按明确日期下载已授权网盘中的 BTCUSDT 文件，供数据质检与研究生产使用。

download_day 保存L2盘口或25档快照，download_file 支持分段续传。
输入为日期、来源类别和私有凭证文件，输出为原始文件与不含凭证的来源清单。
"""

import argparse
import concurrent.futures
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

REMOTE_ROOT = "/STUDY/Binance-BTC-Futures"


# 【非核心】通过官方接口读取元数据，接口错误仅显示错误码。
def request_metadata(resource: str, parameters: dict, token: str) -> dict:
    """输入资源名、接口参数和私有 token，返回百度 JSON 响应。

    用于目录和下载授权查询，HTTP 错误统一去掉包含凭证的请求地址。
    """
    url = "https://pan.baidu.com/rest/2.0/xpan/" + resource
    url += "?" + urlencode(parameters | {"access_token": token})
    try:
        with urlopen(Request(url, headers={"User-Agent": "pan.baidu.com"}), timeout=60) as response:
            result = json.load(response)
    except (HTTPError, URLError) as error:
        raise RuntimeError(f"百度元数据请求失败，类型 {type(error).__name__}") from None
    if result.get("errno") != 0:
        raise RuntimeError(f"百度接口错误码 {result.get('errno')}")
    return result


# 【核心逻辑2】将一个网盘文件下载到临时文件，完整后再替换正式文件名。
def download_file(entry: dict, destination: Path, token: str) -> dict:
    """输入网盘目录项和目标文件，返回来源、长度及下载耗时。

    每次完整接收 8 MiB 才追加到 .part，失败重试时重新获取授权链接。
    已完成文件仅检查长度，错误信息不包含签名链接或账号凭证。
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    expected = entry["size"]
    partial = destination.with_name(destination.name + ".part")
    if not destination.exists():
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected:
            raise ValueError(f"续传文件超出来源长度 {destination}")
        failures = 0
        link = None
        with partial.open("ab") as output:
            while offset < expected:
                if link is None:
                    metadata = request_metadata("multimedia", {
                        "method": "filemetas", "fsids": json.dumps([entry["fs_id"]]), "dlink": 1,
                    }, token)
                    link = metadata["list"][0]["dlink"]
                end = min(offset + 8 * 1024 * 1024, expected) - 1
                request = Request(link + "&" + urlencode({"access_token": token}), headers={
                    "User-Agent": "pan.baidu.com", "Range": f"bytes={offset}-{end}",
                })
                try:
                    with urlopen(request, timeout=90) as response:
                        if response.status != 206:
                            raise ValueError("下载接口未返回字节区间")
                        content = response.read(end - offset + 2)
                    if len(content) != end - offset + 1:
                        raise ValueError("下载区间长度不匹配")
                except (HTTPError, URLError, TimeoutError, ValueError) as error:
                    failures += 1
                    if failures >= 5:
                        raise RuntimeError(f"下载失败 {destination.name}，类型 {type(error).__name__}") from None
                    link = None
                    time.sleep(2 * failures)
                    continue
                output.write(content)
                output.flush()
                offset = end + 1
                failures = 0
        partial.rename(destination)
    if destination.stat().st_size != expected:
        raise ValueError(f"来源长度不一致 {destination}")
    return {
        "source": entry["path"], "bytes": expected, "fs_id": entry["fs_id"],
        "local_file": str(destination), "download_seconds": time.monotonic() - started,
    }


# 【核心逻辑1】按日期定位并下载L2增量盘口或25档快照文件。
def download_day(date: str, kind: str, output_root: Path, token: str, source_extension: str = "zst") -> None:
    """输入日期、来源类别、输出根目录和授权，写入该日期文件及 sources.json。

    book下载L2增量，snapshot下载25档快照。普通成交由官方日档提供。
    source_extension显式指定目录中的zst、zstd或xz后缀，不搜索替代文件名。
    后缀用于定位来源，实际压缩编码由数据内容决定。
    文件名明确区分来源，避免网盘中的同名文件互相覆盖。
    """
    calendar_path = date.replace("-", "/")
    sources = {
        "book": [(f"L2增量 2020-2026/BTCUSDT/{calendar_path}", f"BTCUSDT.csv.{source_extension}", f"book.csv.{source_extension}")],
        "snapshot": [(f"25-orderbook 2020-2026/BTCUSDT/{calendar_path}", f"BTCUSDT.csv.{source_extension}", f"snapshot25.csv.{source_extension}")],
    }
    destination = output_root / f"baidu_{date}"
    records = []
    for directory, filename, local_name in sources[kind]:
        listing = request_metadata("file", {
            "method": "list", "dir": f"{REMOTE_ROOT}/{directory}", "start": 0, "limit": 1000,
        }, token)["list"]
        matches = [entry for entry in listing if entry["server_filename"] == filename]
        if len(matches) != 1:
            raise ValueError(f"未找到唯一来源 {directory}/{filename}")
        records.append(download_file(matches[0], destination / local_name, token))
    (destination / f"sources_{kind}.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print(f"下载完成 {date} {kind}", flush=True)


# 【非核心】启动日期下载任务，账号凭证仅在当前进程内使用。
def main() -> None:
    """读取日期列表、来源类别、并发上限及凭证文件，执行可恢复下载。

    并发仅用于独立日期的网络传输，行情计算继续使用单线程引擎。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="+", required=True)
    parser.add_argument("--kind", choices=["book", "snapshot"], required=True)
    parser.add_argument("--output-root", type=Path, default=Path("workspace/data/raw"))
    parser.add_argument("--credential-file", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--source-extension", choices=["zst", "zstd", "xz"], default="zst",
        help="目录中的原始文件后缀，实际压缩编码由数据文件头确定",
    )
    arguments = parser.parse_args()
    credential = arguments.credential_file.read_text()
    match = re.search(r"access_token=([^&#\s`<>]+)", credential)
    if match is None:
        raise ValueError("凭证文件中缺少 access_token")
    with concurrent.futures.ThreadPoolExecutor(max_workers=arguments.jobs) as pool:
        futures = [pool.submit(download_day, date, arguments.kind, arguments.output_root, match.group(1), arguments.source_extension)
                   for date in arguments.dates]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
