"""研究Parquet的文件锁、元数据和原子发布，供Features与Factors共用。

锁文件在相邻.locks目录长期保留，避免替换数据文件后锁住旧inode。
调用方按输入共享、输出独占的方式取得锁，并在实际物化查询期间持有它。
"""

from contextlib import contextmanager, ExitStack
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import tempfile

import polars as pl
import pyarrow.parquet as pq

KEY_COLUMNS = ["symbol", "date", "timestamp_ns"]
METADATA_KEY = "mm_research"
SCHEMA_VERSION = "single_series_v1"


# 【非核心】按固定顺序取得稳定文件锁，使不同输出能够独立更新。
@contextmanager
def file_locks(filenames: list[Path], shared: bool = False):
    """输入数据文件列表和锁模式，在上下文退出时释放全部锁。

    固定顺序避免多个输入交叉加锁。锁文件保留，进程退出会释放操作系统锁。
    """
    with ExitStack() as descriptors:
        for filename in sorted({path.resolve() for path in filenames}):
            folder = filename.parent / ".locks"
            folder.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(folder / f"{filename.name}.lock", os.O_CREAT | os.O_RDWR, 0o600)
            descriptors.callback(os.close, descriptor)
            fcntl.flock(descriptor, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        yield


# 【非核心】读取当前单变量研究格式的定义，供输入一致性和产物复用检查。
def read_metadata(filename: Path) -> dict:
    """输入研究Parquet，返回内嵌定义，未知格式明确报错。"""
    metadata = json.loads(pl.read_parquet_metadata(filename)[METADATA_KEY])
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"研究文件格式与当前单变量格式不同 {filename}")
    return metadata


# 【非核心】复用同定义的完整文件，定义变化要求新版本或显式重建。
def artifact_ready(filename: Path, metadata: dict, rebuild: bool) -> bool:
    """输入目标、预期定义及重建标记，返回是否可以直接复用。"""
    if not filename.exists() or rebuild:
        return False
    expected = {**metadata, "schema_version": SCHEMA_VERSION}
    if read_metadata(filename) != expected:
        raise ValueError(f"研究定义变化，请使用新版本或显式重建 {filename}")
    return True


# 【非核心】在目标文件旁完成Parquet写入，再原子替换正式文件。
def write_parquet(table: pl.LazyFrame, filename: Path, metadata: dict) -> None:
    """输入惰性查询、目标文件和定义，调用方须持有输出独占锁。

    写入失败时清理临时文件，已有正式文件继续可用。
    """
    filename.parent.mkdir(parents=True, exist_ok=True)
    definition = {**metadata, "schema_version": SCHEMA_VERSION}
    with tempfile.TemporaryDirectory(prefix=".publishing-", dir=filename.parent) as temporary:
        staging = Path(temporary) / filename.name
        schema = pl.DataFrame(schema=table.collect_schema()).to_arrow().schema.with_metadata(
            {METADATA_KEY: json.dumps(definition, ensure_ascii=False, sort_keys=True).encode()},
        )
        # 每个文件重复保存时间键，用差分编码压缩规则网格，避免单列布局放大磁盘占用。
        with pq.ParquetWriter(
            staging, schema, compression="zstd", compression_level=3,
            use_dictionary=["symbol", "date"], column_encoding={"timestamp_ns": "DELTA_BINARY_PACKED"},
        ) as writer:
            for batch in table.collect_batches(chunk_size=100_000, maintain_order=True, engine="streaming"):
                writer.write_table(batch.to_arrow(), row_group_size=100_000)
        staging.replace(filename)


# 【核心逻辑20】按共同时间键读取选定研究列，并检查输入来自同一数据定义。
def scan_series(folder: Path, columns: list[str], dates: list[str] | None = None,
                *, start_timestamp_ns: int | None = None) -> tuple[pl.LazyFrame, dict]:
    """输入单变量文件目录、列名和可选日期，返回时间对齐的查询与共同定义。

    当前研究仅含BTCUSDT，UTC纳秒时间戳唯一决定样本，日期与资产由首列文件携带。
    每份正式文件均按完整时间网格发布，独立来源通过dataset定义核对。
    调用方须在查询物化完成前持有这些文件的共享锁。
    """
    first = folder / f"{columns[0]}.parquet"
    reference = read_metadata(first)["dataset"]
    first_scan = pl.scan_parquet(first)
    if dates is not None:
        first_scan = first_scan.filter(pl.col("date").is_in(dates))
    if start_timestamp_ns is not None:
        first_scan = first_scan.filter(pl.col('timestamp_ns') >= start_timestamp_ns)
    # 正式单变量文件由同一完整时间网格按序发布，日期筛选保持顺序和唯一性。
    # 显式声明顺序让Polars采用流式归并，逐次1:1检查会触发全量内存关联。
    table = first_scan.select(*KEY_COLUMNS, columns[0]).set_sorted("timestamp_ns")
    for column in columns[1:]:
        filename = folder / f"{column}.parquet"
        if read_metadata(filename)["dataset"] != reference:
            raise ValueError(f"研究列的输入区间或定义不一致 {filename}")
        values = pl.scan_parquet(filename)
        if dates is not None:
            values = values.filter(pl.col("date").is_in(dates))
        if start_timestamp_ns is not None:
            values = values.filter(pl.col('timestamp_ns') >= start_timestamp_ns)
        values = values.select("timestamp_ns", column).set_sorted("timestamp_ns")
        table = table.join(values, on="timestamp_ns", how="left", maintain_order="left")
    return table, reference


# 【非核心】逐块追加一组单变量文件，整段完成后分别原子发布。
def write_series_frames(frames, folder: Path, columns: list[str], metadata: dict) -> None:
    """输入按时间排序的有限内存表块，追加Parquet行组并清理未完成临时文件。"""
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".series-", dir=folder) as temporary, ExitStack() as streams, ThreadPoolExecutor(max_workers=5) as compression:
        staging = Path(temporary)
        writers = {}
        for frame in frames:
            pending = []
            for column in columns:
                table = frame.select(*KEY_COLUMNS, column).to_arrow()
                if column not in writers:
                    definition = {**metadata[column], "schema_version": SCHEMA_VERSION}
                    schema = table.schema.with_metadata({METADATA_KEY: json.dumps(definition, sort_keys=True).encode()})
                    writer = pq.ParquetWriter(
                        staging / f"{column}.parquet", schema, compression="zstd", compression_level=3,
                        use_dictionary=["symbol", "date"], column_encoding={"timestamp_ns": "DELTA_BINARY_PACKED"},
                    )
                    writers[column] = streams.enter_context(writer)
                pending.append(compression.submit(writers[column].write_table, table, row_group_size=100_000))
            # 每列拥有独立writer，同一帧完成后再接下一帧，保持各文件行组顺序。
            for completed in pending:
                completed.result()
        streams.close()
        for column in columns:
            (staging / f"{column}.parquet").replace(folder / f"{column}.parquet")


# 【非核心】以原子替换发布小型JSON状态，避免中断后留下半份缓存定义。
def write_json_atomic(filename: Path, content: dict) -> None:
    """调用方负责需要的写锁，临时文件与正式文件位于同一文件系统。"""
    filename.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".json-", dir=filename.parent) as temporary:
        staging = Path(temporary) / filename.name
        staging.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n")
        staging.replace(filename)
