import numpy as np
import os
import sys

# 路径设置应该在导入项目模块之前
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 现在可以安全地导入项目模块
from utils.logger import getGlobalLogger  # noqa: E402
from utils.directory_manager import ensure_directory  # noqa: E402
from utils.config import config  # noqa: E402
# 注意：示例所需的 AES128 导入放在 __main__ 中，避免 E402 警告

logger = getGlobalLogger()


# [ADD] 新增: 解析目标索引的辅助函数，支持字符串范围如 "1-16"
def parseTargetIndices(target_index, num_bits):
    """
    解析目标比特位索引，统一为标准格式（int 或 np.ndarray）。

    支持如下输入类型：None、int、list/tuple/np.ndarray、str（范围）。
    字符串范围格式示例："1-16" 表示闭区间 [1, 16]（基于 0 的比特索引）。

    Args:
        target_index: 目标索引，支持 None、int、list/tuple/np.ndarray、str("a-b")。
        num_bits (int): 比特总数，用于边界检查。

    Returns:
        Union[None, int, np.ndarray]:
            - None: 不进行提取（返回完整密文）
            - int: 单个比特位索引
            - np.ndarray: 多个比特位索引数组

    Raises:
        ValueError: 当 target_index 类型不支持或字符串格式非法时抛出。
        IndexError: 当索引超出范围时抛出。

    Example:
        >>> parseTargetIndices("1-3", 8)
        array([1, 2, 3])
        >>> parseTargetIndices([0, 7], 8)
        array([0, 7])
    """
    # [ADD] 支持 None：表示不提取（直接返回完整密文）
    if target_index is None:
        return None

    # [ADD] 支持 int：单索引
    if isinstance(target_index, int):
        if target_index < 0 or target_index >= num_bits:
            raise IndexError(f"比特位索引{target_index}超出范围[0, {num_bits-1}]")
        return int(target_index)

    # [ADD] 支持 list/tuple/ndarray：多索引集合
    if isinstance(target_index, (list, tuple, np.ndarray)):
        target_indices = np.array(target_index, dtype=int)
        if target_indices.ndim != 1:
            raise ValueError("目标索引数组必须为一维")
        if np.any(target_indices < 0) or np.any(target_indices >= num_bits):
            invalid_indices = target_indices[(target_indices < 0) | (target_indices >= num_bits)]
            raise IndexError(f"比特位索引{invalid_indices}超出范围[0, {num_bits-1}]")
        return target_indices

    # [ADD] 支持字符串范围：例如 "1-16"（闭区间，基于 0 的索引）
    if isinstance(target_index, str):
        rng = target_index.strip()
        # 仅支持简单的 a-b 格式；更复杂格式可后续扩展
        if "-" not in rng:
            raise ValueError("字符串索引仅支持例如 '1-16' 的范围格式")
        start_str, end_str = rng.split("-", 1)
        try:
            start = int(start_str.strip())
            end = int(end_str.strip())
        except ValueError:
            raise ValueError("范围字符串必须是整数，例如 '1-16'")
        if start > end:
            raise ValueError("范围起始值不能大于结束值")
        if start < 0 or end >= num_bits:
            raise IndexError(f"范围[{start}, {end}]超出比特长度范围[0, {num_bits-1}]")
        return np.arange(start, end + 1, dtype=int)

    # [ADD] 其他类型不支持
    raise ValueError("target_index必须是None、int、list、tuple、np.ndarray或形如'1-16'的字符串")


def extractTargetBits(cipher_texts, target_index):
    """
    从密文中提取指定的目标比特位作为标签

    支持提取单个或多个比特位，返回相应的标签数组。当target_index为None时，返回完整的密文数组

    Args:
        cipher_texts (np.ndarray): 密文数组，形状为 (样本数, 比特位数)
        target_index: 目标比特位索引，支持以下格式：
            - None: 不进行提取，返回完整密文数组
            - int: 单个比特位索引，返回一维标签数组
            - list/tuple/np.ndarray: 多个比特位索引，返回二维标签数组

    Returns:
        np.ndarray: 提取的标签数组
            - None: 返回完整密文，形状为 (样本数, 比特位数)
            - 单个比特位: 形状为 (样本数,)
            - 多个比特位: 形状为 (样本数, 选定比特数)

    Raises:
        ValueError: 当target_index类型不支持时抛出异常
        IndexError: 当索引超出密文比特位范围时抛出异常

    Example:
        >>> cipher_texts = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
        >>> # 不进行提取，返回完整密文
        >>> labels = extractTargetBits(cipher_texts, None)
        >>> print(labels)  # [[1, 0, 1, 0], [0, 1, 0, 1]]
        >>> # 提取单个比特位
        >>> labels = extractTargetBits(cipher_texts, 0)
        >>> print(labels)  # [1, 0]
        >>> # 提取多个比特位（列表）
        >>> labels = extractTargetBits(cipher_texts, [0, 2])
        >>> print(labels)  # [[1, 1], [0, 0]]
        >>> # 提取范围（字符串）：闭区间 [1, 3]
        >>> labels = extractTargetBits(cipher_texts, "1-3")
        >>> print(labels)  # [[0, 1, 0], [1, 0, 1]]
    """
    # 验证输入参数
    if not isinstance(cipher_texts, np.ndarray):
        logger.error("cipher_texts必须是numpy数组")
        raise ValueError("cipher_texts必须是numpy数组")

    if len(cipher_texts.shape) != 2:
        raise ValueError("cipher_texts必须是二维数组，形状为(样本数, 比特位数)")

    _, num_bits = cipher_texts.shape

    # [MOD] 统一通过解析函数处理 target_index，支持字符串范围
    resolved = parseTargetIndices(target_index, num_bits)

    # 不进行提取，返回完整密文
    if resolved is None:
        return cipher_texts

    # 单个比特位索引
    if isinstance(resolved, int):
        return cipher_texts[:, resolved]

    # 多个比特位索引
    return cipher_texts[:, resolved]


def _save_batch_data(batch_data, batch_labels, batch_num, save_dir, data_type):
    """
    保存单个批次的数据到npy文件
    
    Args:
        batch_data: 批次数据（明文或密文）
        batch_labels: 批次标签
        batch_num: 批次编号
        save_dir: 保存目录
        data_type: 数据类型（'plain' 或 'cipher'）
    """
    try:
        batch_dir = os.path.join(save_dir, "batches")
        ensure_directory(batch_dir)
        
        data_filename = f"batch_{batch_num}_{data_type}_texts.npy"
        labels_filename = f"batch_{batch_num}_labels.npy"
        
        np.save(os.path.join(batch_dir, data_filename), batch_data)
        np.save(os.path.join(batch_dir, labels_filename), batch_labels)
        
        logger.info(
            f"批次 {batch_num} 数据已保存: {data_filename}, {labels_filename}"
        )
        
    except Exception as e:
        logger.error(f"保存批次 {batch_num} 数据时出错: {e}")
        raise


def _merge_batch_files(save_dir, total_batches, data_type="plain"):
    """
    合并所有批次文件为最终的npy文件
    
    Args:
        save_dir: 保存目录
        total_batches: 总批次数
        data_type: 数据类型（'plain' 或 'cipher'）
    
    Returns:
        tuple: (合并后的数据, 合并后的标签)
    """
    try:
        batch_dir = os.path.join(save_dir, "batches")
        all_data = []
        all_labels = []
        
        logger.info(f"开始合并 {total_batches} 个批次文件...")
        
        for batch_num in range(1, total_batches + 1):
            data_filename = f"batch_{batch_num}_{data_type}_texts.npy"
            labels_filename = f"batch_{batch_num}_labels.npy"
            
            data_path = os.path.join(batch_dir, data_filename)
            labels_path = os.path.join(batch_dir, labels_filename)
            
            if (not os.path.exists(data_path) or
                    not os.path.exists(labels_path)):
                raise FileNotFoundError(f"批次 {batch_num} 文件不存在")
            
            batch_data = np.load(data_path)
            batch_labels = np.load(labels_path)
            
            all_data.append(batch_data)
            all_labels.append(batch_labels)
            
            logger.info(f"已加载批次 {batch_num}/{total_batches}")
        
        # 合并所有批次
        merged_data = np.concatenate(all_data, axis=0)
        merged_labels = np.concatenate(all_labels, axis=0)
        
        # 保存最终文件
        final_data_path = os.path.join(save_dir, f"{data_type}_texts.npy")
        final_labels_path = os.path.join(save_dir, "cipher_texts.npy")
        
        np.save(final_data_path, merged_data)
        np.save(final_labels_path, merged_labels)
        
        logger.info(f"合并完成: {final_data_path}, {final_labels_path}")
        logger.info(
            f"最终数据形状: {merged_data.shape}, "
            f"标签形状: {merged_labels.shape}"
        )
        
        return merged_data, merged_labels
        
    except Exception as e:
        logger.error(f"合并批次文件时出错: {e}")
        raise


def _cleanup_batch_files(save_dir, total_batches, data_type="plain"):
    """
    清理中间批次文件
    
    Args:
        save_dir: 保存目录
        total_batches: 总批次数
        data_type: 数据类型
    """
    try:
        batch_dir = os.path.join(save_dir, "batches")
        
        if not os.path.exists(batch_dir):
            logger.warning("批次目录不存在，无需清理")
            return
        
        deleted_count = 0
        for batch_num in range(1, total_batches + 1):
            data_filename = f"batch_{batch_num}_{data_type}_texts.npy"
            labels_filename = f"batch_{batch_num}_labels.npy"
            
            data_path = os.path.join(batch_dir, data_filename)
            labels_path = os.path.join(batch_dir, labels_filename)
            
            for file_path in [data_path, labels_path]:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_count += 1
        
        # 删除空的批次目录
        try:
            os.rmdir(batch_dir)
            logger.info(f"已删除批次目录: {batch_dir}")
        except OSError:
            logger.warning(f"批次目录不为空，保留: {batch_dir}")
        
        logger.info(f"清理完成，删除了 {deleted_count} 个批次文件")
        
    except Exception as e:
        logger.error(f"清理批次文件时出错: {e}")
        raise


def _get_resume_info(save_dir, data_type="plain"):
    """
    获取断点续传信息
    
    Args:
        save_dir: 保存目录
        data_type: 数据类型
    
    Returns:
        int: 已完成的批次数
    """
    try:
        batch_dir = os.path.join(save_dir, "batches")
        
        if not os.path.exists(batch_dir):
            return 0
        
        completed_batches = 0
        batch_num = 1
        
        while True:
            data_filename = f"batch_{batch_num}_{data_type}_texts.npy"
            labels_filename = f"batch_{batch_num}_labels.npy"
            
            data_path = os.path.join(batch_dir, data_filename)
            labels_path = os.path.join(batch_dir, labels_filename)
            
            if os.path.exists(data_path) and os.path.exists(labels_path):
                completed_batches = batch_num
                batch_num += 1
            else:
                break
        
        if completed_batches > 0:
            logger.info(f"发现已完成的批次: {completed_batches}")
        
        return completed_batches
        
    except Exception as e:
        logger.error(f"获取断点续传信息时出错: {e}")
        return 0


def generate_dataset(
    cipher,
    num_keys: int,
    total_data: int,
    save_dir: str,
    target_index=0,
    zero_range=None,
    shuffle=True,
    batch_size=1000,
    resume=True
):
    """
    生成明文-密文对数据集（支持分批次生成和断点续传）

    从密文中提取指定的单个或多个比特位作为标签，用于机器学习训练。
    可选支持在提取前对密文指定区间进行置零（屏蔽）。

    Args:
        cipher: 密码实例，用于加密操作
        num_keys (int): 生成的密钥数量
        total_data (int): 总样本数量，按 num_keys 平均分配，余数由最后一个密钥生成
        save_dir (str): 数据集保存目录路径
        target_index: 目标比特位索引，支持以下格式：
            - int: 单个比特位索引 (例如: 0)
            - list: 多个比特位索引列表 (例如: [0, 1, 2])
            - tuple: 多个比特位索引元组 (例如: (0, 1, 2))
            - np.ndarray: 多个比特位索引数组
        zero_range (tuple|list|None): 可选。密文置零区间 (start, end)，闭区间，按位索引。
            例如：16比特密文置零 8-15 -> zero_range=(8, 15)。默认 None 表示不置零。
        shuffle (bool): 是否打乱数据顺序，默认为True
        batch_size (int): 每个批次的样本数量，默认1000
        resume (bool): 是否启用断点续传，默认True

    Returns:
        None: 数据集直接保存到指定目录

    Raises:
        ValueError: 当target_index类型不支持时抛出异常
        FileNotFoundError: 当断点续传时找不到必要文件时抛出异常
        IOError: 当文件读写出错时抛出异常

    Example:
        >>> # 分批次生成数据，支持断点续传
        >>> generate_dataset(cipher, 10, 5000, "./data", target_index=0,
        ...                  batch_size=500, resume=True)
        >>> # 使用字符串范围选择标签（例如选择比特位 1 到 16）
        >>> generate_dataset(cipher, 10, 5000, "./data", target_index="1-16",
        ...                  batch_size=500, resume=True)
    """
    try:
        # 使用目录管理器确保目录存在
        save_dir = ensure_directory(save_dir)
        
        # 总样本数量按密钥平均分配，余数由最后一个密钥生成
        if num_keys <= 0:
            raise ValueError("num_keys 必须为正整数")
        if total_data <= 0:
            raise ValueError("total_data 必须为正整数")

        total_samples = int(total_data)
        per_key = total_samples // num_keys
        remainder = total_samples % num_keys
        total_batches = (total_samples + batch_size - 1) // batch_size
        
        logger.info(
            f"开始生成数据集: 总样本 {total_samples}，密钥数 {num_keys}，"
            f"每密钥分配 {per_key}，最后一个密钥额外 {remainder}"
        )
        logger.info(f"每批次大小: {batch_size}，目标索引: {target_index}")
        
        # 检查断点续传
        completed_batches = 0
        if resume:
            completed_batches = _get_resume_info(save_dir, "plain")
            if completed_batches > 0:
                logger.info(f"启用断点续传，从批次 {completed_batches + 1} 开始")
        
        # 生成数据
        sample_count = 0
        current_batch_plain = []
        current_batch_cipher = []
        current_batch_num = completed_batches + 1
        
        # 跳过已完成的样本
        samples_to_skip = completed_batches * batch_size
        
        # 校验与准备置零参数
        use_zeroing = zero_range is not None
        zero_bounds = None  # (start, end)
        if use_zeroing:
            if not (isinstance(zero_range, (tuple, list)) and len(zero_range) == 2):
                raise ValueError("zero_range 必须是长度为2的 (start, end) 元组或列表")
            s, e = int(zero_range[0]), int(zero_range[1])
            if s > e:
                raise ValueError("zero_range 的 start 不能大于 end")
            zero_bounds = (s, e)
            logger.info(f"启用密文置零: 区间 [{s}, {e}] (闭区间)")

        for key_idx in range(num_keys):
            # [MOD] 使用统一的随机位生成函数替换旧的 generate_key
            # [DEL] 删除旧接口的直接调用以兼容 BaseCipher 的精简接口
            key = cipher.generateRandomBits(cipher.key_size)

            # 该密钥的样本数（最后一个密钥承担余数）
            this_key_samples = (
                per_key
                + (remainder if key_idx == num_keys - 1 else 0)
            )

            for sample_idx in range(this_key_samples):
                # 跳过已完成的样本
                if sample_count < samples_to_skip:
                    sample_count += 1
                    continue
                
                # [MOD] 使用统一的随机位生成函数替换旧的 generate_plaintext
                # [DEL] 删除旧接口的直接调用以兼容 BaseCipher 的精简接口
                plain_text = cipher.generateRandomBits(cipher.block_size)
                
                # 在提取标签前，对明指定区间置零（闭区间）
                if use_zeroing:
                    # 使用明文长度进行边界检查
                    num_bits = int(plain_text.size)
                    if zero_bounds is None:
                        raise RuntimeError("zero_bounds 为 None，无法解包")
                    else:
                        s, e = zero_bounds

                    if s < 0 or e >= num_bits:
                        raise IndexError(
                            f"zero_range 超出明文长度范围 [0, {num_bits-1}]"
                        )
                    plain_text[s:e + 1] = 0
                cipher_text = cipher.encrypt(plain_text, key)
                current_batch_plain.append(plain_text)
                current_batch_cipher.append(cipher_text)
                sample_count += 1
                
                # 当前批次已满或是最后一个样本
                if (len(current_batch_plain) >= batch_size or
                        sample_count >= total_samples):
                    
                    # 转换为numpy数组
                    batch_plain_np = np.array(
                        current_batch_plain,
                        dtype=np.uint8
                    )
                    batch_cipher_np = np.array(
                        current_batch_cipher,
                        dtype=np.uint8
                    )
                    
                    # 提取标签
                    batch_labels = extractTargetBits(
                        batch_cipher_np,
                        target_index
                    )
                    
                    # 保存批次
                    _save_batch_data(
                        batch_plain_np, batch_labels, 
                        current_batch_num, save_dir, "plain"
                    )
                    
                    logger.info(
                        f"批次 {current_batch_num}/{total_batches} 完成 "
                        f"({len(current_batch_plain)} 样本)"
                    )
                    
                    # 重置当前批次
                    current_batch_plain = []
                    current_batch_cipher = []
                    current_batch_num += 1
        
        # 合并所有批次文件
        logger.info("开始合并批次文件...")
        merged_data, merged_labels = _merge_batch_files(
            save_dir, total_batches, "plain"
        )
        
        # 打乱数据顺序（如果需要）
        if shuffle:
            logger.info("打乱数据顺序...")
            indices = np.random.permutation(len(merged_data))
            merged_data = merged_data[indices]
            merged_labels = merged_labels[indices]
            
            # 重新保存打乱后的数据
            np.save(os.path.join(save_dir, "plain_texts.npy"), merged_data)
            np.save(os.path.join(save_dir, "cipher_texts.npy"), merged_labels)
            logger.info("数据已打乱并重新保存")
        
        # 清理中间文件
        logger.info("清理中间批次文件...")
        _cleanup_batch_files(save_dir, total_batches, "plain")
        
        logger.info(
            f"数据集生成完成！共 {len(merged_data)} 个样本，"
            f"标签形状: {merged_labels.shape}，"
            f"{'已打乱' if shuffle else '未打乱'}数据顺序，保存至 {save_dir}"
        )
        
    except KeyboardInterrupt:
        logger.warning("用户中断操作，已保存的批次文件将保留用于断点续传")
        raise
    except Exception as e:
        logger.error(f"生成数据集时出错: {e}")
        # 在出错时不自动清理，保留用于调试
        logger.info("出错时保留中间文件用于调试和断点续传")
        raise


# 示例：生成超参数优化实验的数据集（20个密钥，每个密钥1638个样本，共32760≈2^15）
if __name__ == "__main__":
    # 将示例所需的 AES128 导入放在模块末尾，避免 E402
    from ciphers import AES128  # noqa: E402
    cipher = AES128(rounds=4)
    # 示例1：单个目标比特位
    generate_dataset(
        cipher=cipher,
        num_keys=10,
        total_data=32768,  # 总样本数≈2^15，最后一个密钥承担余数
        save_dir=os.path.join(config.getDataDirectory(), "test_sample"),  # [MOD] 使用新接口获取数据目录
        target_index="0-7",
        shuffle=False,
        batch_size=1000
    )
