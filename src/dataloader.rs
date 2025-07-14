use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::ClusterInfo;
use crate::structs::{BinaryDatasetResult, ForwardIndex, KeywordDocPair, KeywordInfo};
use ahash::{AHashMap, AHashSet};
use anyhow::Result;
use byteorder::{LittleEndian, ReadBytesExt};
use rayon::prelude::*;
pub struct DatasetLoader {
    dataset_path: PathBuf,
}
impl Default for DatasetLoader {
    fn default() -> Self {
        Self {
            dataset_path: Path::new("/tmp/data/enron/processed").to_path_buf(),
        }
    }
}

impl DatasetLoader {
    pub fn new(dataset_path: impl Into<PathBuf>) -> Self {
        Self {
            dataset_path: dataset_path.into(),
        }
    }
}

impl DatasetLoader {
    pub fn load_clusters(&self, min_cluster_size: usize) -> BinaryDatasetResult {
        let cluster_dir = self
            .dataset_path
            .join(format!("cluster_{min_cluster_size}"));

        // 1. 读取元数据
        let metadata_path = cluster_dir.join("metadata.bin");
        let mut metadata = Vec::new();
        File::open(metadata_path)?.read_to_end(&mut metadata)?;
        let mut reader = &metadata[..];

        let _total_pairs = reader.read_u64::<LittleEndian>()?;
        let _num_keywords = reader.read_u64::<LittleEndian>()?;
        let _cluster_size = reader.read_u64::<LittleEndian>()? as usize;

        // 2. 读取关键词列表
        let keywords_path = cluster_dir.join("keywords.bin");
        let mut keywords_file = File::open(keywords_path)?;
        let mut keywords_bytes = Vec::new();
        keywords_file.read_to_end(&mut keywords_bytes)?;
        let mut reader = &keywords_bytes[..];

        let stored_keyword_count = reader.read_u64::<LittleEndian>()?;
        let mut all_keywords = Vec::with_capacity(stored_keyword_count as usize);

        for _ in 0..stored_keyword_count {
            let len = reader.read_u64::<LittleEndian>()?;
            let mut keyword_bytes = vec![0u8; len as usize];
            reader.read_exact(&mut keyword_bytes)?;
            all_keywords.push(String::from_utf8_lossy(&keyword_bytes).into_owned());
        }

        // 3. 重建关键词到ID的映射
        let _keyword_to_id: HashMap<_, _> = all_keywords
            .par_iter()
            .enumerate()
            .map(|(idx, kw)| (kw.clone(), idx as u64))
            .collect();

        // 4. 读取簇信息
        let clusters_path = cluster_dir.join("clusters.bin");
        let mut clusters_file = File::open(clusters_path)?;
        let mut clusters_bytes = Vec::new();
        clusters_file.read_to_end(&mut clusters_bytes)?;
        let mut reader = &clusters_bytes[..];

        let cluster_count = reader.read_u64::<LittleEndian>()?;
        let mut cluster = AHashMap::new();

        for _ in 0..cluster_count {
            let keyword_count = reader.read_u64::<LittleEndian>()?;
            let mut keyword_ids = Vec::with_capacity(keyword_count as usize);

            for _ in 0..keyword_count {
                keyword_ids.push(reader.read_u64::<LittleEndian>()?);
            }

            let min_freq = reader.read_u64::<LittleEndian>()? as usize;
            let max_freq = reader.read_u64::<LittleEndian>()? as usize;
            let avg_freq = reader.read_f64::<LittleEndian>()?;
            let threshold = reader.read_u64::<LittleEndian>()? as usize;

            cluster.insert(
                cluster.len(),
                ClusterInfo {
                    keywords: keyword_ids
                        .into_iter()
                        .map(|id| all_keywords[id as usize].clone())
                        .collect(),
                    min_freq,
                    max_freq,
                    avg_freq,
                    threshold,
                },
            );
        }

        // 5. 读取关键词/文档对数据
        let pairs_path = cluster_dir.join("keyword_doc_pairs.bin");
        let mut pairs_file = File::open(pairs_path)?;
        let mut pairs_bytes = Vec::new();
        pairs_file.read_to_end(&mut pairs_bytes)?;
        let mut reader = &pairs_bytes[..];

        let stored_pairs = reader.read_u64::<LittleEndian>()?;
        let mut keyword_docs = AHashMap::<String, Vec<KeywordDocPair>>::new();

        for _ in 0..stored_pairs {
            let kid = reader.read_u64::<LittleEndian>()?;
            let doc_len = reader.read_u64::<LittleEndian>()?;
            let mut doc_id_bytes = vec![0u8; doc_len as usize];
            reader.read_exact(&mut doc_id_bytes)?;
            let doc_id = String::from_utf8_lossy(&doc_id_bytes).into_owned();
            let access_level = reader.read_u8()?;
            let state = reader.read_u8()?;

            let keyword = all_keywords[kid as usize].clone();
            keyword_docs
                .entry(keyword)
                .or_default()
                .push(KeywordDocPair::new(doc_id, access_level, state));
        }

        Ok((keyword_docs, cluster))
    }

    pub fn convert_single_cluster_to_forward_index(
        &self,
        min_cluster_size_in_this_group: usize,
        cluster_id: usize,
    ) -> Result<ForwardIndex> {
        // 定位到目标cluster目录
        let cluster_dir = self
            .dataset_path
            .join(format!("cluster_{min_cluster_size_in_this_group}"));

        // 1. 验证目录存在性
        if !cluster_dir.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Cluster directory not found: {}", cluster_dir.display()),
            )
            .into());
        }

        // 2. 读取元数据并验证cluster_id一致性
        let metadata_path = cluster_dir.join("metadata.bin");
        let mut metadata = Vec::new();
        File::open(metadata_path)?.read_to_end(&mut metadata)?;
        let mut reader = &metadata[..];

        let _total_pairs = reader.read_u64::<LittleEndian>()?;
        let _num_keywords = reader.read_u64::<LittleEndian>()?;
        let actual_min_cluster_size = reader.read_u64::<LittleEndian>()? as usize;

        if min_cluster_size_in_this_group != actual_min_cluster_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Cluster metadata mismatch: expected {min_cluster_size_in_this_group}, found {actual_min_cluster_size}"),
            )
            .into());
        }

        // 3. 读取关键词列表
        let keywords_path = cluster_dir.join("keywords.bin");
        let mut keywords_bytes = Vec::new();
        File::open(keywords_path)?.read_to_end(&mut keywords_bytes)?;
        let mut reader = &keywords_bytes[..];

        let keyword_count = reader.read_u64::<LittleEndian>()?;
        let mut all_keywords = Vec::with_capacity(keyword_count as usize);

        for _ in 0..keyword_count {
            let len = reader.read_u64::<LittleEndian>()?;
            let mut keyword_bytes = vec![0u8; len as usize];
            reader.read_exact(&mut keyword_bytes)?;
            all_keywords.push(String::from_utf8_lossy(&keyword_bytes).into_owned());
        }

        // 4. 建立关键词到ID的映射
        // let keyword_to_id: HashMap<_, _> = all_keywords
        //     .iter()
        //     .enumerate()
        //     .map(|(idx, kw)| (kw.clone(), idx as u64))
        //     .collect();

        // 5. 读取簇信息
        let clusters_path = cluster_dir.join("clusters.bin");
        let mut clusters_bytes = Vec::new();
        File::open(clusters_path)?.read_to_end(&mut clusters_bytes)?;
        let mut reader = &clusters_bytes[..];

        let cluster_count = reader.read_u64::<LittleEndian>()?;
        if cluster_count == 0 {
            return Err(
                std::io::Error::new(std::io::ErrorKind::InvalidData, "Expected cluster").into(),
            );
        }

        //跳过其他簇
        for _ in 0..cluster_id {
            let keyword_count = reader.read_u64::<LittleEndian>()?;
            for _ in 0..keyword_count {
                reader.read_u64::<LittleEndian>()?;
            }
            reader.read_u64::<LittleEndian>()?;
            reader.read_u64::<LittleEndian>()?;
            reader.read_f64::<LittleEndian>()?;
            reader.read_u64::<LittleEndian>()?;
        }

        // 读取当前簇的关键词列表
        let keyword_count = reader.read_u64::<LittleEndian>()?;
        let mut keyword_ids = Vec::with_capacity(keyword_count as usize);

        for _ in 0..keyword_count {
            keyword_ids.push(reader.read_u64::<LittleEndian>()?);
        }

        let cluster_keywords: Vec<String> = keyword_ids
            .par_iter()
            .map(|id| all_keywords[*id as usize].clone())
            .collect();

        //  跳过其他簇信息字段
        // let _min_freq = reader.read_u64::<LittleEndian>()?;
        // let _max_freq = reader.read_u64::<LittleEndian>()?;
        // let _avg_freq = reader.read_f64::<LittleEndian>()?;
        // let _threshold = reader.read_u64::<LittleEndian>()?;

        // 6. 读取关键词/文档对并构建正向索引
        let pairs_path = cluster_dir.join("keyword_doc_pairs.bin");
        let mut pairs_bytes = Vec::new();
        File::open(pairs_path)?.read_to_end(&mut pairs_bytes)?;
        let mut reader = &pairs_bytes[..];

        let stored_pairs = reader.read_u64::<LittleEndian>()?;
        let mut forward_index = AHashMap::<String, Vec<KeywordInfo>>::new();

        // 创建关键词集合用于快速查找
        let cluster_keyword_set: AHashSet<_> = cluster_keywords.iter().cloned().collect();

        for _ in 0..stored_pairs {
            let kid = reader.read_u64::<LittleEndian>()?;
            let doc_len = reader.read_u64::<LittleEndian>()?;
            let mut doc_id_bytes = vec![0u8; doc_len as usize];
            reader.read_exact(&mut doc_id_bytes)?;
            let doc_id = String::from_utf8_lossy(&doc_id_bytes);
            let access_level = reader.read_u8()?;
            let state = reader.read_u8()?;

            let Some(keyword) = all_keywords.get(kid as usize) else {
                continue;
            };

            // 只处理当前簇的关键词
            if cluster_keyword_set.contains(keyword) {
                forward_index
                    .entry(doc_id.to_string())
                    .or_default()
                    .push(KeywordInfo {
                        keyword: keyword.clone(),
                        access_level,
                        state,
                    });
            }
        }

        Ok(forward_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{LittleEndian, WriteBytesExt};
    use std::fs::File;
    use std::io::Write;
    use tempfile::{TempDir, tempdir};

    // 测试固定数据
    const CLUSTER_ID: usize = 42;
    const CLUSTER_SIZE: usize = 100;

    // 测试环境构建器
    struct TestEnv {
        dir: TempDir,
        _dataset_path: PathBuf,
    }

    impl TestEnv {
        fn new() -> Self {
            let dir = tempdir().unwrap();
            let dataset_path = dir.path().to_path_buf();

            // 创建测试所需的基础目录结构
            std::fs::create_dir_all(dataset_path.join("cluster_100")).unwrap();

            Self {
                dir,
                _dataset_path: dataset_path,
            }
        }

        fn create_metadata(&self, size: usize) {
            let mut f = File::create(self.dir.path().join("cluster_100/metadata.bin")).unwrap();
            f.write_u64::<LittleEndian>(1000).unwrap(); // total_pairs
            f.write_u64::<LittleEndian>(10).unwrap(); // num_keywords
            f.write_u64::<LittleEndian>(size as u64).unwrap(); // cluster_size
        }

        fn create_keywords(&self) {
            let mut f = File::create(self.dir.path().join("cluster_100/keywords.bin")).unwrap();
            f.write_u64::<LittleEndian>(3).unwrap(); // keyword_count

            // 写入三个关键词："apple", "banana", "cherry"
            write_string(&mut f, "apple");
            write_string(&mut f, "banana");
            write_string(&mut f, "cherry");
        }

        fn create_clusters(&self, total_clusters: u64) {
            let mut f = File::create(self.dir.path().join("cluster_100/clusters.bin")).unwrap();
            f.write_u64::<LittleEndian>(total_clusters).unwrap();

            for _ in 0..total_clusters {
                f.write_u64::<LittleEndian>(2).unwrap(); // keyword_count

                // 写入两个关键词ID
                f.write_u64::<LittleEndian>(0).unwrap();
                f.write_u64::<LittleEndian>(1).unwrap();

                // 写入簇统计信息
                f.write_u64::<LittleEndian>(10).unwrap(); // min_freq
                f.write_u64::<LittleEndian>(100).unwrap(); // max_freq
                f.write_f64::<LittleEndian>(50.0).unwrap(); // avg_freq
                f.write_u64::<LittleEndian>(20).unwrap(); // threshold

                // 如果是目标簇，保存其ID
                // if i as usize == cluster_id {
                //     // 重写文件指针到簇ID位置
                //     let _ = f.seek(std::io::SeekFrom::Current(-24));
                //     f.write_u64::<LittleEndian>(cluster_id as u64).unwrap();
                // }
            }
        }

        fn create_pairs(&self) {
            let mut f =
                File::create(self.dir.path().join("cluster_100/keyword_doc_pairs.bin")).unwrap();
            f.write_u64::<LittleEndian>(3).unwrap(); // stored_pairs

            // 写入三组测试数据
            write_pair(&mut f, 0, "doc1", 1, 0); // 属于目标簇
            write_pair(&mut f, 1, "doc2", 2, 1); // 属于目标簇
            write_pair(&mut f, 2, "doc3", 3, 0); // 不属于目标簇
        }
    }

    // 辅助函数：写入带长度前缀的字符串
    fn write_string(f: &mut File, s: &str) {
        f.write_u64::<LittleEndian>(s.len() as u64).unwrap();
        f.write_all(s.as_bytes()).unwrap();
    }

    // 辅助函数：写入单个关键词/文档对
    fn write_pair(f: &mut File, kid: u64, doc_id: &str, access_level: u8, state: u8) {
        f.write_u64::<LittleEndian>(kid).unwrap();
        f.write_u64::<LittleEndian>(doc_id.len() as u64).unwrap();
        f.write_all(doc_id.as_bytes()).unwrap();
        f.write_u8(access_level).unwrap();
        f.write_u8(state).unwrap();
    }

    #[test]
    fn test_normal_case() {
        let env = TestEnv::new();

        // 准备测试数据
        env.create_metadata(CLUSTER_SIZE);
        env.create_keywords();
        env.create_clusters(50);
        env.create_pairs();

        // 构造测试对象
        let indexer = DatasetLoader {
            dataset_path: env._dataset_path,
        };

        // 执行测试
        let result = indexer.convert_single_cluster_to_forward_index(CLUSTER_SIZE, CLUSTER_ID);

        // 验证结果
        assert!(result.is_ok());
        let index = result.unwrap();

        // 验证文档数量
        assert_eq!(index.len(), 2);

        // 验证文档1的内容
        assert!(index.contains_key("1"));
        assert_eq!(index["1"].len(), 1);
        assert_eq!(index["1"][0].access_level, 1);
        assert_eq!(index["1"][0].state, 0);
        assert_eq!(index["1"][0].keyword, "apple");

        // 验证文档2的内容
        assert!(index.contains_key("2"));
        assert_eq!(index["2"].len(), 1);
        assert_eq!(index["2"][0].access_level, 2);
        assert_eq!(index["2"][0].state, 1);
        assert_eq!(index["2"][0].keyword, "banana");

        // 验证文档3是否被过滤
        assert!(!index.contains_key("3"));
    }

    #[test]
    fn test_directory_not_found() {
        let env = TestEnv::new();

        // 不创建任何文件，直接测试
        let indexer = DatasetLoader {
            dataset_path: env._dataset_path,
        };

        let result = indexer.convert_single_cluster_to_forward_index(CLUSTER_SIZE, CLUSTER_ID);

        assert!(result.is_err());
        assert_eq!(
            result
                .unwrap_err()
                .downcast_ref::<std::io::Error>()
                .unwrap()
                .kind(),
            std::io::ErrorKind::NotFound
        );
    }

    #[test]
    fn test_cluster_size_mismatch() {
        let env = TestEnv::new();

        // 创建元数据但使用不同的簇大小
        env.create_metadata(CLUSTER_SIZE + 1);

        let indexer = DatasetLoader {
            dataset_path: env._dataset_path,
        };

        let result = indexer.convert_single_cluster_to_forward_index(CLUSTER_SIZE, CLUSTER_ID);

        assert!(result.is_err());
        assert_eq!(
            result
                .unwrap_err()
                .downcast_ref::<std::io::Error>()
                .unwrap()
                .kind(),
            std::io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn test_zero_clusters() {
        let env = TestEnv::new();

        // 创建元数据
        env.create_metadata(CLUSTER_SIZE);

        // 创建空的keywords.bin
        File::create(env.dir.path().join("cluster_100/keywords.bin")).unwrap();

        let indexer = DatasetLoader {
            dataset_path: env._dataset_path,
        };

        let result = indexer.convert_single_cluster_to_forward_index(CLUSTER_SIZE, CLUSTER_ID);

        assert!(result.is_err());
        assert_eq!(
            result
                .unwrap_err()
                .downcast_ref::<std::io::Error>()
                .unwrap()
                .kind(),
            std::io::ErrorKind::UnexpectedEof
        );
    }
}
