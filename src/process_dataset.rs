use crate::cluster::ClusterInfo;
use crate::stemmer::get_stem;
use crate::structs::BinaryDatasetResult;
use crate::utils::generate_file_id_str;
use ahash::RandomState;
use anyhow::Result;
use anyhow::anyhow;
use byteorder::{LittleEndian, WriteBytesExt};
use dashmap::DashMap;
use flate2::read::GzDecoder;

use indicatif::{ParallelProgressIterator, ProgressStyle};
use log::info;
use mailparse::parse_mail;
use num_format::{Locale, ToFormattedString};
use rand::prelude::*;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::LazyLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Instant;
use tar::Archive;
use walkdir::WalkDir;

use ahash::AHashMap;

use crate::structs::KeywordDocPair;

#[derive(Debug, Clone)]
pub struct EnronDataProcessor {
    // stemmer: impl Fn(&str) -> Result<String>,
    stopwords: HashSet<String, RandomState>,
    access_levels: Vec<u8>,
    states: Vec<u8>,
    target_keywords: usize,
    cluster_sizes: Vec<usize>,
}

impl Default for EnronDataProcessor {
    fn default() -> Self {
        // 添加停用词（简化版）
        let default_stopwords = [
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
            "this",
            "that",
            "these",
            "those",
            "am",
            "is",
            "are",
            "was",
            "were",
            "has",
            "have",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "shall",
            "should",
            "can",
            "could",
            "may",
            "might",
            // 邮件相关停用词
            "email",
            "mail",
            "from",
            "to",
            "cc",
            "bcc",
            "subject",
            "date",
            "time",
            "day",
            "week",
            "month",
            "year",
            "send",
            "sent",
            "forward",
            "forwarded",
            "reply",
            "please",
            "thank",
            "thanks",
            "regard",
            "regards",
            "sincerely",
            "dear",
            "hi",
            "hello",
            "hey",
        ];

        let stopwords: HashSet<String, RandomState> = default_stopwords
            .par_iter()
            .map(|stopword| get_stem(stopword))
            .collect();

        Self {
            // stemmer,
            stopwords,
            access_levels: vec![1, 2, 3],
            states: (1..=10).collect(),
            target_keywords: 10,
            cluster_sizes: vec![3],
        }
    }
}

impl EnronDataProcessor {
    pub fn process_dataset(&self, input_file: &Path, output_dir: &Path) -> Result<()> {
        let start = Instant::now();
        println!("\nProcessing Enron dataset...");
        // remove extension from input file
        let input_file_name = input_file
            .file_stem()
            .ok_or(anyhow!("Invalid input file name"))?;
        let input_files_dir = Path::new(input_file_name);
        extract_tar_gz(input_file, input_files_dir)?;
        let (keyword_counts, email_count, _doc_lengths) = self.first_pass(input_files_dir)?;
        info!("\nKeyword Statistics:");
        info!(
            "Total unique keywords before filtering: {}",
            keyword_counts.len()
        );
        println!("Total emails: {email_count}");

        // 选择最常用的关键词
        let top_keywords_and_counts = self.select_top_keywords(&keyword_counts);
        info!("Selected top {} keywords", self.target_keywords);

        let top_keywords_and_counts: AHashMap<_, _> = top_keywords_and_counts.collect();

        // 保存top 10关键词
        self.save_top_keywords(output_dir, top_keywords_and_counts.keys().cloned())?;

        // 第二遍：构建数据集
        let keyword_file_pairs =
            self.second_pass(input_files_dir, &top_keywords_and_counts, output_dir)?;

        // 生成数据集
        for &cluster_size in &self.cluster_sizes {
            info!("\nGenerating dataset for cluster size {cluster_size}...");
            let (keyword_docs, clusters) = self
                .generate_dataset(&keyword_file_pairs, cluster_size)
                .map_err(|e| anyhow!("{e:?}"))?;
            std::thread::scope(|s| {
                let handle1 = s.spawn(|| -> Result<()> {
                    self.save_binary_dataset(&keyword_docs, &clusters, output_dir, cluster_size)
                });
                let handle2 = s.spawn(|| -> Result<()> {
                    ClusterInfo::save_cluster_info(
                        &keyword_docs,
                        &clusters,
                        cluster_size,
                        output_dir,
                    )
                });
                let status1 = handle1.join();
                let status2 = handle2.join();
                // only two threads end successfully and their return values are Ok,
                // then return Ok
                if let (Ok(status1), Ok(status2)) = (status1, status2) {
                    if status1.is_ok() && status2.is_ok() {
                        Ok(())
                    } else {
                        Err(anyhow!("{status1:?} {status2:?}"))
                    }
                } else {
                    Err(anyhow!("Error in one of the threads"))
                }
            })?;
        }

        // 生成统计报告
        self.generate_statistics_report(&keyword_counts, &keyword_file_pairs, None)?;

        let duration = start.elapsed().as_secs_f32();
        info!("\nProcessing completed in {duration:.2} seconds");
        Ok(())
    }

    fn first_pass(
        &self,
        input_file: &Path,
    ) -> Result<(AHashMap<String, usize>, usize, Vec<usize>)> {
        let directories = WalkDir::new(input_file);
        let keyword_counts: DashMap<String, usize, RandomState> = DashMap::default();
        let email_count = AtomicUsize::new(0);
        let doc_lengths = boxcar::Vec::new();
        let entries: Vec<_> = directories.into_iter().filter_map(|e| e.ok()).collect();

        entries
            .into_par_iter()
            .progress_with_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] [{wide_bar}] {pos}/{len} ({eta})")?,
            )
            .try_for_each(|entry: walkdir::DirEntry| -> anyhow::Result<()> {
                if !entry.file_type().is_file() {
                    info!("Skipping non-regular file: {:?}", entry.file_name());
                } else if !entry.file_name().to_string_lossy().ends_with('.') {
                } else {
                    let mut content = Vec::new();
                    let file_path = entry.path();
                    let mut file = File::open(file_path)?;
                    file.read_to_end(&mut content)?;
                    match self.process_email(&content) {
                        Ok((keywords, len)) => {
                            doc_lengths.push(len);
                            for word in keywords {
                                *keyword_counts.entry(word).or_insert(0) += 1;
                            }
                            email_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        Err(e) => println!(
                            "Error processing {}: with error {:?}",
                            file_path.display(),
                            e
                        ),
                    }
                }

                Ok(())
            })?;
        let email_count = email_count.load(std::sync::atomic::Ordering::Relaxed);
        info!("\nTotal emails processed: {email_count}");
        Ok((
            keyword_counts.into_iter().collect(),
            email_count,
            doc_lengths.into_iter().collect(),
        ))
    }

    fn process_email(&self, content: &[u8]) -> Result<(Vec<String>, usize)> {
        // 直接内联处理邮件内容
        let parsed = parse_mail(content)?;
        let mut results = Vec::new();

        // 定义递归
        fn walk_part(
            part: &mailparse::ParsedMail<'_>,
            results: &mut Vec<Vec<String>>,
            extractor: &impl Fn(&str) -> Vec<String>,
        ) -> Result<()> {
            if part.subparts.is_empty() {
                // 获取内容类型和解码后的body
                let body = part.get_body().unwrap_or_default();
                let keywords = extractor(&body);
                // 添加到结果列表中
                results.push(keywords);
                Ok(())
            } else {
                // 处理子部分
                for subpart in &part.subparts {
                    walk_part(subpart, results, extractor)?;
                }
                Ok(())
            }
        }

        // 执行递归处理
        walk_part(&parsed, &mut results, &|text| self.extract_keywords(text))?;

        // 计算总长度
        let total_len = results.par_iter().map(|v| v.len()).sum();

        Ok((results.into_iter().flatten().collect(), total_len))
    }

    fn extract_keywords(&self, text: &str) -> Vec<String> {
        // lazy_static::lazy_static! {
        //     static ref RE: Regex = Regex::new(r"\b[a-zA-Z]{3,}\b").unwrap();
        //     static ref STEMMER: Stemmer = Stemmer::create(Algorithm::English);
        // };

        static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b[a-zA-Z]{3,}\b").unwrap());
        RE.find_iter(text)
            .par_bridge()
            .map(|m| m.as_str())
            .filter(|word| !self.stopwords.contains(*word))
            .map(get_stem)
            .collect()
    }

    fn select_top_keywords(
        &self,
        keyword_counts: &AHashMap<String, usize>,
    ) -> impl Iterator<Item = (String, usize)> {
        // 避免克隆直接处理迭代器
        let mut keywords = keyword_counts
            .par_iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect::<Vec<_>>();

        // 使用不稳定排序提升性能
        keywords.par_sort_unstable_by(|a, b| b.1.cmp(&a.1));

        // 提前分配容量避免多次分配
        // let mut result = HashMap::with_capacity(self.target_keywords);
        keywords.into_iter().take(self.target_keywords)
    }

    fn save_top_keywords(
        &self,
        output_dir: &Path,
        top_keywords: impl Iterator<Item = String>,
    ) -> Result<()> {
        fs::create_dir_all(output_dir)?;
        let output_path = Path::new(output_dir).join("top_keywords.txt");
        let mut file = File::create(output_path)?;
        // 缓存要写入的数据
        let mut data = String::new();

        for keyword in top_keywords {
            data = format!("{data} {keyword}");
        }
        // for key in top_10_keywords.keys() {
        //     write!(file, "{key} ")?;
        // }
        file.write_all(data.as_bytes())?;

        Ok(())
    }

    fn second_pass(
        &self,
        input_dir: &Path,
        top_keywords: &AHashMap<String, usize>,
        output_dir: &Path,
    ) -> Result<AHashMap<String, HashSet<String, RandomState>>> {
        // FIXME: change to `AHashSet` when `AHashSet` is useable in this content
        let directories = WalkDir::new(input_dir);
        // let reader = BufReader::new(file);
        let keyword_file_pairs: DashMap<String, HashSet<String,RandomState>, RandomState> =
            // DashMap::with_capacity::<RandomState>(self.target_keywords);
            DashMap::with_capacity_and_hasher(self.target_keywords,RandomState::new());

        let entries: Vec<_> = directories.into_iter().filter_map(|e| e.ok()).collect();
        let file_id = AtomicUsize::new(1);
        entries
            .into_par_iter()
            .progress_with_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] [{wide_bar}] {pos}/{len} ({eta})")?,
            )
            .try_for_each(|entry| -> anyhow::Result<()> {
                if !entry.file_type().is_file()
                    || !entry.file_name().to_string_lossy().ends_with('.')
                {
                } else {
                    let mut content = Vec::new();
                    let file_path = entry.path();
                    let mut file = File::open(file_path)?;
                    file.read_to_end(&mut content)?;

                    match self.process_email(&content) {
                        Ok(keywords) => {
                            // 只保留选定的关键词
                            let mut filtered_keywords = keywords
                                .0
                                .into_par_iter()
                                .filter(|k| top_keywords.contains_key(k))
                                // de-duplicate
                                .collect::<Vec<_>>();

                            filtered_keywords.sort_unstable();
                            filtered_keywords.dedup();
                            if !filtered_keywords.is_empty() {
                                let doc_id =
                                    generate_file_id_str(file_id.fetch_add(1, Ordering::SeqCst));
                                let output_path = Path::new(output_dir).join(&doc_id);
                                let mut file = File::create(output_path)?;

                                let mut file_content = String::new();
                                // 写入关键词
                                for keyword in &filtered_keywords[..filtered_keywords.len() - 1] {
                                    file_content.push_str(&format!("{keyword},"));
                                }

                                if let Some(last) = filtered_keywords.last() {
                                    file_content.push_str(last);
                                }

                                file.write_all(file_content.as_bytes())?;

                                // 更新keyword_file_pairs
                                for keyword in filtered_keywords {
                                    let mut entry = keyword_file_pairs.entry(keyword).or_default();
                                    entry.insert(doc_id.clone());
                                }
                            }
                        }
                        Err(e) => println!("Error processing {}: {:?}", file_path.display(), e),
                    }
                }

                Ok(())
            })?;
        Ok(keyword_file_pairs.into_iter().collect())
    }

    fn generate_dataset(
        &self,
        keyword_file_pairs: &AHashMap<String, HashSet<String, RandomState>>,
        min_cluster_size: usize,
    ) -> BinaryDatasetResult {
        // 计算关键词实际频率
        let keyword_freqs: HashMap<_, _> = keyword_file_pairs
            .par_iter()
            .map(|(k, v)| (k.clone(), v.len()))
            .collect();

        // 计算总的关键词文档对数量
        let total_pairs = keyword_file_pairs
            .values()
            .par_bridge()
            .map(|docs| docs.len())
            .sum::<usize>();

        info!("\nTotal keyword-document pairs: {total_pairs}");

        // 按频率降序排序关键词
        let mut sorted_keywords: Vec<_> = keyword_freqs
            .par_iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        sorted_keywords.par_sort_by(|a, b| b.1.cmp(&a.1));

        // 初始化簇
        let mut clusters = Vec::new();
        let mut current_cluster = Vec::new();

        for (keyword, freq) in sorted_keywords {
            // 如果当前簇为空，初始化新簇
            if current_cluster.is_empty() {
                current_cluster.push(keyword);
                continue;
            }

            // 计算与当前簇第一个词（最大频率）的比值
            let first_freq = keyword_freqs[&current_cluster[0]];
            if first_freq == 0 {
                current_cluster.push(keyword);
                continue;
            }
            let freq_ratio = freq as f64 / first_freq as f64;

            // 如果频率比过小且簇已经达到最小大小，开始新的簇
            if freq_ratio < 0.7 && current_cluster.len() >= min_cluster_size {
                let cluster_info = create_cluster_info(&mut current_cluster, &keyword_freqs, false);
                clusters.push(cluster_info);

                current_cluster = vec![keyword];
            } else {
                current_cluster.push(keyword);
            }
        }

        // 处理最后一个簇
        if !current_cluster.is_empty() && current_cluster.len() >= min_cluster_size {
            let cluster_info = create_cluster_info(&mut current_cluster, &keyword_freqs, true);
            clusters.push(cluster_info);
        }

        // 打印簇的分布情况
        println!("\nCluster distribution for size {min_cluster_size}:");
        for (i, cluster) in clusters.iter().enumerate() {
            let cluster_pairs = cluster
                .keywords
                .par_iter()
                .map(|k| keyword_file_pairs[k].len())
                .sum::<usize>();

            println!("Cluster {i}: ");
            println!("Keywords: {}", cluster.keywords.len());
            let formatted_min_freq = cluster.min_freq.to_formatted_string(&Locale::en);
            let formatted_max_freq = cluster.max_freq.to_formatted_string(&Locale::en);
            println!("Frequency range: {formatted_min_freq}-{formatted_max_freq}");
            println!("Average frequency: {:.1}", cluster.avg_freq);
            println!("Scaled threshold: {:?}", cluster.threshold);
            println!(
                "Total pairs: {}",
                cluster_pairs.to_formatted_string(&Locale::en)
            );
        }

        // 为每个文档生成属性
        let mut doc_properties = AHashMap::new();
        let mut rng = ChaCha20Rng::from_os_rng();
        for docs in keyword_file_pairs.values() {
            for doc_id in docs {
                let access_level = self.access_levels.choose(&mut rng).cloned().unwrap_or(1);
                let state = self.states.choose(&mut rng).cloned().unwrap_or(1);

                doc_properties.insert(doc_id.clone(), (access_level, state));
            }
        }

        // 重组关键词/文档对映射
        let mut keyword_docs = AHashMap::with_capacity(keyword_file_pairs.len());
        for (keyword, doc_ids) in keyword_file_pairs {
            let mut docs = Vec::new();
            for doc_id in doc_ids {
                if let Some((access_level, state)) = doc_properties.get(doc_id).cloned() {
                    // docs.push(KeywordDocPair {
                    //     doc_id: doc_id.clone(),
                    //     access_level,
                    //     state,
                    // });
                    docs.push(KeywordDocPair::new(doc_id, access_level, state));
                }
            }
            keyword_docs.insert(keyword.clone(), docs);
        }

        // 将clusters转换为HashMap<usize, ClusterInfo>
        let clusters = clusters.into_iter().enumerate().collect();

        Ok((keyword_docs, clusters))
    }

    fn save_binary_dataset(
        &self,
        keyword_docs: &AHashMap<String, Vec<KeywordDocPair>>,
        clusters: &AHashMap<usize, ClusterInfo>,
        output_dir: &Path,
        min_cluster_size: usize,
    ) -> Result<()> {
        // 实现二进制文件保存逻辑
        // let cluster_dir = format!("{}/cluster_{}", output_dir, cluster_size);
        let cluster_dir = output_dir.join(format!("cluster_{min_cluster_size}"));
        fs::create_dir_all(&cluster_dir)?;

        // 1. 创建关键词到ID的映射
        let mut all_keywords: Vec<_> = keyword_docs.keys().cloned().collect();
        all_keywords.par_sort_unstable();

        let keyword_to_id: HashMap<_, _> = all_keywords
            .par_iter()
            .enumerate()
            .map(|(idx, kw)| (kw.clone(), idx as u64))
            .collect();

        // 2. 保存元数据
        let metadata_path = Path::new(&cluster_dir).join("metadata.bin");
        let mut metadata_file = File::create(metadata_path)?;
        let mut metadata_file_content = Vec::new();
        let total_pairs = keyword_docs
            .values()
            .par_bridge()
            .map(|v| v.len())
            .sum::<usize>() as u64;

        metadata_file_content.write_u64::<LittleEndian>(total_pairs)?; // 总关键词/文档对数
        metadata_file_content.write_u64::<LittleEndian>(all_keywords.len() as u64)?; // 关键词数量
        metadata_file_content.write_u64::<LittleEndian>(min_cluster_size as u64)?; // 簇大小
        metadata_file.write_all(&metadata_file_content)?;

        // 3. 保存关键词列表
        let keywords_path = Path::new(&cluster_dir).join("keywords.bin");
        let mut keywords_file = File::create(keywords_path)?;
        let mut keywords_file_content = Vec::new();
        keywords_file_content.write_u64::<LittleEndian>(all_keywords.len() as u64)?;

        for keyword in &all_keywords {
            let encoded = keyword.as_bytes();
            keywords_file_content.write_u64::<LittleEndian>(encoded.len() as u64)?;
            keywords_file_content.write_all(encoded)?;
        }
        keywords_file.write_all(&keywords_file_content)?;

        // 4. 保存簇信息
        let clusters_path = Path::new(&cluster_dir).join("clusters.bin");
        let mut clusters_file = File::create(clusters_path)?;
        let mut clusters_file_content = Vec::new();
        clusters_file_content.write_u64::<LittleEndian>(clusters.len() as u64)?;

        for (_cluster_id, cluster) in clusters {
            let keyword_ids: Vec<_> = cluster
                .keywords
                .par_iter()
                .filter_map(|kw| keyword_to_id.get(kw).copied())
                .collect();

            clusters_file_content.write_u64::<LittleEndian>(keyword_ids.len() as u64)?;
            for &kid in &keyword_ids {
                clusters_file_content.write_u64::<LittleEndian>(kid)?;
            }

            clusters_file_content.write_u64::<LittleEndian>(cluster.min_freq as u64)?;
            clusters_file_content.write_u64::<LittleEndian>(cluster.max_freq as u64)?;
            clusters_file_content.write_f64::<LittleEndian>(cluster.avg_freq)?;
            clusters_file_content.write_u64::<LittleEndian>(cluster.threshold as u64)?;
        }
        clusters_file.write_all(&clusters_file_content)?;

        // 5. 保存关键词/文档对数据
        let pairs_path = Path::new(&cluster_dir).join("keyword_doc_pairs.bin");
        let mut pairs_file = File::create(pairs_path)?;
        let mut pairs_file_content = Vec::new();
        pairs_file_content.write_u64::<LittleEndian>(total_pairs)?;

        for (keyword, docs) in keyword_docs {
            if let Some(kid) = keyword_to_id.get(keyword) {
                for KeywordDocPair {
                    doc_id,
                    access_level,
                    state,
                } in docs
                {
                    pairs_file_content.write_u64::<LittleEndian>(*kid)?;
                    let doc_id_bytes = doc_id.as_bytes();
                    pairs_file_content.write_u64::<LittleEndian>(doc_id_bytes.len() as u64)?;
                    pairs_file_content.write_all(doc_id_bytes)?;
                    pairs_file_content.write_u8(*access_level)?;
                    pairs_file_content.write_u8(*state)?;
                }
            }
        }
        pairs_file.write_all(&pairs_file_content)?;

        println!("\nDataset saved to {}", cluster_dir.display());
        println!("Total keyword-document pairs: {total_pairs:?}");
        println!("Total keywords: {:?}", all_keywords.len());

        Ok(())
    }

    fn generate_statistics_report(
        &self,
        keyword_counts: &AHashMap<String, usize>,
        keyword_file_pairs: &AHashMap<String, HashSet<String, RandomState>>, // FIXME: change to `AHashSet` when `AHashSet` is useable in this content
        output_dir: Option<String>,
    ) -> Result<()> {
        // 实现统计报告生成逻辑

        let total_unique_keywords = keyword_counts.len();

        let total_keyword_doc_pairs = keyword_file_pairs
            .values()
            .par_bridge()
            .map(|docs| docs.len())
            .sum::<usize>();

        let (min_keyword_frequency, max_keyword_frequency) = keyword_counts
            .values()
            .fold(None, |acc, &count| match acc {
                None => Some((count, count)),
                Some((min_val, max_val)) => Some((min_val.min(count), max_val.max(count))),
            })
            .unwrap_or((0, 0));

        println!("\nDataset Statistics:");
        println!(
            "Total unique keywords: {}",
            total_unique_keywords.to_formatted_string(&Locale::en)
        );
        println!(
            "Total keyword-document pairs: {}",
            total_keyword_doc_pairs.to_formatted_string(&Locale::en)
        );
        println!(
            "Keyword frequency range: {} - {}",
            min_keyword_frequency.to_formatted_string(&Locale::en),
            max_keyword_frequency.to_formatted_string(&Locale::en)
        );
        if let Some(output_dir) = output_dir {
            // 可选：将统计信息保存到文件
            let stats_path = format!("{output_dir}/statistics.txt");
            let mut stats_file = File::create(stats_path)?;
            writeln!(stats_file, "Total unique keywords: {total_unique_keywords}")?;
            writeln!(
                stats_file,
                "Total keyword-document pairs: {total_keyword_doc_pairs}"
            )?;
            writeln!(
                stats_file,
                "Keyword frequency range: {min_keyword_frequency} - {max_keyword_frequency}"
            )?;
        }

        Ok(())
        //unimplemented!("Statistics report generation logic needs to be implemented")
    }
}

// 新增的辅助函数
fn create_cluster_info(
    current_cluster: &mut Vec<String>,
    keyword_freqs: &HashMap<String, usize>,
    is_last: bool,
) -> ClusterInfo {
    let cluster_freqs: Vec<_> = current_cluster.iter().map(|k| keyword_freqs[k]).collect();

    let avg_freq = cluster_freqs.iter().sum::<usize>() as f64 / cluster_freqs.len() as f64;
    let max_freq = *cluster_freqs.iter().max().unwrap_or(&0);
    let min_freq = *cluster_freqs.iter().min().unwrap_or(&0);

    ClusterInfo::new(
        std::mem::take(current_cluster),
        avg_freq,
        max_freq,
        min_freq,
        if is_last {
            (avg_freq / 10.0) as usize
        } else {
            (avg_freq / 2.0) as usize
        },
    )
}
fn extract_tar_gz(input_path: &Path, output_path: &Path) -> Result<(), anyhow::Error> {
    let file = File::open(input_path)?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);
    archive.unpack(output_path)?;
    Ok(())
}
