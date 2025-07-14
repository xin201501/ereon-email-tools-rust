use ahash::AHashMap;
use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::path::Path;

use crate::structs::KeywordDocPair;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterInfo {
    pub(crate) keywords: Vec<String>,
    pub(crate) avg_freq: f64,
    pub(crate) max_freq: usize,
    pub(crate) min_freq: usize,
    pub(crate) threshold: usize,
}

impl ClusterInfo {
    pub fn new(
        keywords: Vec<String>,
        avg_freq: f64,
        max_freq: usize,
        min_freq: usize,
        threshold: usize,
    ) -> Self {
        Self {
            keywords,
            avg_freq,
            max_freq,
            min_freq,
            threshold,
        }
    }
}

impl ClusterInfo {
    pub(crate) fn save_cluster_info(
        keyword_docs: &AHashMap<String, Vec<KeywordDocPair>>,
        clusters: &AHashMap<usize, ClusterInfo>,
        min_cluster_size: usize,
        output_dir: &Path,
    ) -> Result<()> {
        let output_dir = output_dir.join(format!("cluster_{min_cluster_size}"));
        fs::create_dir_all(&output_dir)?;

        // 收集数据
        let mut cluster_data = Vec::new();
        let mut thresholds = Vec::new();
        let mut cluster_ids = Vec::new();

        for (cluster_id, cluster) in clusters {
            let _cluster_pairs = cluster
                .keywords
                .iter()
                .map(|kw| keyword_docs.get(kw).map_or(0, |docs| docs.len() as u64))
                .sum::<u64>();

            let cluster_info = ClusterInfo {
                min_freq: cluster.min_freq,
                max_freq: cluster.max_freq,
                avg_freq: cluster.avg_freq,
                threshold: cluster.threshold,
                keywords: cluster.keywords.clone(),
            };

            cluster_data.push(cluster_info);
            thresholds.push(cluster.threshold);
            cluster_ids.push(*cluster_id);
        }

        // 按平均频率排序
        cluster_data.par_sort_by(|a, b| a.avg_freq.partial_cmp(&b.avg_freq).unwrap());
        let _sorted_thresholds: Vec<_> = cluster_data.iter().map(|c| c.threshold).collect();
        // let sorted_ids: Vec<i32> = cluster_data.iter().map(|c| c.id).collect();

        // 保存JSON文件
        let json_path = output_dir.join("cluster_info.json");
        let file = File::create(json_path)?;
        serde_json::to_writer_pretty(file, &cluster_data)?;

        Ok(())
    }
}
