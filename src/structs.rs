use crate::ClusterInfo;
use ahash::AHashMap;
use anyhow::Result;
use serde::{Deserialize, Serialize};
pub struct _Cluster {
    pub keywords: Vec<String>,
    pub min_freq: u64,
    pub max_freq: u64,
    pub avg_freq: f64,
    pub threshold: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DatasetMetadata {
    total_pairs: usize,
    keyword_count: usize,
    cluster_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordDocPair {
    pub(crate) doc_id: String,
    pub(crate) access_level: u8,
    pub(crate) state: u8,
}

impl KeywordDocPair {
    pub fn new(doc_id: impl Into<String>, access_level: u8, state: u8) -> Self {
        Self {
            doc_id: doc_id.into(),
            access_level,
            state,
        }
    }
}

pub(crate) type BinaryDatasetResult = Result<(
    AHashMap<String, Vec<KeywordDocPair>>,
    AHashMap<usize, ClusterInfo>,
)>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordInfo {
    pub keyword: String,
    pub access_level: u8,
    pub state: u8,
}
pub(crate) type ForwardIndex = AHashMap<String, Vec<KeywordInfo>>;
