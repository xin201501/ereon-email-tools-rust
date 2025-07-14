use anyhow::{Result, anyhow};
use std::collections::HashMap;
use tools_rust::{ClusterInfo, EnronDataProcessor};
fn main() -> Result<()> {
    let _processor = EnronDataProcessor::default();
    //processor.process_dataset();

    // 示例用法
    let _keyword_docs: HashMap<String, Vec<(String, u64)>> = HashMap::new();
    let clusters: HashMap<i32, ClusterInfo> = HashMap::new();
    let _min_cluster_size = 10;

    let _ = clusters
        .iter()
        .next()
        // transform option to result
        .ok_or(anyhow!("No clusters found"))?
        .1;

    // .save_cluster_info(&keyword_docs, &clusters, min_cluster_size)?;

    Ok(())
}
