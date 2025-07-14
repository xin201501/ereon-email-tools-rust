use anyhow::Result;
use rayon::prelude::*;
use std::fs::File;
use std::io::Write;
use std::{env, fs};
use tools_rust::DatasetLoader;
fn main() -> Result<()> {
    // load `DATASET_PATH`
    let dataset_path = env::var("DATASET_PATH").unwrap_or("/tmp/data/enron/processed".to_string());
    let output_dir =
        env::var("OUTPUT_DIR").unwrap_or("/tmp/data/enron/processed/single_cluster".to_string());
    let loader = DatasetLoader::new(dataset_path);
    let forward_index = loader.convert_single_cluster_to_forward_index(3, 1)?;
    // println!("{forward_index:?}");
    // let file = File::create("forward_index.json")?;
    // serde_json::to_writer_pretty(file, &forward_index)?;
    fs::create_dir_all(&output_dir)?;

    forward_index
        .into_values()
        .enumerate()
        .try_for_each(|(i, keywords)| -> Result<()> {
            let file_content: Vec<_> = keywords
                .par_iter()
                .map(|keyword| keyword.keyword.clone())
                .collect();
            let file_content = file_content.join(",");
            let mut file = File::create(format!("{output_dir}/{}", i + 1))?;
            file.write_all(file_content.as_bytes())?;
            Ok(())
        })?;
    Ok(())
}
