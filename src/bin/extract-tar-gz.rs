use flate2::read::GzDecoder;
use std::fs::File;
use tar::Archive;

fn extract_tar_gz(input_path: &str, output_dir: &str) -> Result<(), anyhow::Error> {
    let file = File::open(input_path)?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);
    archive.unpack(output_dir)?;
    Ok(())
}

fn main() -> Result<(), anyhow::Error> {
    // 示例调用：解压文件到指定目录
    extract_tar_gz("example.tar.gz", "output_directory")?;
    Ok(())
}
