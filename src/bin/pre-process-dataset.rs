use std::path::{Path, PathBuf};

use anyhow::Result;
use log::info;
use pyroscope::PyroscopeAgent;
use pyroscope_pprofrs::{PprofConfig, pprof_backend};
use tools_rust::EnronDataProcessor;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() -> Result<()> {
    env_logger::init();
    // get project directory
    let _project_dir = std::env::current_dir()?;
    unsafe { std::env::set_var("INPUT_DIR", "/tmp/enron_mail.tar.gz") };
    // parse environment variables `INPUT_DIR`, `OUTPUT_DIR`
    let input_file = std::env::var("INPUT_DIR")
        .unwrap_or("/home/c306/multiLevelSgx/data/raw/enron_mail.tar.gz".to_string());
    let input_file = Path::new(&input_file);

    info!("input_dir: {}", input_file.display());
    let custom_output_dir = std::env::var("OUTPUT_DIR");
    let output_dir = match custom_output_dir {
        Ok(dir) => PathBuf::new().join(dir),
        Err(_) => "/tmp".into(),
    };
    let output_dir = output_dir.join("data").join("enron").join("processed");

    info!("output_dir: {}", output_dir.display());
    // let agent = PyroscopeAgent::builder("http://localhost:4040", "myapp-profile")
    //     .backend(pprof_backend(PprofConfig::new().sample_rate(100)))
    //     .build()?;
    // let agent_running = agent.start()?;

    let processor = EnronDataProcessor::default();
    processor.process_dataset(input_file, &output_dir)?;

    // let agent_ready = agent_running.stop()?;
    // agent_ready.shutdown();
    Ok(())
}
