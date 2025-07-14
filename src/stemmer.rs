use ahash::RandomState;
use rust_stemmers::{Algorithm, Stemmer};
use scc::HashMap;
use std::sync::LazyLock;

pub fn get_stem(word: &str) -> String {
    static STEMMER: LazyLock<Stemmer> = LazyLock::new(|| Stemmer::create(Algorithm::English));
    static STEM_CACHE: LazyLock<HashMap<String, String, RandomState>> =
        LazyLock::new(HashMap::default);
    let word = word.to_lowercase();
    STEM_CACHE
        .entry(word.clone())
        .or_insert(STEMMER.stem(&word).to_string())
        .clone()
}
