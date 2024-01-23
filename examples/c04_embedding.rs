use std::fs;
use std::fs::read_to_string;
use std::path::Path;
use xp_ollama::Result;
use ollama_rs::Ollama;
use simple_fs::{ensure_dir, ensure_file_dir, save_be_f64, save_json};
use xp_ollama::consts::MODEL;

const MOCK_DIR: &str = "_mock_data";
const C04_DIR: &str = "c04_data";

#[tokio::main]
async fn main() -> Result<()> {
  let ollama = Ollama::default();

  ensure_dir(C04_DIR)?;
  let txt = read_to_string(Path::new(MOCK_DIR).join("for_embeddings.txt"))?;

  let splits = simple_text_splitter(&txt, 500)?;
  println!("->>      splits count: {}", splits.len());

  for (i, seg) in splits.into_iter().enumerate() {
    println!();
    let file_name = format!("c04_embeddings_{:0>2}.txt", i);
    let path = Path::new(C04_DIR).join(file_name);
    fs::write(path, &seg);

    println!("->>      text length: {}", txt.len());
    let res = ollama.generate_embeddings(MODEL.to_string(), seg, None).await?;

    println!("->> embeddings length: {}", res.embeddings.len());

    let file_name = format!("c04_emeddings_{:0>2}.json", i);
    save_json(Path::new(C04_DIR).join(file_name), &res.embeddings)?;

    let file_name = format!("c04_embeddings_{:0>2}.be-f64.bin", i);
    let file_path = Path::new(C04_DIR).join(file_name);
    save_be_f64(file_path, &res.embeddings)?;
  }

  Ok(())
}

fn simple_text_splitter(txt: &str, num: u32) -> Result<Vec<String>> {
  let mut result = Vec::new();
  let mut last = 0;
  let mut count = 0;
  for (idx, _) in txt.char_indices() {
    count += 1;
    if count == num {
      result.push(&txt[last..idx + 1]);
      last = idx + 1;
      count = 0;
    }
  }
  // Handle any remaining characters
  if last < txt.len() {
    result.push(&txt[last..]);
  }
  Ok(result.into_iter().map(String::from).collect())
}