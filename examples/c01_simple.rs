use ollama_rs::generation::completion::request::GenerationRequest;
use xp_ollama::{DEFAULT_SYSTEM_MOCK, MODEL, Result};
use ollama_rs::Ollama;
use xp_ollama::gen::gen_stream_print;

#[tokio::main]
async fn main() -> Result<() > {
  let ollama = Ollama::default();

  let model = MODEL.to_string();
  let prompt: String = "What is the best programming language?".to_string();

  let gen_req: GenerationRequest = GenerationRequest::new(model, prompt)
    .system(DEFAULT_SYSTEM_MOCK.to_string());

  gen_stream_print(&ollama, gen_req).await?;

  Ok(())
}

