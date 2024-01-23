use futures::StreamExt;
use ollama_rs::generation::completion::{GenerationContext, GenerationResponse};
use ollama_rs::generation::completion::request::GenerationRequest;
use xp_ollama::{DEFAULT_SYSTEM_MOCK, MODEL, Result};
use ollama_rs::Ollama;
use simple_fs::{ensure_file_dir, save_json};
use tokio::io::AsyncWriteExt;
use xp_ollama::gen::gen_stream_print;


#[tokio::main]
async fn main() -> Result<() > {
  let ollama = Ollama::default();

  let prompts = &[
    "Why is the sky red?",
    "What was my first question?",
  ];

  let mut last_ctx: Option<GenerationContext> = None;

  for prompt in prompts {
    println!("->> prompt: {prompt}");
    let mut gen_req = GenerationRequest::new(MODEL.to_string(), prompt.to_string());

    if let Some(last_ctx) = last_ctx.take() {
      gen_req = gen_req.context(last_ctx);
    }

    let final_data = gen_stream_print(&ollama, gen_req).await?;

    if let Some(final_data) = final_data {
      last_ctx = Some(final_data.context);

      let ctx_file_path = ".c02_context/ctx.json";
      ensure_file_dir(ctx_file_path)?;
      save_json(ctx_file_path, &last_ctx)?;
    }
  }

  Ok(())
}

