use arkhe_eval::{EvalEngine, EvalConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = EvalConfig::default();
    let engine = EvalEngine::new(config);
    let report = engine.evaluate().await?;
    println!("{}", report.to_json()?);
    Ok(())
}
