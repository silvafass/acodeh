use anyhow::anyhow;
use futures::stream::StreamExt;
use serde::{Deserialize, Serialize};

const DEFAULT_API_URL: &str = "http://localhost:11434/api/generate";

#[derive(Debug, Serialize)]
pub struct GeneratePayload {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<ModelParameters>,
}

#[derive(Debug, Serialize)]
pub struct ModelParameters {
    pub num_ctx: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct GenerateResponse {
    pub created_at: String,
    pub done_reason: String,
    pub done: bool,
    pub eval_count: u64,
    pub eval_duration: u64,
    pub load_duration: u64,
    pub model: String,
    pub prompt_eval_count: u64,
    pub prompt_eval_duration: u64,
    pub response: String,
    pub total_duration: u64,
    pub error: Option<String>,
}

pub struct LLMClient {
    api_url: String,
    client: reqwest::Client,
}

impl Default for LLMClient {
    fn default() -> Self {
        Self::new(DEFAULT_API_URL)
    }
}

impl LLMClient {
    pub fn new(api_url: &str) -> Self {
        Self {
            api_url: api_url.to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub async fn generate_stream<F, Fut>(
        &self,
        payload: &GeneratePayload,
        mut on_chunk: F,
    ) -> anyhow::Result<()>
    where
        F: FnMut(GenerateResponse) -> Fut + Send,
        Fut: std::future::Future<Output = bool> + Send,
    {
        let response = self
            .client
            .post(&self.api_url)
            .json(&payload)
            .send()
            .await?;

        if response.error_for_status_ref().is_err() {
            let error_response: serde_json::Value = response.json().await?;
            return Err(anyhow!("API error: {error_response}"));
        } else {
            let mut stream = response.bytes_stream();
            let mut no_parsed_chunks: Vec<u8> = vec![];
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                if let Ok(chunk) = serde_json::from_slice::<GenerateResponse>(&chunk) {
                    if let Some(err) = chunk.error {
                        return Err(anyhow!("LLM error: {err}"));
                    }

                    let stop = on_chunk(chunk).await;
                    if stop {
                        break;
                    }
                } else {
                    no_parsed_chunks = [no_parsed_chunks, chunk.to_vec()].concat();
                }
            }
            if !no_parsed_chunks.is_empty() {
                let chunk = serde_json::from_slice::<GenerateResponse>(&no_parsed_chunks)?;
                on_chunk(chunk).await;
            }
        }

        Ok(())
    }

    pub async fn generate_once(
        &self,
        payload: &GeneratePayload,
    ) -> anyhow::Result<GenerateResponse> {
        let response = self
            .client
            .post(&self.api_url)
            .json(&payload)
            .send()
            .await?;

        if response.error_for_status_ref().is_err() {
            let error_response: serde_json::Value = response.json().await?;
            return Err(anyhow!("API error: {error_response}"));
        }

        let generated: GenerateResponse = response.json().await?;
        Ok(generated)
    }
}
