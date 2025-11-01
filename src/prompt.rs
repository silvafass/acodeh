use anyhow::{Ok, anyhow};
use std::path::PathBuf;

const DEFAULT_MAX_CONTEXT: u64 = 16 * 1_024;

#[derive(Debug)]
pub struct PromptStats {
    pub file_count: usize,
    pub document_count: usize,
    pub total_content_len: u64,
    pub context_len_estimated: u64,
    pub prompt_context_len_estimated: u64,
    pub max_context: u64,
}

pub struct PromptBuilder {
    prompt: String,
    files: Vec<(PathBuf, String)>,
    documents: Vec<String>,
    total_content_len: u64,
    max_context: Option<u64>,
}

impl PromptBuilder {
    pub fn new(prompt: String) -> Self {
        Self {
            prompt,
            files: vec![],
            documents: vec![],
            total_content_len: 0,
            max_context: None,
        }
    }

    pub fn max_context(mut self, value: Option<u64>) -> Self {
        self.max_context = value;
        self
    }

    pub async fn add_file(&mut self, path: PathBuf) -> anyhow::Result<u64> {
        let extension = path
            .extension()
            .map(|extension| extension.to_string_lossy().to_string())
            .unwrap_or_default();
        let path_as_string = path.to_string_lossy().to_string();

        let content = if extension == "pdf" {
            pdf_extract::extract_text(&path)?
        } else {
            tokio::fs::read_to_string(&path).await?
        };

        let content = format!(
            "path: {}\n```{}\n{}\n```",
            path_as_string, extension, content
        );

        let content_len = content.len() as u64;
        if let Some(max_context) = self.max_context.or(Some(DEFAULT_MAX_CONTEXT))
            && (self.total_content_len + content_len) / 4 > max_context
        {
            return Err(anyhow!(
                "Maximum context exceeded ({max_context:?}) while adding {path_as_string} ({content_len}b)"
            ));
        }
        self.total_content_len += content_len;

        self.files.push((path, content));

        Ok(content_len)
    }

    pub fn add_document(&mut self, content: String) -> anyhow::Result<u64> {
        let content_len = content.len() as u64;
        if let Some(max_context) = self.max_context.or(Some(DEFAULT_MAX_CONTEXT))
            && (self.total_content_len + content_len) / 4 > max_context
        {
            return Err(anyhow!(
                "Maximum context exceeded {max_context:?} while adding document ({content_len}b)"
            ));
        }
        self.total_content_len += content_len;

        self.documents.push(content);

        Ok(content_len)
    }

    pub fn files(&self) -> &Vec<(PathBuf, String)> {
        &self.files
    }

    pub fn documents(&self) -> &Vec<String> {
        &self.documents
    }

    pub fn build(&self) -> anyhow::Result<(String, PromptStats)> {
        let mut context: Vec<String> = vec![];

        if !self.files.is_empty() {
            context.push(format!(
                "<files>\n{}\n</files>",
                self.files
                    .iter()
                    .fold(String::new(), |acc, (.., content)| format!(
                        "{acc}\n{content}"
                    ))
            ));
        }
        if !self.documents.is_empty() {
            context.push(format!(
                "<documents>\n{}\n</documents>",
                self.documents.join("\n")
            ));
        }

        let prompt;
        let prompt_context_len_estimated;
        if context.is_empty() {
            prompt = self.prompt.clone();
            prompt_context_len_estimated = prompt.len() as u64 / 4;
        } else {
            prompt = [
                self.prompt.clone(),
                format!(
                    include_str!("prompt_context_templete.in"),
                    context.join("\n")
                ),
            ]
            .join("\n");
            prompt_context_len_estimated = prompt.len() as u64 / 4;
        }

        let max_context = match self.max_context {
            Some(max_context) => max_context,
            _ if prompt_context_len_estimated > DEFAULT_MAX_CONTEXT => DEFAULT_MAX_CONTEXT,
            _ => {
                let mut aligned_context_len = 2 * 1024;
                while aligned_context_len < prompt_context_len_estimated {
                    aligned_context_len *= 2;
                }

                if aligned_context_len > DEFAULT_MAX_CONTEXT {
                    DEFAULT_MAX_CONTEXT
                } else {
                    aligned_context_len
                }
            }
        };

        Ok((
            prompt,
            PromptStats {
                file_count: self.files.len(),
                document_count: self.documents.len(),
                total_content_len: self.total_content_len,
                context_len_estimated: self.total_content_len / 4,
                prompt_context_len_estimated,
                max_context,
            },
        ))
    }
}
