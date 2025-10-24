use acodeh::fs::FileSearcher;
use futures::stream::StreamExt;
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use serde::{Deserialize, Serialize};
use std::io::{self, Write};
use std::path::PathBuf;

use clap::Parser;

const DEFAULT_MAX_CONTEXT: u64 = 16 * 1_024;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
enum Command {
    Run {
        prompt: String,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        path: Vec<PathBuf>,
        #[arg(long)]
        includes: Vec<PathBuf>,
        #[arg(long)]
        excludes: Vec<PathBuf>,
        #[arg(long)]
        extensions: Option<String>,
        #[arg(long)]
        overall: bool,
        #[arg(short, long)]
        recursive: bool,
        #[arg(long, default_value_t = 1)]
        max_depth: usize,
        #[arg(long)]
        max_context: Option<u64>,
        #[arg(long, default_value_t = false)]
        debug: bool,
    },
}

#[derive(Debug, Default, Serialize)]
struct ModelParameters {
    num_ctx: Option<u64>,
}

#[derive(Debug, Default, Serialize)]
struct GeneratePayload {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<ModelParameters>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct GenerateResponse {
    created_at: String,
    done_reason: String,
    done: bool,
    eval_count: u64,
    eval_duration: u64,
    load_duration: u64,
    model: String,
    prompt_eval_count: u64,
    prompt_eval_duration: u64,
    response: String,
    total_duration: u64,
    error: Option<String>,
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let command = Command::parse();

    match command {
        Command::Run {
            model,
            prompt,
            path,
            includes,
            excludes,
            extensions,
            overall,
            recursive,
            max_depth,
            max_context,
            debug,
        } => {
            let max_depth = if recursive { usize::MAX } else { max_depth };

            let paths_iter = path
                .iter()
                .flat_map(|start_path| {
                    let mut ignore_build = GitignoreBuilder::new(start_path);
                    if let Err(error) = ignore_build.add_line(None, ".git")
                        && debug
                    {
                        eprintln!("Could not add .git to ignore: {error:?}");
                    }

                    let mut ignore = match ignore_build.build() {
                        Ok(ignore) => ignore,
                        Err(error) => {
                            if debug {
                                eprintln!("Failed to build ignore patterns: {error:?}");
                                println!("Using a empty ignore pattern...");
                            }
                            Gitignore::empty()
                        }
                    };

                    FileSearcher::new(start_path)
                        .overall(overall)
                        .max_depth(max_depth)
                        .includes(&includes)
                        .excludes(&excludes)
                        .extensions(extensions.as_ref())
                        .into_iter()
                        .filter_path(move |path| {
                            if path.ends_with(".gitignore") {
                                if let Some(error) = ignore_build.add(path)
                                    && debug
                                {
                                    eprintln!("ERROR: {}", error);
                                }
                                ignore = match ignore_build.build() {
                                    Ok(ignore) => ignore,
                                    Err(error) => {
                                        if debug {
                                            eprintln!("Failed to build ignore patterns: {error:?}");
                                            println!("Using a empty ignore pattern...");
                                        }
                                        Gitignore::empty()
                                    }
                                };
                            }
                            ignore.matched(path, path.is_dir()).is_none()
                        })
                        .filter_map(|result| result.ok())
                })
                .filter(|path| path.is_file());

            let mut total_content_len = 0;
            let mut documents = vec![];
            for path in paths_iter {
                let extension = path
                    .extension()
                    .map(|extension| extension.to_string_lossy().to_string())
                    .unwrap_or_default();
                let path_as_string = path.to_string_lossy().to_string();

                let content = if path.extension().unwrap_or_default().to_string_lossy() == "pdf" {
                    pdf_extract::extract_text(&path)
                        .inspect_err(|error| {
                            if debug {
                                eprintln!("{error:?}\nwhile reading {path_as_string}");
                            }
                        })
                        .ok()
                } else {
                    tokio::fs::read_to_string(&path)
                        .await
                        .inspect_err(|error| {
                            if debug {
                                eprintln!("{error:?}\nwhile reading {path_as_string}");
                            }
                        })
                        .ok()
                };

                if let Some(content) = content {
                    let content = format!(
                        "path: {}\n```{}\n{}\n```",
                        path_as_string, extension, content
                    );

                    if let Some(max_context) = max_context.or(Some(DEFAULT_MAX_CONTEXT))
                        && (total_content_len + content.len() as u64) / 4 > max_context
                    {
                        break;
                    }
                    total_content_len += content.len() as u64;

                    documents.push((path, content));
                }
            }

            let document_count = documents.len();
            let context = documents
                .into_iter()
                .inspect(|(path, ..)| {
                    if debug {
                        println!("Adding file {path:?}");
                    }
                })
                .map(|(.., content)| content)
                .reduce(|acc, content| format!("{acc}\n{content}"))
                .unwrap_or_default();
            let prompt = format!(include_str!("prompt_templete.in"), prompt, context);

            let context_len_estimated = total_content_len / 4;
            let prompt_context_len_estimated: u64 = prompt.len() as u64 / 4;

            let max_context = match max_context {
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

            println!(
                concat!(
                    "Total of documents {},\n",
                    "Total of documents contents: {},\n",
                    "Documents context size: {}\n",
                    "Prompt context size: {}\n",
                    "Max Context size: {}"
                ),
                document_count,
                total_content_len,
                context_len_estimated,
                prompt_context_len_estimated,
                max_context,
            );

            let client = reqwest::Client::new();

            let is_stream = true;
            let payload = GeneratePayload {
                model: model.unwrap_or("llama3.2:latest".to_string()),
                system: Some(include_str!("system.in").to_string()),
                prompt: Some(prompt),
                stream: Some(is_stream),
                options: Some(ModelParameters {
                    num_ctx: Some(max_context),
                }),
            };

            let response = client
                .post("http://localhost:11434/api/generate")
                .json(&payload)
                .send()
                .await?;

            if let Err(error) = response.error_for_status_ref() {
                dbg!(error);
                let error_response: serde_json::Value = response.json().await?;
                eprintln!("ERROR: {}", error_response.get("error").unwrap_or_default());
            } else if is_stream {
                let mut stream = response.bytes_stream();
                println!("\n");
                let mut stdout = io::stdout();
                let mut no_parsed_chunks: Vec<u8> = vec![];
                while let Some(chunk) = stream.next().await {
                    let chunk = chunk.unwrap();
                    if let Ok(chunk) = serde_json::from_slice::<GenerateResponse>(&chunk) {
                        if chunk.error.is_some() {
                            eprintln!("ERROR: {}", chunk.error.unwrap_or_default());
                            break;
                        }

                        write!(stdout, "{}", &chunk.response)?;
                        stdout.flush()?;

                        if chunk.done {
                            println!("\n");
                            dbg!(chunk);
                        }
                    } else {
                        no_parsed_chunks = [no_parsed_chunks, chunk.to_vec()].concat();
                    }
                }
                if !no_parsed_chunks.is_empty() {
                    let chunk = serde_json::from_slice::<GenerateResponse>(&no_parsed_chunks)?;
                    println!("\n");
                    dbg!(chunk);
                }
            } else {
                let generated: GenerateResponse = response.json().await?;
                println!("{}", generated.response);
                println!("\n");
                dbg!(generated);
            }
        }
    }

    Ok(())
}
